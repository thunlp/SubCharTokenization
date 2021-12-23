"""
@name = 'roberta_wwm_ext_large'
@author = 'zhangxinrui'
@time = '2019/11/15'
roberta_wwm_ext_large 的baseline版本

coding=utf-8
Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

import argparse
import os
import pickle
import random
import logging
import json
from shutil import copyfile

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, IterableDataset
from tqdm import tqdm

import consts
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import modeling
from tokenization import (
    ALL_TOKENIZERS, 
    BertTokenizer, 
    ConcatSepTokenizer, 
    WubiZhTokenizer, 
    RawZhTokenizer, 
    BertZhTokenizer,
)
from optimization import BertAdam, warmup_linear, get_optimizer
from schedulers import LinearWarmUpScheduler
from utils import (
    mkdir, get_freer_gpu, get_device, output_dir_to_tokenizer_name
)

from mrc.preprocess.CHID_preprocess import (
    RawResult, 
    get_final_predictions, 
    write_predictions, 
    evaluate,
    read_chid_examples,
    convert_examples_to_features,
    write_features_json,
)
from mrc.pytorch_modeling import ALBertConfig, ALBertForMultipleChoice
from mrc.pytorch_modeling import BertConfig, BertForMultipleChoice
from run_pretraining import pretraining_dataset, WorkerInitObj

import kara_storage
import utils

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


USE_KARA = True     # 用 kara_storage（以减少内存占用）
# Some fixed settings,
SHUFFLE = True      # 保存 feature 文件前打乱（默认 True 就好）
SPLIT_CHAR = False  # 拆字
ADD_DEF = True      # 加定义


class ChidDataset(IterableDataset):
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'r', encoding='utf8')

    def reset(self):
        self.file.seek(0)

    def seek(self, *args):
        self.file.seek(*args)

    def read(self):
        line = self.file.readline()
        if line == '':
            raise EOFError
        else:
            feature = json.loads(line)
            input_ids = torch.tensor(feature['input_ids'], dtype=torch.long)
            input_masks = torch.tensor(feature['input_masks'], dtype=torch.long)
            segment_ids = torch.tensor(feature['segment_ids'], dtype=torch.long)
            choice_masks = torch.tensor(feature['choice_masks'], dtype=torch.long)
            labels = torch.tensor(feature['label'], dtype=torch.long)
            return input_ids, input_masks, segment_ids, choice_masks, labels


def reset_model(args, bert_config, model_cls):
    # Prepare model
    model = model_cls(bert_config, num_choices=args.max_num_choices)
    if args.init_checkpoint is not None:
        print('load bert weight')
        state_dict = torch.load(args.init_checkpoint, map_location='cpu')
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))

    if args.fp16:
        model.half()

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--vocab_model_file', type=str, required=True)
    parser.add_argument('--init_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tokenizer_type', type=str, required=True)

    ## Other parameters
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--train_ans_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, required=True)
    parser.add_argument('--dev_ans_file', type=str, required=True)

    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--do_eval', action='store_true', default=False)

    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_num_choices", default=10, type=int,
                        help="The maximum number of cadicate answer,  shorter than this will be padded.")
    parser.add_argument("--train_batch_size", default=20, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=16, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.06, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--num_train_epochs", type=int, required=True)
    # parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    return parser.parse_args()


def get_filename_examples_and_features(
    data_type, 
    data_dir, 
    max_seq_length, 
    tokenizer_type, 
    vocab_size):
    '''
    Return:
        file_examples: str,
        file_features: str,
    '''
    file_examples = os.path.join(data_dir, data_type + '_examples.pkl')
    suf = '_{}_{}_{}'.format(str(max_seq_length), tokenizer_type, str(vocab_size))
    suf2 = ''
    if data_type == 'train' and USE_KARA and SHUFFLE:
        suf2 += '_shuffled'
    if not SPLIT_CHAR:
        suf2 += '_whole'
    if ADD_DEF:
        suf2 += '_def'
    if data_type == 'train':
        if USE_KARA:
            file_features = os.path.join(data_dir, data_type + '_features' + suf + suf2 + '.json')
        else:
            file_features = os.path.join(data_dir, data_type + '_features' + suf + suf2 + '.pkl')
    else:
        # assert not SHUFFLE, 'Prediction features should not be shuffled'
        file_features = os.path.join(data_dir, data_type + '_features' + suf + suf2 + '.pkl')
    return file_examples, file_features, 


def get_example_ids_and_tags(features):
    example_ids = [f.example_id for f in features]
    tags = [f.tag for f in features]
    return example_ids, tags


def expand_features(features, is_training=False):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_masks for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_choice_masks = torch.tensor([f.choice_masks for f in features], dtype=torch.long)
    if is_training:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_masks, all_segment_ids, all_choice_masks, all_labels
    else:
        all_examples_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        return all_input_ids, all_input_masks, all_segment_ids, all_choice_masks, all_example_index


def get_eval_dataloader_and_dataset(features, batch_size):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_masks for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_choice_masks = torch.tensor([f.choice_masks for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids, 
        all_input_masks, 
        all_segment_ids, 
        all_choice_masks,
        all_example_index)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader, dataset


def gen_examples(file_data, file_labels, file_examples, is_training):    
    if os.path.exists(file_examples):
        logger.info('Found example file: ' + file_examples + ', loading...')
        examples = pickle.load(open(file_examples, 'rb'))
        logger.info(f'Loaded {len(examples)} examples')
    else:
        logger.info('Did not find example file, generating...')
        examples = read_chid_examples(file_data, file_labels, is_training=is_training)
        logger.info(f'Loaded {len(examples)} examples, saving...')
        pickle.dump(examples, open(file_examples, 'wb'))
    return examples


def gen_train_features_json(
    examples, 
    file_features, 
    file_idiom_dict,
    tokenizer,
    max_seq_length, 
    max_num_choices,
    is_training=True,
    features_use_json=False,
    split_char=False,
    shuffle=False):
    assert not os.path.exists(file_features), 'JSON features already exists'
    idiom_dict = json.load(open(file_idiom_dict))
    logger.info('Generating train features...')
    features = convert_examples_to_features(
        examples,
        tokenizer,
        idiom_dict,
        max_seq_length,
        max_num_choices,
        split_char=split_char,
        shuffle=shuffle,
        max_def_length=32,
        add_def=ADD_DEF)
    print(features[0])
    logger.info(f'Generated {len(features)} features, saving to "{file_features}"...')
    write_features_json(features, file_features)


def load_or_gen_features(
    examples, 
    file_features, 
    file_idiom_dict,
    tokenizer,
    max_seq_length, 
    max_num_choices,
    is_training=True,
    features_use_json=False,
    split_char=False,
    shuffle=False):
    idiom_dict = json.load(open(file_idiom_dict))
    if os.path.exists(file_features):
        logger.info('Found feature file, loading...')
        features = pickle.load(open(file_features, 'rb'))
        logger.info(f'Loaded {len(features)} features')
    else:
        logger.info('Feature file not found, generating...')
        features = convert_examples_to_features(
            examples,
            tokenizer,
            idiom_dict,
            max_def_length=32,
            max_seq_length=max_seq_length,
            max_num_choices=max_num_choices,
            split_char=split_char,
            shuffle=shuffle,
            add_def=ADD_DEF)
        logger.info(f'Generated {len(features)} features, saving...')
        pickle.dump(features, open(file_features, 'wb'))
    return features


def train(args):
    # Manage output files
    output_dir = os.path.join(args.output_dir, str(args.seed))
    filename_scores = os.path.join(output_dir, 'scores.txt')
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Arguments:")
    logger.info(json.dumps(vars(args), indent=4))
    filename_params = os.path.join(output_dir, 'params.json')
    json.dump(vars(args), open(filename_params, 'w'), indent=4)

    device = get_device()
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(device, n_gpu, args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    # Set seed
    utils.set_seed(args.seed)

    # Prepare model
    logger.info('Preparing model from checkpoint {}'.format(args.init_checkpoint))
    config = modeling.BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = modeling.BertForMultipleChoice(config, args.max_num_choices)
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training

    state_dict = torch.load(args.init_checkpoint, map_location='cpu')['model']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Tokenizer
    logger.info('Loading tokenizer...')
    tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    real_tokenizer_type = output_dir_to_tokenizer_name(args.output_dir)
    logger.info('Loaded tokenizer')

    # Save config
    model_to_save = model.module if hasattr(model, 'module') else model
    filename_config = os.path.join(output_dir, modeling.CONFIG_NAME)
    with open(filename_config, 'w') as f:
        f.write(model_to_save.config.to_json_string())

    # Generate (or load) train features
    file_train_examples, file_train_features = get_filename_examples_and_features(
        'train',
        args.data_dir,
        args.max_seq_length,
        real_tokenizer_type,
        config.vocab_size)
    logger.info('Loading train input...')
    logger.info(f'  file_train_examples: {file_train_examples}')
    logger.info(f'  file_train_features: {file_train_features}')
    file_idiom_dict = os.path.join(args.data_dir, 'idiomDict.json')

    if USE_KARA:
        assert file_train_features[-5:] == '.json', 'kara_storage must load from JSON'
    else:
        assert file_train_features[-4:] == '.pkl', 'Must load from pkl by default'


    if not os.path.exists(file_train_features):
        logger.info('Feature file not found, generating...')
        train_examples = gen_examples(
            args.train_file, 
            args.train_ans_file,
            file_train_examples,
            is_training=True)
        gen_train_features_json(
            train_examples,
            file_train_features,
            file_idiom_dict,
            tokenizer,
            max_seq_length=args.max_seq_length,
            max_num_choices=args.max_num_choices,
            is_training=True,
            features_use_json=USE_KARA,
            split_char=SPLIT_CHAR,
            shuffle=SHUFFLE)
        del train_examples
    
    logger.info('Making ChidDataset from training feature file...')
    chid_dataset = ChidDataset(file_train_features)
    train_data = kara_storage.make_torch_dataset(chid_dataset)
    train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)

    # Load dev data
    dev_example_file, dev_feature_file = get_filename_examples_and_features(
        'dev',
        args.data_dir,
        args.max_seq_length,
        real_tokenizer_type,
        config.vocab_size)

    # Generate (or load) dev features
    logger.info('Loading dev input...')
    logger.info('  dev_example_file: ' + dev_example_file)
    logger.info('  dev_feature_file: ' + dev_feature_file)
    dev_examples = gen_examples(
        args.dev_file,
        None,
        dev_example_file,
        is_training=False,
    )
    eval_features = load_or_gen_features(
        dev_examples, 
        dev_feature_file, 
        file_idiom_dict,
        tokenizer,
        max_seq_length=args.max_seq_length, 
        max_num_choices=args.max_num_choices,
        is_training=False,
        split_char=SPLIT_CHAR)
    
    # NOTE: The following two vars might not hold if you change arguments
    # such as `max_num_choices`
    num_train_steps = 86567
    num_features = 519407
    logger.info("Num generated examples = {}".format(num_features))
    logger.info("Batch size = {}".format(args.train_batch_size))
    logger.info("Num steps for a epoch = {}".format(num_train_steps))

    eval_dataloader, eval_data = get_eval_dataloader_and_dataset(eval_features, args.predict_batch_size)
    all_example_ids, all_tags = get_example_ids_and_tags(eval_features)
    
    # Optimizer
    optimizer = get_optimizer(
        model,
        float16=args.fp16,
        learning_rate=args.learning_rate,
        total_steps=num_train_steps,
        schedule='warmup_linear',
        warmup_rate=args.warmup_proportion,
        weight_decay_rate=0.01,
        max_grad_norm=1.0,
        opt_pooler=True)

    global_step = 0
    best_acc = None
    acc = 0
    
    dev_acc_history = []
    train_loss_history = []

    # Start training and evaluation
    logger.info('***** Training *****')
    logger.info('Number of epochs: ' + str(args.num_train_epochs))
    logger.info('Batch size: ' + str(args.train_batch_size))
    for ep in range(int(args.num_train_epochs)):
        num_step = 0
        total_train_loss = 0
        model.train()
        model.zero_grad()  # 等价于optimizer.zero_grad()
        steps_per_epoch = num_train_steps // args.num_train_epochs
        with tqdm(total=int(steps_per_epoch), desc='Epoch %d' % (ep + 1), mininterval=10.0) as pbar:
            chid_dataset.reset()
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_masks, segment_ids, choice_masks, labels = batch
                if step == 0 and ep == 0:
                    logger.info('shape of input_ids: {}'.format(input_ids.shape))
                    logger.info('shape of labels: {}'.format(labels.shape))
                loss = model(input_ids=input_ids,
                             token_type_ids=segment_ids,
                             attention_mask=input_masks,
                             labels=labels)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used and handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1


                # train_loss_history.append(loss.item())
                total_train_loss += loss.item()
                num_step += 1

                pbar.set_postfix({'loss': '{0:1.5f}'.format(total_train_loss / (num_step + 1e-5))})
                pbar.update(1)

        logger.info("***** Running predictions *****")
        logger.info('Epoch = {}'.format(ep))
        logger.info("Num split examples = {}".format(len(eval_features)))
        logger.info("Batch size = {}".format(args.predict_batch_size))

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        for batch in tqdm(eval_dataloader,
                          desc="Evaluating",
                          disable=None,
                          mininterval=10.0):
            input_ids, input_masks, segment_ids, choice_masks, example_indices = batch
            if len(all_results) == 0:
                print('shape of input_ids: {}'.format(input_ids.shape))
            input_ids = input_ids.to(device)
            input_masks = input_masks.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_logits = model(input_ids=input_ids,
                                     token_type_ids=segment_ids,
                                     attention_mask=input_masks,
                                     labels=None)
            for i, example_index in enumerate(example_indices):
                logits = batch_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             example_id=all_example_ids[unique_id],
                                             tag=all_tags[unique_id],
                                             logit=logits))

        dev_file = 'dev_predictions.json'
        logger.info('Decode raw results')

        tmp_predict_file = os.path.join(output_dir, "raw_predictions.pkl")
        output_prediction_file = os.path.join(output_dir, dev_file)
        results = get_final_predictions(all_results, tmp_predict_file, g=True)
        write_predictions(results, output_prediction_file)
        
        logger.info('Predictions saved to {}'.format(output_prediction_file))

        acc = evaluate(args.dev_ans_file, output_prediction_file)
        logger.info(f'{args.dev_file} 预测精度：{acc}')

        dev_acc_history.append(acc)
        train_loss = total_train_loss / (num_step + 1e-5)
        train_loss_history.append(train_loss)

        with open(filename_scores, 'w') as f:
            f.write('epoch\ttrain_loss\tdev_acc\n')
            for i in range(ep + 1):
                train_loss = train_loss_history[i]
                dev_acc = dev_acc_history[i]
                f.write(f'{i}\t{train_loss}\t{dev_acc}\n')

        # Save model
        model_to_save = model.module if hasattr(model, 'module') else model
        dir_models = os.path.join(output_dir, 'models')
        os.makedirs(dir_models, exist_ok=True)
        model_filename = os.path.join(dir_models, 'model_epoch_' + str(ep) + '.bin')
        torch.save(
            {"model": model_to_save.state_dict()},
            model_filename,
        )

        # Save best model
        if best_acc is None or acc > best_acc:
            best_acc = acc
            best_model_filename = os.path.join(output_dir, 'best_model.bin')
            copyfile(model_filename, best_model_filename)
            logger.info('New best model saved')

    print('Training finished')


def test(args):
    # Testing
    # Manage output files
    output_dir = os.path.join(args.output_dir, str(args.seed))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Arguments:")
    logger.info(json.dumps(vars(args), indent=4))
    filename_params = os.path.join(output_dir, 'params.json')
    json.dump(vars(args), open(filename_params, 'w'), indent=4)

    device = get_device()
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(device, n_gpu, args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # Set seed
    utils.set_seed(args.seed)

    # Prepare model
    file_checkpoint = os.path.join(output_dir, consts.FILENAME_BEST_MODEL)
    logger.info('Preparing model from checkpoint {}'.format(file_checkpoint))
    config = modeling.BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = modeling.BertForMultipleChoice(config, args.max_num_choices)
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training

    state_dict = torch.load(file_checkpoint, map_location='cpu')['model']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)


    # Tokenizer
    logger.info('Loading tokenizer...')
    tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    real_tokenizer_type = args.output_dir.split(os.path.sep)[-2]
    logger.info('Loaded tokenizer')

    # Load test data
    file_data = os.path.join(args.data_dir, 'test.json')
    file_ans = os.path.join(args.data_dir, 'test_answer.json')
    logger.info('Loading test data...')
    file_examples, file_features = get_filename_examples_and_features(
        'test',
        args.data_dir,
        args.max_seq_length, 
        real_tokenizer_type, 
        config.vocab_size)
    logger.info(f'  file_examples: "{file_examples}"')
    logger.info(f'  file_features: "{file_features}"')

    # Generate (or load) test features
    examples = gen_examples(
        file_data,
        file_ans,
        file_examples,
        is_training=False,
    )
    file_idiom_dict = os.path.join(args.data_dir, 'idiomDict.json')
    features = load_or_gen_features(
        examples,
        file_features, 
        file_idiom_dict=file_idiom_dict,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length, 
        max_num_choices=args.max_num_choices,
        is_training=False,
        split_char=SPLIT_CHAR,
        )
    dataloader, dataset = get_eval_dataloader_and_dataset(
        features, 
        args.predict_batch_size)
    all_example_ids, all_tags = get_example_ids_and_tags(features)

    num_steps = len(features) // args.predict_batch_size

    # Optimizer
    optimizer = get_optimizer(
        model,
        float16=args.fp16,
        learning_rate=args.learning_rate,
        total_steps=num_steps,
        schedule='warmup_linear',
        warmup_rate=args.warmup_proportion,
        weight_decay_rate=0.01,
        max_grad_norm=1.0,
        opt_pooler=True)

    # Save config
    model_to_save = model.module if hasattr(model, 'module') else model
    filename_config = os.path.join(output_dir, modeling.CONFIG_NAME)
    with open(filename_config, 'w') as f:
        f.write(model_to_save.config.to_json_string())
    
    
    # Start Testing
    logger.info('***** Testing *****')
    logger.info('Number of examples: {}'.format(len(dataset)))
    logger.info('Batch size: ' + str(args.predict_batch_size))
    logger.info('Number of steps: {}'.format(num_steps))
    
    num_step = 0
    model.eval()
    model.zero_grad()

    all_results = []
    for batch in tqdm(dataloader, desk='Testing', disable=None, mininterval=5.0):
        input_ids, input_masks, segment_ids, choice_masks, example_indices = batch
        # input_ids, input_masks, segment_ids, choice_masks, example_indices = expand_features(batch, is_training=False)
        if len(all_results) == 0:
            print(f'shape of input_ids: {input_ids.shape}')
        input_ids = input_ids.to(device)
        input_masks = input_masks.to(device)
        segment_ids = segment_ids.to(device)
        
        with torch.no_grad():
            batch_logits = model(input_ids=input_ids,
                                 token_type_ids=segment_ids,
                                 attention_mask=input_masks,
                                 labels=None)
        
        for i, example_idx in enumerate(example_indices):
            logits = batch_logits[i].detach().cpu().tolist()
            feature = features[example_idx.item()]
            unique_id = int(feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         example_id=all_example_ids[unique_id],
                                         tag=all_tags[unique_id],
                                         logit=logits))
    dev_file = 'test_predictions.json'
    logger.info('Decoding test raw results...')
    
    tmp_predict_file = os.path.join(output_dir, 'raw_test_predictions.pkl')
    output_prediction_file = os.path.join(output_dir, dev_file)
    results = get_final_predictions(all_results, tmp_predict_file, g=True)
    write_predictions(results, output_prediction_file)
    
    logger.info('Predictions saved to {}'.format(output_prediction_file))

    # Get acc and save
    acc = evaluate(file_ans, output_prediction_file)
    logger.info(f'{file_data} 预测精度：{acc}')
    file_result = os.path.join(output_dir, consts.FILENAME_TEST_RESULT)
    with open(file_result, 'w') as f:
        f.write(f'test_loss\ttest_acc\n')
        f.write(f'None\t{acc}\n')
    
    print('Testing finished')


def main(args):
    if args.do_train:
        train(args)
    if args.do_test:
        test(args)
    print('DONE')


if __name__ == "__main__":
    main(parse_args())
