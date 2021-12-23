# coding=utf-8

import argparse
import collections
import json
import os
import random
import logging
from shutil import copyfile

from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import consts
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import modeling
from tokenization import (
    ALL_TOKENIZERS,
    BertTokenizer, 
    ConcatSepTokenizer, 
    WubiZhTokenizer, 
    RawZhTokenizer, 
    BertZhTokenizer
)
from optimization import BertAdam, warmup_linear, get_optimizer
from schedulers import LinearWarmUpScheduler
import utils
from utils import (
    json_load_by_line, json_save_by_line, mkdir, get_freer_gpu, get_device, output_dir_to_tokenizer_name
)
from run_pretraining import pretraining_dataset, WorkerInitObj

from mrc.google_albert_pytorch_modeling import AlbertConfig, AlbertForMRC
from mrc.preprocess.cmrc2018_evaluate import get_eval
from mrc.tools import official_tokenization, utils
from mrc.preprocess.cmrc2018_output import write_predictions
from mrc.preprocess.cmrc2018_preprocess import (
    read_cmrc_examples,
    convert_examples_to_features,
)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def evaluate(model, args, file_data, examples, features, device, epoch, output_dir):
    logger.info("***** Eval *****")
    RawResult = collections.namedtuple(
        "RawResult",
        ["unique_id", "start_logits", "end_logits"])
    dir_preds = os.path.join(output_dir, 'predictions')
    os.makedirs(dir_preds, exist_ok=True)
    file_preds = os.path.join(dir_preds, 'predictions_' + str(epoch) + '.json')
    output_nbest_file = file_preds.replace('predictions_', 'nbest_')
    
    all_examples_index = torch.arange(len(features), dtype=torch.long)
    dataset = features_to_dataset(features, is_training=False, two_level_embeddings=args.two_level_embeddings)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    for batch in tqdm(dataloader, desc="Evaluating", mininterval=5.0):
        batch = tuple(t.to(device) for t in batch)
        (input_ids, input_mask, segment_ids, example_indices,
         token_ids, pos_left, pos_right) = expand_batch(batch, is_training=False, two_level_embeddings=args.two_level_embeddings)

        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(
                input_ids, 
                segment_ids, 
                input_mask,
                token_ids=token_ids, 
                pos_left=pos_left,
                pos_right=pos_right,
                use_token_embeddings=args.two_level_embeddings)
        
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            feature = features[example_index.item()]
            unique_id = int(feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    logger.info(f'Writing predictions to "{args.n_best}" and "{file_preds}"')
    write_predictions(
        examples, 
        features, 
        all_results,
        n_best_size=args.n_best, 
        max_answer_length=args.max_ans_length,
        do_lower_case=True, 
        output_prediction_file=file_preds,
        output_nbest_file=output_nbest_file,
        two_level_embeddings=False)

    file_truth = os.path.join(args.data_dir, file_data)
    res = get_eval(file_truth, file_preds)
    model.train()
    return res['em'], res['f1']


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_rate', type=float, default=0.05)
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--n_best', type=int, default=20)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)

    # Other arguments
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--init_checkpoint', type=str, required=True)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--tokenizer_type', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--vocab_model_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='logs/temp')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument('--do_test', action='store_true', help='Whether to test.')
    parser.add_argument('--two_level_embeddings', action='store_true')
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


def features_to_dataset(features, is_training, two_level_embeddings):
    '''
    Turn list of features into Tensor datasets
    '''
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
    if is_training:
        all_start_positions = torch.tensor([f['start_position'] for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f['end_position'] for f in features], dtype=torch.long)
    else:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    if not two_level_embeddings:
        if is_training:
            return TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_start_positions,
                all_end_positions,
            )
        else:
            return TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_example_index,
            )
    else:
        all_token_ids = torch.tensor([f['token_ids'] for f in features], dtype=torch.long)
        all_pos_left = torch.tensor([f['pos_left'] for f in features], dtype=torch.long)
        all_pos_right = torch.tensor([f['pos_right'] for f in features], dtype=torch.long)
        if is_training:
            return TensorDataset(
                all_input_ids, 
                all_input_mask, 
                all_segment_ids, 
                all_start_positions, 
                all_end_positions, 
                all_token_ids,
                all_pos_left,
                all_pos_right,
            )
        else:
            return TensorDataset(
                all_input_ids, 
                all_input_mask, 
                all_segment_ids, 
                all_example_index, 
                all_token_ids,
                all_pos_left,
                all_pos_right,
            )


def expand_batch(batch, is_training, two_level_embeddings):
    input_ids = batch[0]
    input_mask = batch[1]
    segment_ids = batch[2]
    token_ids = None
    pos_left = None
    pos_right = None
    if two_level_embeddings:
        token_ids = batch[-3]
        pos_left = batch[-2]
        pos_right = batch[-1]
    if is_training:
        start_positions = batch[3]
        end_positions = batch[4]
        return (input_ids, input_mask, segment_ids, start_positions, end_positions,
                token_ids, pos_left, pos_right)
    else:
        example_index = batch[3]
        return (input_ids, input_mask, segment_ids, example_index,
                token_ids, pos_left, pos_right)


def get_filename_examples_and_features(
    data_type,
    data_dir,
    max_seq_length,
    tokenizer_name,
    vocab_size,
    two_level_embeddings=False):
    '''
    Return:
        example_file: str,
        feature_file: str,
    '''
    if two_level_embeddings:
        suffix = '_{}_{}_{}_twolevel'.format(str(max_seq_length), tokenizer_name, str(vocab_size))
        suffix_ex = '_examples_twolevel.json'
    else:
        suffix = '_{}_{}_{}'.format(str(max_seq_length), tokenizer_name, str(vocab_size))
        suffix_ex = '_examples.json'
    file_examples = os.path.join(data_dir, data_type + suffix_ex)
    file_features = os.path.join(data_dir, data_type + '_features' + suffix + '.json')
    return file_examples, file_features


def gen_examples_and_features(
    file_data,
    file_examples,
    file_features,
    is_training,
    tokenizer,
    max_seq_length,
    max_query_length=64,
    doc_stride=128,
    two_level_embeddings=False
    ):
    '''
    Return:
        examples: [dict]
        features: [dict]
    '''

    use_example_cache = True
    use_feature_cache = True

    examples, features = None, None
    # Examples
    if use_example_cache and os.path.exists(file_examples):
        logger.info('Found example file, loading...')
        examples = json_load_by_line(file_examples)
        logger.info(f'Loaded {len(examples)} examples')
    else:
        logger.info('Example file not found, generating...')
        examples, mismatch = read_cmrc_examples(file_data, is_training, 
                                                two_level_embeddings=two_level_embeddings)
        logger.info(f'num examples: {len(examples)}')
        logger.info(f'mismatch: {mismatch}')
        logger.info(f'Generated {len(examples)} examples')

        logger.info(f'Saving to "{file_examples}"...')
        json_save_by_line(examples, file_examples)
        logger.info(f'Saved {len(examples)} examples')
        # Somehow, just saving will result in all empty evaluation
        logger.info(f'Loading from "{file_examples}"...')
        examples = json_load_by_line(file_examples)
        logger.info(f'Loaded {len(examples)} examples')

    # Load or gen features
    if use_feature_cache and os.path.exists(file_features):
        logger.info('Found feature file, loading...')
        features = json_load_by_line(file_features)
        logger.info(f'Loaded {len(features)} features')
    else:
        logger.info('Feature file not found, generating...')
        features = convert_examples_to_features(
            examples,
            tokenizer,
            is_training=is_training,
            max_seq_length=max_seq_length,
            two_level_embeddings=two_level_embeddings,
        )
        logger.info(f'Generated {len(features)} features')
        logger.info(f'Saving to "{file_features}"...')
        json_save_by_line(features, file_features)
        logger.info(f'Saved {len(features)} features')
        logger.info('Found feature file, loading...')
        features = json_load_by_line(file_features)
        logger.info(f'Loaded {len(features)} features')

    return examples, features


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def train(args):
    # Prepare files
    output_dir = os.path.join(args.output_dir, str(args.seed))
    filename_scores = os.path.join(output_dir, 'scores.txt')
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Arguments:")
    logger.info(json.dumps(vars(args), indent=4))
    filename_params = os.path.join(output_dir, 'params.json')
    json.dump(vars(args), open(filename_params, 'w'), indent=4)  # Save arguments

    # Device
    device = get_device()  # Get gpu with most free RAM
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    logger.info('SEED: ' + str(args.seed))
    set_seed(args.seed, n_gpu)

    # Prepare model
    logger.info('Loading model from checkpoint "{}"'.format(args.init_checkpoint))
    config = modeling.BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = modeling.BertForQuestionAnswering(config)
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    state_dict = torch.load(args.init_checkpoint, map_location='cpu')
    model.load_state_dict(state_dict['model'], strict=False)
    model.to(device)

    # Save config
    model_to_save = model.module if hasattr(model, 'module') else model
    filename_config = os.path.join(output_dir, modeling.CONFIG_NAME)
    with open(filename_config, 'w') as f:
        f.write(model_to_save.config.to_json_string())

    # Tokenizer
    logger.info('Loading tokenizer...')
    tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    real_tokenizer_type = output_dir_to_tokenizer_name(args.output_dir)
    logger.info('Loaded tokenizer')

    # Because tokenizer_type is a part of the feature file name,
    # new features will be generated for every tokenizer type.
    tokenizer_name = output_dir_to_tokenizer_name(args.output_dir)
    file_train = os.path.join(args.data_dir, 'train.json')
    file_dev = os.path.join(args.data_dir, 'dev.json')
    file_train_examples, file_train_features = get_filename_examples_and_features(
        'train',
        args.data_dir,
        args.max_seq_length,
        tokenizer_name=tokenizer_name,
        vocab_size=config.vocab_size,
        two_level_embeddings=args.two_level_embeddings)
    file_dev_examples, file_dev_features = get_filename_examples_and_features(
        'dev',
        args.data_dir,
        args.max_seq_length,
        tokenizer_name=tokenizer_name,
        vocab_size=config.vocab_size,
        two_level_embeddings=args.two_level_embeddings)
    # Generate train examples and features
    logger.info('Generating train data:')
    logger.info(f'  file_examples: {file_train_examples}')
    logger.info(f'  file_features: {file_train_features}')

    train_examples, train_features = gen_examples_and_features(
        file_train,
        file_train_examples,
        file_train_features,
        is_training=True,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        two_level_embeddings=args.two_level_embeddings)
    logger.info('Generating dev data:')
    logger.info(f'  file_examples: {file_dev_examples}')
    logger.info(f'  file_features: {file_dev_features}')
    dev_examples, dev_features = gen_examples_and_features(
        file_dev,
        file_dev_examples,
        file_dev_features,
        is_training=False,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        two_level_embeddings=args.two_level_embeddings)
    del train_examples  # Only need examples for predictions
    logger.info('Done generating data')

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    train_steps_per_epoch = len(train_features) // args.batch_size
    dev_steps_per_epoch = len(dev_features) // args.batch_size
    if len(train_features) % args.batch_size != 0:
        train_steps_per_epoch += 1
    if len(dev_features) % args.batch_size != 0:
        dev_steps_per_epoch += 1
    total_steps = train_steps_per_epoch * args.epochs

    logger.info('steps per epoch: ' + str(train_steps_per_epoch))
    logger.info('total steps: ' + str(total_steps))
    logger.info('warmup steps: ' + str(int(args.warmup_rate * total_steps)))


    optimizer = get_optimizer(
        model=model,
        float16=False,
        learning_rate=args.lr,
        total_steps=total_steps,
        schedule=args.schedule,
        warmup_rate=args.warmup_rate,
        max_grad_norm=args.clip_norm,
        weight_decay_rate=args.weight_decay_rate)

    # Train and evaluation
    train_data = features_to_dataset(train_features, is_training=True, two_level_embeddings=args.two_level_embeddings)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    logger.info('***** Training *****')
    logger.info(f'num epochs = {args.epochs}')
    logger.info(f'steps for epoch = {train_steps_per_epoch}')
    logger.info(f'batch size = {args.batch_size}')
    logger.info(f'num train features = {len(train_features)}')
    logger.info(f'num dev features = {len(dev_features)}')

    # 存一个全局最优的模型
    global_steps = 1

    train_loss_history = []
    dev_acc_history = []
    dev_f1_history = []

    for ep in range(args.epochs):
        logger.info('Starting epoch %d' % (ep + 1))
        num_train_steps = 0
        total_loss = 0  # of this epoch
        model.train()
        model.zero_grad()
        with tqdm(total=train_steps_per_epoch, desc='Epoch %d' % (ep + 1), mininterval=5.0) as pbar:
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                expand_batch(batch, is_training=True, two_level_embeddings=args.two_level_embeddings)
                (input_ids, input_mask, segment_ids, start_positions, end_positions,
                 token_ids, pos_left, pos_right) = batch

                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions,
                             token_ids=token_ids, pos_left=pos_left, pos_right=pos_right,
                             use_token_embeddings=args.two_level_embeddings)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                total_loss = loss.item()
                pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss / (num_train_steps + 1e-5))})
                pbar.update(1)

                loss.backward()

                num_train_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_steps += 1

        dev_acc, dev_f1 = evaluate(
            model, args, 
            file_data='dev.json',
            examples=dev_examples, 
            features=dev_features, 
            device=device, 
            epoch=ep, 
            output_dir=output_dir,
        )

        train_loss = total_loss / train_steps_per_epoch
        train_loss_history.append(train_loss)
        dev_acc_history.append(dev_acc)
        dev_f1_history.append(dev_f1)
        logger.info(f'train_loss = {train_loss}, dev_acc = {dev_acc}, dev_f1 = {dev_f1}')

        # Save all loss and acc
        with open(filename_scores, 'w') as f:
            f.write(f'epoch\ttrain_loss\tdev_acc\n')
            for i in range(ep+1):
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
        if len(dev_acc_history) == 0 or dev_acc_history[-1] == max(dev_acc_history):
            best_model_filename = os.path.join(output_dir, modeling.FILENAME_BEST_MODEL)
            copyfile(model_filename, best_model_filename)
            logger.info('New best model saved')

    file_scores_backup = filename_scores.replace('.txt', '_backup.txt')
    copyfile(filename_scores, file_scores_backup)
    mean_acc = sum(dev_acc_history) / len(dev_acc_history)
    max_acc = max(dev_acc_history)
    mean_f1 = sum(dev_f1_history) / len(dev_f1_history)
    max_f1 = max(dev_f1_history)

    logger.info(f'Mean F1: {mean_f1} Mean EM: {mean_acc}')
    logger.info(f'Max F1: {max_f1} Max EM: {max_acc}')

    # release the memory
    del model
    del optimizer
    torch.cuda.empty_cache()

    print('Training finished')
    

def test(args):
    logger.info('Test')
    logger.info(json.dumps(vars(args), indent=4))

    # Prepare files
    output_dir = os.path.join(args.output_dir, str(args.seed))
    assert os.path.exists(output_dir)
    assert args.batch_size > 0, 'Batch size must be positive'

    # Device
    device = get_device()  # Get gpu with most free RAM
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    logger.info('SEED: ' + str(args.seed))
    set_seed(args.seed, n_gpu)

    # Prepare model
    file_ckpt = os.path.join(output_dir, modeling.FILENAME_BEST_MODEL)
    logger.info('Preparing model from checkpoint {}'.format(file_ckpt))
    config = modeling.BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = modeling.BertForQuestionAnswering(config)
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    state_dict = torch.load(file_ckpt, map_location='cpu')['model']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Tokenizer
    logger.info('Loading tokenizer...')
    tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    logger.info('Loaded tokenizer')

    # Because tokenizer_type is a part of the feature file name,
    # new features will be generated for every tokenizer type.
    tokenizer_name = output_dir_to_tokenizer_name(args.output_dir)
    file_data = os.path.join(args.data_dir, 'test.json')
    file_examples, file_features = get_filename_examples_and_features(
        'test',
        args.data_dir,
        args.max_seq_length,
        tokenizer_name=tokenizer_name,
        vocab_size=config.vocab_size,
        two_level_embeddings=args.two_level_embeddings)
    # Generate train examples and features
    logger.info('Generating data:')
    logger.info(f'  file_examples: {file_examples}')
    logger.info(f'  file_features: {file_features}')
    examples, features = gen_examples_and_features(
        file_data,
        file_examples,
        file_features,
        is_training=False,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        two_level_embeddings=args.two_level_embeddings)
    logger.info('Done generating data')

    logger.info('***** Testing *****')
    logger.info(f'batch size = {args.batch_size}')
    logger.info(f'num features = {len(features)}')

    model.zero_grad()
    acc, f1 = evaluate(
        model, 
        args,
        file_data='test.json',
        examples=examples,
        features=features,
        device=device,
        epoch='test',
        output_dir=output_dir
    )
    logger.info(f'test: acc = {acc}, f1 = {f1}')

    file_test_result = os.path.join(output_dir, 'result_test.txt')
    with open(file_test_result, 'w') as f:
        f.write('test_loss\ttest_acc\n')
        f.write(f'None\t{acc}\n')
    print('Testing finished')


def main(args):
    if args.do_train:
        train(args)
    if args.do_test:
        test(args)
    print('DONE')


if __name__ == '__main__':
    main(parse_args())