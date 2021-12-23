# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import logging
import os
import pickle
import random
from shutil import copyfile

import consts

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import modeling
from optimization import BertAdam, warmup_linear, get_optimizer
from schedulers import LinearWarmUpScheduler
from utils import mkdir, is_main_process, get_freer_gpu, load_tokenizer

# from google_albert_pytorch_modeling import AlbertConfig, AlbertForMultipleChoice
# from pytorch_modeling import BertConfig, BertForMultipleChoice, ALBertConfig, ALBertForMultipleChoice
from mrc.tools import official_tokenization as tokenization
from mrc.tools import utils
from run_pretraining import pretraining_dataset, WorkerInitObj
# from tools.pytorch_optimization import get_optimization, warmup_linear

# Contants for C3
NUM_CHOICES = 4  # 数据集里不一定有四个选项，但是会手动加 “无效答案” 至4个
REVERSE_ORDER = False
SA_STEP = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class c3Processor(DataProcessor):
    def __init__(self, data_dir, do_train=False, do_eval=False, do_test=False):
        self.D = [[], [], []]
        self.data_dir = data_dir

        for sid in range(3):
        # for sid in range(2):
            # Skip files that are not going to use
            if not do_train:
                if sid == 0:
                    continue
            if not do_eval:
                if sid == 1:
                    continue
            if not do_test:
                if sid == 2:
                    continue
                
            data = []
            for subtask in ["d", "m"]:
                files = ["train.json", "dev.json", "test.json"]
                # files = ['train.json', 'dev.json']
                filename = self.data_dir + "/" + subtask + "-" + files[sid]
                with open(filename, "r", encoding="utf8") as f:
                    data += json.load(f)
            logger.info('Loaded {} examples from "{}"'.format(len(data), filename))
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
                    for k in range(len(data[i][1][j]["choice"])):
                        d += [data[i][1][j]["choice"][k].lower()]
                    for k in range(len(data[i][1][j]["choice"]), 4):
                        d += ['无效答案']  # 有些C3数据选项不足4个，添加[无效答案]能够有效增强模型收敛稳定性
                    d += [data[i][1][j]["answer"].lower()]
                    self.D[sid] += [d]

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self.D[0], "train")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(self.D[2], "test")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        cache_dir = os.path.join(self.data_dir, set_type + '_examples.pkl')
        # if os.path.exists(cache_dir):
        if False:
            examples = pickle.load(open(cache_dir, 'rb'))
        else:
            examples = []
            for (i, d) in enumerate(data):
                answer = -1
                # 这里data[i]有6个元素，0是context，1是问题，2~5是choice，6是答案
                for k in range(4):
                    if data[i][2 + k] == data[i][6]:
                        answer = str(k)
                label = tokenization.convert_to_unicode(answer)
                for k in range(4):
                    guid = "%s-%s-%s" % (set_type, i, k)
                    text_a = tokenization.convert_to_unicode(data[i][0])
                    text_b = tokenization.convert_to_unicode(data[i][k + 2])
                    text_c = tokenization.convert_to_unicode(data[i][1])
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, text_c=text_c))

            with open(cache_dir, 'wb') as w:
                pickle.dump(examples, w)

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # print("#examples", len(examples))

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = [[]]
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)

        tokens_c = tokenizer.tokenize(example.text_c)

        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        tokens_b = tokens_c + ["[SEP]"] + tokens_b

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features[-1].append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))
        if len(features[-1]) == NUM_CHOICES:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tokenizer_type', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--vocab_model_file', type=str, required=True)
    parser.add_argument('--cws_vocab_file', type=str, default='')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--init_checkpoint', type=str, required=True)

    ## Other parameters
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--do_test', 
                        default=False, 
                        action='store_true', 
                        help='Whether to test model on test set (will load model from "best_model.bin")')
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--schedule",
                        default='warmup_linear',
                        type=str,
                        help='schedule')
    parser.add_argument("--weight_decay_rate",
                        default=0.01,
                        type=float,
                        help='weight_decay_rate')
    parser.add_argument('--clip_norm',
                        type=float,
                        default=1.0)
    parser.add_argument("--num_train_epochs",
                        default=8.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--float16',
                        action='store_true',
                        default=False)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--test_model', type=str, default=None)
    # parser.add_argument('--setting_file', type=str, default='setting.txt')
    return parser.parse_args()


def get_features(
    examples, 
    data_type, 
    data_dir, 
    max_seq_length,
    tokenizer,
    tokenizer_type,
    vocab_size,
    label_list):

    if data_type == 'eval':
        data_type = 'dev'

    if data_type not in ['train', 'dev', 'test']:
        raise ValueError('Expected "train", "dev" or "test", but got', data_type)

    file_feature = '{}_features_{}_{}_{}.pkl'.format(data_type, max_seq_length, tokenizer_type, vocab_size)
    file_feature = os.path.join(data_dir, file_feature)

    if data_type != 'test' and os.path.exists(file_feature):
    # if False:
        logger.info('Loading features from \"' + file_feature + '\"...')
        features = pickle.load(open(file_feature, 'rb'))
        logger.info('Loaded {} features.'.format(len(features)))
    else:
        logger.info('Converting {} examples into features...'.format(len(examples)))
        features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)
        with open(file_feature, 'wb') as w:
            pickle.dump(features, w)
        logger.info('Saved {} features to "{}".'.format(len(features), file_feature))
    return features


def get_device(args):
    if torch.cuda.is_available():
        return torch.device('cuda')  # Only one gpu
        free_gpu = get_freer_gpu()
        return torch.device('cuda', free_gpu)
    else:
        return torch.device('cpu')


def train(args):
    # Output files
    output_dir = os.path.join(args.output_dir, str(args.seed))
    os.makedirs(output_dir, exist_ok=True)

    filename_scores = os.path.join(output_dir, 'scores.txt')
    filename_params = os.path.join(output_dir, 'params.json')
    logger.info(json.dumps(vars(args), indent=4))
    json.dump(vars(args), open(filename_params, 'w'), indent=4)
    with open(filename_scores, 'w') as f:
        f.write('\t'.join(['epoch', 'train_loss', 'dev_loss', 'dev_acc']) + '\n')

    device = get_device(args)
    n_gpu = torch.cuda.device_count()
    logger.info('Device: ' + str(device))
    logger.info('Num gpus: ' + str(n_gpu))
    # logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    # Processor
    logger.info('Loading processor...')
    processor = c3Processor(args.data_dir, do_train=True, do_eval=True)
    label_list = processor.get_labels()

    # Tokenizer
    logger.info('Loading tokenizer...')
    logger.info('vocab file={}, vocab_model_file={}'.format(args.vocab_file, args.vocab_model_file))
    # tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    tokenizer = load_tokenizer(args)
    real_tokenizer_type = args.output_dir.split(os.path.sep)[-2]
    
    # Load training data
    logger.info('Loading training data...')
    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples()
        num_train_steps = int(len(train_examples) / NUM_CHOICES / args.train_batch_size /
                              args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare Model
    logger.info('Loading model from checkpoint "{}"...'.format(args.init_checkpoint))
    config = modeling.BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForMultipleChoice(config, NUM_CHOICES)
    state_dict = torch.load(args.init_checkpoint, map_location='cpu')['model']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, config.max_position_embeddings))
    if args.float16:
        model.half()
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank],
            output_device=args.local_rank)
    elif n_gpu > 1:
        # model = torch.nn.DataParallel(model)
        pass

    logger.info('Saving config file...')
    model_to_save = model.module if hasattr(model, 'module') else model
    filename_config = os.path.join(output_dir, modeling.FILENAME_CONFIG)
    with open(filename_config, 'w') as f:
        f.write(model_to_save.config.to_json_string())


    # Optimizer
    optimizer = get_optimizer(
        model=model,
        float16=args.float16,
        learning_rate=args.learning_rate,
        total_steps=num_train_steps,
        schedule=args.schedule,
        warmup_rate=args.warmup_proportion,
        max_grad_norm=args.clip_norm,
        weight_decay_rate=args.weight_decay_rate,
        opt_pooler=True)  # multi_choice must update pooler

    global_step = 0

    # Load eval data
    eval_dataloader = None
    if args.do_eval:
        logger.info('Loading eval data...')
        eval_examples = processor.get_dev_examples()
        eval_features = get_features(
            eval_examples, 
            'eval', 
            args.data_dir, 
            args.max_seq_length,
            tokenizer,
            real_tokenizer_type,
            config.vocab_size,
            label_list)
        # feature_file = f'dev_features_{args.max_seq_length}_{args.tokenizer_type}.pkl'
        # feature_dir = os.path.join(args.data_dir, feature_file)

        # logger.info('Loading (or generating) eval features...')
        # if os.path.exists(feature_dir):
        #     logger.info(f'Loading features from "{feature_dir}"...')
        #     eval_features = pickle.load(open(feature_dir, 'rb'))
        #     logger.info(f'Loaded {len(eval_features)} features.')
        # else:
        #     logger.info(f'Converting {len(eval_examples)} examples to features')
        #     eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        #     with open(feature_dir, 'wb') as w:
        #         pickle.dump(eval_features, w)
        #     logger.info(f'Saved {len(eval_examples)} features to "{feature_dir}".')

        # TODO: remove (for debugging only)
        # eval_features = eval_features[:100]

        input_ids = []
        input_mask = []
        segment_ids = []
        label_id = []

        for f in eval_features:
            input_ids.append([])
            input_mask.append([])
            segment_ids.append([])
            for i in range(NUM_CHOICES):
                input_ids[-1].append(f[i].input_ids)
                input_mask[-1].append(f[i].input_mask)
                segment_ids[-1].append(f[i].segment_ids)
            label_id.append(f[0].label_id)

        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(label_id, dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_train:
        best_accuracy = 0

        # feature_dir = os.path.join(args.data_dir, 'train_features{}.pkl'.format(args.max_seq_length))
        # if os.path.exists(feature_dir):
        # # if False:
        #     logger.info(f'Loading train features from "{feature_dir}"...')
        #     train_features = pickle.load(open(feature_dir, 'rb'))
        #     logger.info(f'Loading {len(train_features)} train features.')
        # else:
        #     logger.info(f'Converting {len(train_examples)} into features...')
        #     train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
        #     with open(feature_dir, 'wb') as w:
        #         pickle.dump(train_features, w)
        #     logger.info(f'Saved {len(train_features)} features to "{feature_dir}".')
        train_features = get_features(
            train_examples, 
            'train', 
            args.data_dir,
            args.max_seq_length,
            tokenizer,
            real_tokenizer_type,
            config.vocab_size,
            label_list)
        # TODO: remove (for debugging only)
        # train_features = train_features[:600]

        logger.info("***** Running training *****")
        logger.info('  Num epochs = %d', args.num_train_epochs)
        logger.info("  Num examples = %d", len(train_examples))
        logger.info('  Num features = %d', len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        # Process features
        input_ids = []
        input_mask = []
        segment_ids = []
        label_id = []
        for f in train_features:
            input_ids.append([])
            input_mask.append([])
            segment_ids.append([])
            for i in range(NUM_CHOICES):
                input_ids[-1].append(f[i].input_ids)
                input_mask[-1].append(f[i].input_mask)
                segment_ids[-1].append(f[i].segment_ids)
            label_id.append(f[0].label_id)

        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(label_id, dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      drop_last=True)
        steps_per_epoch = int(num_train_steps / args.num_train_epochs)

        train_loss_history = []
        eval_loss_history = []
        eval_acc_history = []

        for ep in range(int(args.num_train_epochs)):
            model.train()
            total_train_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            with tqdm(total=int(steps_per_epoch), desc='Epoch %d' % (ep + 1)) as pbar:
                for step, batch in enumerate(train_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch

                    loss = model(input_ids, segment_ids, input_mask, label_ids)

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    total_train_loss += loss.item()

                    if args.float16:
                        optimizer.backward(loss)
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    else:
                        loss.backward()

                    nb_tr_examples += input_ids.size(0)
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()  # We have accumulated enought gradients
                        model.zero_grad()
                        global_step += 1
                        nb_tr_steps += 1
                        pbar.set_postfix({'loss': '{0:1.5f}'.format(total_train_loss / (nb_tr_steps + 1e-5))})
                        pbar.update(1)

            # Evaluation
            if args.do_eval:

                logger.info("***** Running Evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info('  Num features = %d', len(eval_features))
                logger.info("  Batch size = %d", args.eval_batch_size)

                model.eval()
                eval_loss, eval_acc = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                logits_all = []
                for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, return_logits=True)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.cpu().numpy()
                    for i in range(len(logits)):
                        logits_all += [logits[i]]

                    tmp_eval_acc = accuracy(logits, label_ids.reshape(-1))

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_acc += tmp_eval_acc

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

                # Average
                eval_loss = eval_loss / (nb_eval_steps + 1e-5)
                eval_acc = eval_acc / (nb_eval_examples + 1e-5)
                train_loss = total_train_loss / (nb_tr_steps + 1e-5)
                
                eval_acc_history.append(eval_acc)
                eval_loss_history.append(eval_loss)
                train_loss_history.append(train_loss)

                if args.do_train:
                    result = {'eval_loss': eval_loss,
                              'eval_acc': eval_acc,
                              'global_step': global_step,
                              'train_loss': train_loss}
                else:
                    result = {'eval_loss': eval_loss,
                              'eval_acc': eval_acc}

                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info('  Epoch = {}'.format(ep))

                # Save result of this epoch
                with open(filename_scores, 'a') as f:
                    f.write("{}\t{}\t{}\t{}\n".format(ep, train_loss, eval_loss, eval_acc))

                # with open(args.log_file, 'a') as aw:
                #     aw.write("-------------------global steps:{}-------------------\n".format(global_step))
                #     aw.write(str(json.dumps(result, indent=2)) + '\n')

                # if eval_acc >= best_accuracy:
                #     torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                #     best_accuracy = eval_acc

            # Save model
            if is_main_process():
                model_to_save = model.module if hasattr(model, 'module') else model
                dir_model = os.path.join(output_dir, 'models')
                os.makedirs(dir_model, exist_ok=True)
                filename_model = os.path.join(dir_model, 'model_epoch_' + str(ep) + '.bin')
                # filename_model = os.path.join(output_dir, 'model_epoch_' + str(ep) + '.bin')
                torch.save(
                    {"model": model_to_save.state_dict()},
                    filename_model,
                )
                
                if args.do_eval:
                    if len(eval_acc_history) == 0 or eval_acc_history[-1] == max(eval_acc_history):
                        filename_best_model = os.path.join(output_dir, modeling.FILENAME_BEST_MODEL)
                        copyfile(filename_model, filename_best_model)
                        logger.info('New best model saved')

        # model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_best.pt")))
        # torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print('Training finished')


def test(args):
    # Output files
    output_dir = os.path.join(args.output_dir, str(args.seed))
    os.makedirs(output_dir, exist_ok=True)

    # filename_scores = os.path.join(output_dir, 'scores.txt')
    # filename_params = os.path.join(output_dir, 'params.json')
    logger.info(json.dumps(vars(args), indent=4))
    # json.dump(vars(args), open(filename_params, 'w'), indent=4)
    # with open(filename_scores, 'w') as f:


    # Setup cuda
    # if args.local_rank == -1 or args.no_cuda:
    #     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #     n_gpu = torch.cuda.device_count()
    # else:
    #     device = torch.device("cuda", args.local_rank)
    #     n_gpu = 1
    #     # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.distributed.init_process_group(backend='nccl')
    # logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))
    device = get_device(args)
    n_gpu = torch.cuda.device_count()
    logger.info('Device: ' + str(device))
    logger.info('Num gpus: ' + str(n_gpu))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    batch_size = args.eval_batch_size


    # Tokenizer and processor
    logger.info('Loading processor...')
    processor = c3Processor(args.data_dir, do_test=True)
    label_list = processor.get_labels()
    logger.info('Loading tokenizer...')
    # tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    tokenizer = load_tokenizer(args)
    

    # Load test data
    logger.info('Loading test data...')
    test_examples = processor.get_test_examples()
    num_examples = len(test_examples)
    num_steps = int(num_examples / NUM_CHOICES / batch_size)

    # Load model
    # filename_best_model = os.path.join(output_dir, modeling.WEIGHTS_NAME + '_best')
    if args.test_model is not None and len(args.test_model) > 0:
        filename_best_model = args.test_model
    else:
        filename_best_model = os.path.join(output_dir, modeling.FILENAME_BEST_MODEL)
    if args.config_file is not None and len(args.config_file) > 0:
        filename_config = args.config_file
    else:
        filename_config = os.path.join(output_dir, modeling.FILENAME_CONFIG)

    logger.info('Loading model from "{}"...'.format(filename_best_model))
    config = modeling.BertConfig.from_json_file(filename_config)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForMultipleChoice(config, NUM_CHOICES)
    state_dict = torch.load(filename_best_model, map_location='cpu')['model']
    model.load_state_dict(state_dict, strict=False)
    
    # for i in range(len(logits_all)):
    #     for j in range(len(logits_all[i])):
    #         f.write(str(logits_all[i][j]))
    #         if j == len(logits_all[i]) - 1:
    #             f.write("\n")
    #         else:
    #             f.write(" ")

    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, config.max_position_embeddings))
    if args.float16:
        model.half()
    model.to(device)

    real_tokenizer_type = args.output_dir.split(os.path.sep)[-2]
    logger.info('Loading test data...')
    test_examples = processor.get_test_examples()
    test_features = get_features(
        test_examples, 
        'test', 
        args.data_dir, 
        args.max_seq_length,
        tokenizer,
        real_tokenizer_type,
        config.vocab_size,
        label_list)
    # feature_file = f'test_features_{args.max_seq_length}_{args.tokenizer_type}.pkl'
    # feature_dir = os.path.join(args.data_dir, feature_file)
    # if os.path.exists(feature_dir):
    #     test_features = pickle.load(open(feature_dir, 'rb'))
    # else:
    #     test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
    #     with open(feature_dir, 'wb') as w:
    #         pickle.dump(test_features, w)

    # TODO: remove (for debugging only)
    # test_features = test_features[:100]

    logger.info("***** Running testing *****")
    logger.info('  Num examples = {}'.format(len(test_examples)))
    logger.info('  Num features = {}'.format(len(test_features)))
    logger.info('  Batch size   = {}'.format(args.eval_batch_size))

    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []

    for f in test_features:
        input_ids.append([])
        input_mask.append([])
        segment_ids.append([])
        for i in range(NUM_CHOICES):
            input_ids[-1].append(f[i].input_ids)
            input_mask[-1].append(f[i].input_mask)
            segment_ids[-1].append(f[i].segment_ids)
        label_id.append(f[0].label_id)

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(label_id, dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args.local_rank == -1:
        test_sampler = SequentialSampler(test_data)
    else:
        test_sampler = DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    model.eval()
    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0
    logits_all = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_test_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, return_logits=True)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        for i in range(len(logits)):
            logits_all += [logits[i]]


        tmp_test_accuracy = accuracy(logits, label_ids.reshape(-1))

        test_loss += tmp_test_loss.mean().item()
        test_accuracy += tmp_test_accuracy

        nb_test_examples += input_ids.size(0)
        nb_test_steps += 1

    test_loss = test_loss / (nb_test_steps + 1e-5)
    test_accuracy = test_accuracy / (nb_test_examples + 1e-5)

    result = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy}

    output_test_file = os.path.join(output_dir, consts.FILENAME_TEST_RESULT)
    with open(output_test_file, "w") as writer:
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    output_test_file = os.path.join(output_dir, "logits_test.txt")

    logger.info('saving to logits_test.txt')
    with open(output_test_file, "w") as f:
        for i in range(len(logits_all)):
            for j in range(len(logits_all[i])):
                f.write(str(logits_all[i][j]))
                if j == len(logits_all[i]) - 1:
                    f.write("\n")
                else:
                    f.write(" ")

    logger.info('Saving predictions to submission_test.json')
    # the test submission order can't be changed
    submission_test = os.path.join(output_dir, "submission_test.json")
    test_preds = [int(np.argmax(logits_)) for logits_ in logits_all]
    with open(submission_test, "w") as f:
            json.dump(test_preds, f)

    print('Testing finished')


def main():
    args = parse_args()
    
    # Seed
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

    if args.do_train:
        train(args)

    if args.do_test:
        test(args)

    print('DONE')


if __name__ == "__main__":
    main()
