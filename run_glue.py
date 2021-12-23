# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import absolute_import, division, print_function

import pickle
import argparse
import logging
import os
import random
import json
import time
from shutil import copyfile

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

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
    CommonZhTokenizer,
)
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler
from apex import amp
from sklearn.metrics import matthews_corrcoef, f1_score
import utils
from utils import (is_main_process, mkdir_by_main_process, format_step,
                   get_world_size, get_freer_gpu)
from processors.glue import PROCESSORS, convert_examples_to_features
from run_pretraining import pretraining_dataset, WorkerInitObj

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

FILENAME_BEST_MODEL = 'best_model.bin'
FILENAME_TEST_RESULT = 'result_test.txt'

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        # use acc for other classification tasks. Add exceptions above.
        return {"acc": simple_accuracy(preds, labels)}


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


from apex.multi_tensor_apply import multi_tensor_applier


class GradientClipper:
    """
    Clips gradient norm of an iterable of parameters.
    """

    def __init__(self, max_grad_norm):
        self.max_norm = max_grad_norm
        if multi_tensor_applier.available:
            import amp_C
            self._overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
            self.multi_tensor_scale = amp_C.multi_tensor_scale
        else:
            raise RuntimeError('Gradient clipping requires cuda extensions')

    def step(self, parameters):
        l = [p.grad for p in parameters if p.grad is not None]
        total_norm, _ = multi_tensor_applier(
            self.multi_tensor_l2norm,
            self._overflow_buf,
            [l],
            False,
        )
        total_norm = total_norm.item()
        if (total_norm == float('inf')): return
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            multi_tensor_applier(
                self.multi_tensor_scale,
                self._overflow_buf,
                [l, l],
                clip_coef,
            )


def parse_args(parser=argparse.ArgumentParser()):
    ## Required parameters
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--dev_dir', type=str)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument("--task_name", type=str, required=True, choices=PROCESSORS.keys())
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--init_checkpoint", type=str, required=True, help="The checkpoint file from pretraining",)
    parser.add_argument('--tokenizer_type', type=str, required=True, help="Type of tokenizer")
    parser.add_argument('--vocab_file', type=str, required=True, help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument('--vocab_model_file', type=str, required=True, help="Model file for sentencepiece")
    parser.add_argument("--config_file", type=str, required=True, help="The BERT model config")

    ## Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int,
        help="Maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--train_batch_size", default=32, type=int,)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--epochs", default=-1, type=int)
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup "
        "for. E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument("--no_cuda", action='store_true', 
                        help="If true, don't use CUDA")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when "
        "fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument('--skip_checkpoint', action='store_true', help="Whether to save checkpoints")
    parser.add_argument('--two_level_embeddings', action="store_true")
    parser.add_argument('--fewshot', type=int, default=0)
    parser.add_argument('--test_model', type=str, default=None)
    parser.add_argument('--cws_vocab_file', type=str, default=None)
    return parser.parse_args()


def init_optimizer_and_amp(model, learning_rate, loss_scale, warmup_proportion,
                           num_train_optimization_steps, use_fp16):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]
    optimizer, scheduler = None, None
    if use_fp16:
        logger.info("using fp16")
        try:
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from "
                              "https://www.github.com/nvidia/apex to use "
                              "distributed and fp16 training.")

        if num_train_optimization_steps is not None:
            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=learning_rate,
                bias_correction=False,
            )
        amp_inits = amp.initialize(
            model,
            optimizers=optimizer,
            opt_level="O2",
            keep_batchnorm_fp32=False,
            loss_scale="dynamic" if loss_scale == 0 else loss_scale,
        )
        model, optimizer = (amp_inits
                            if num_train_optimization_steps is not None else
                            (amp_inits, None))
        if num_train_optimization_steps is not None:
            scheduler = LinearWarmUpScheduler(
                optimizer,
                warmup=warmup_proportion,
                total_steps=num_train_optimization_steps,
            )
    else:
        logger.info("using fp32")
        if num_train_optimization_steps is not None:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=learning_rate,
                warmup=warmup_proportion,
                t_total=num_train_optimization_steps,
            )
    return model, optimizer, scheduler


def gen_tensor_dataset(features, two_level_embeddings):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long,
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long,
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features],
        dtype=torch.long,
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in features],
        dtype=torch.long,
    )
    if not two_level_embeddings:
        return TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
        )
    else:
        all_token_ids = torch.tensor(
            [f.token_ids for f in features],
            dtype=torch.long,
        )
        all_pos_left = torch.tensor(
            [f.pos_left for f in features],
            dtype=torch.long,
        )
        all_pos_right = torch.tensor(
            [f.pos_right for f in features],
            dtype=torch.long,
        )
        return TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_token_ids,
            all_pos_left,
            all_pos_right,
        )


def get_train_features(data_dir, max_seq_length, train_batch_size,
                       gradient_accumulation_steps, epochs, tokenizer,
                       processor, is_fewshot=False, two_level_embeddings=False):
    train_examples = processor.get_train_examples(data_dir)
        
    train_features, _ = convert_examples_to_features(
        train_examples,
        processor.get_labels(),
        max_seq_length,
        tokenizer,
        two_level_embeddings=two_level_embeddings,
    )
    return train_features


def dump_predictions(path, label_map, preds, examples):
    label_rmap = {label_idx: label for label, label_idx in label_map.items()}
    predictions = {
        example.guid: label_rmap[preds[i]] for i, example in enumerate(examples)
    }
    with open(path, "w") as writer:
        json.dump(predictions, writer)


def load_model(config_file, filename, num_labels):
    # Prepare model
    config = modeling.BertConfig.from_json_file(config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = modeling.BertForSequenceClassification(config, num_labels=num_labels)
    print('filename =', filename)
    state_dict = torch.load(filename, map_location='cpu')
    model.load_state_dict(state_dict["model"], strict=False)
    return model


def get_device(args):
    if torch.cuda.is_available():
        return 'cuda'
        # free_gpu = get_freer_gpu()
        # return torch.device('cuda', free_gpu)
    else:
        return torch.device('cpu')


def expand_batch(batch, two_level_embeddings):
    input_ids = batch[0]
    input_mask = batch[1]
    segment_ids = batch[2]
    label_ids = batch[3]

    if two_level_embeddings:
        token_ids = batch[4]
        pos_left = batch[5]
        pos_right = batch[6]
    else:
        token_ids = None
        pos_left = None
        pos_right = None
    return (input_ids, input_mask, segment_ids, label_ids,
            token_ids, pos_left, pos_right)


def train(args):
    device = get_device(args)
    n_gpu = torch.cuda.device_count()
    logger.info(f'Device: {device}')
    logger.info(f'Num gpus: {n_gpu}')

    logger.info('Loading processor and tokenizer...')
    processor = PROCESSORS[args.task_name]()
    num_labels = len(processor.get_labels())
    tokenizer = utils.load_tokenizer(args)

    # Setup output files
    output_dir = os.path.join(args.output_dir, str(args.seed))
    filename_params = os.path.join(output_dir, 'args_train.json')
    json.dump(vars(args), open(filename_params, 'w'), indent=4)

    # Load data
    logger.info('Getting training features...')
    train_features = get_train_features(
        args.train_dir,
        args.max_seq_length,
        args.train_batch_size,
        args.gradient_accumulation_steps,
        args.epochs,
        tokenizer,
        processor,
        two_level_embeddings=args.two_level_embeddings,
    )
    num_train_optimization_steps = int(
        len(train_features) / args.train_batch_size /
        args.gradient_accumulation_steps) * args.epochs

    utils.set_seed(args.seed)
    # Prepare model
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    logger.info('Loading model from "{}"...'.format(args.init_checkpoint))
    model = load_model(args.config_file, args.init_checkpoint, num_labels)
    logger.info('Loaded model from "{}"'.format(args.init_checkpoint))

    model.to(device)

    # Prepare optimizer
    model, optimizer, scheduler = init_optimizer_and_amp(
        model,
        args.learning_rate,
        args.loss_scale,
        args.warmup_proportion,
        num_train_optimization_steps,
        # args.fp16,
        False,
    )
    loss_fct = torch.nn.CrossEntropyLoss()

    logger.info("***** Running training *****")
    logger.info('  Num epochs = %d', args.epochs)
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    train_data = gen_tensor_dataset(train_features, two_level_embeddings=args.two_level_embeddings)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
    )

    global_step = 0
    num_train_steps = 0
    total_train_loss = 0
    latency_train = 0.0
    num_train_examples = 0
    model.train()
    tic_train = time.perf_counter()

    train_acc_history = []
    eval_acc_history = []
    train_loss_history = []
    eval_loss_history = []
    filename_scores = os.path.join(output_dir, "scores.txt")
    with open(filename_scores, 'w') as f:
        f.write('\t'.join(['epoch', 'train_loss', 'dev_loss', 'dev_acc']) + '\n')

    # Save config
    model_to_save = model.module if hasattr(model, 'module') else model
    filename_config = os.path.join(output_dir, modeling.CONFIG_NAME)

    for ep in trange(int(args.epochs), desc="Epoch"):
        # Train
        model.train()
        total_train_loss, num_train_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            (input_ids, input_mask, segment_ids, label_ids,
             token_ids, pos_left, pos_right) = expand_batch(batch, args.two_level_embeddings)
            if args.two_level_embeddings:
                assert token_ids is not None
            else:
                assert token_ids is None

            logits = model(input_ids, segment_ids, input_mask,
                            token_ids=token_ids, pos_left=pos_left, pos_right=pos_right,
                            use_token_embeddings=args.two_level_embeddings)
            loss = loss_fct(
                logits.view(-1, num_labels),
                label_ids.view(-1),
            )
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            total_train_loss += loss.item()
            num_train_examples += input_ids.size(0)
            num_train_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # if args.fp16:
                #     # modify learning rate with special warm up for BERT
                #     # which FusedAdam doesn't do
                #     scheduler.step()

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        
        train_loss = total_train_loss / (num_train_steps + 1e-10)
        train_loss_history.append(train_loss)

        # Evaluation
        if args.do_eval and is_main_process():
            eval_examples = processor.get_dev_examples(args.dev_dir)
            eval_features, label_map = convert_examples_to_features(
                eval_examples,
                processor.get_labels(),
                args.max_seq_length,
                tokenizer,
                two_level_embeddings=args.two_level_embeddings,
            )
            logger.info("***** Running evaluation *****")
            logger.info("  Epoch = %d", ep)
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            eval_data = gen_tensor_dataset(eval_features, two_level_embeddings=args.two_level_embeddings)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data,
                sampler=eval_sampler,
                batch_size=args.eval_batch_size,
            )

            model.eval()
            preds = None
            out_label_ids = None
            total_eval_loss = 0
            num_eval_steps, num_eval_examples = 0, 0
            for i, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating"):
                batch = tuple(t.to(device) for t in batch)
                (input_ids, input_mask, segment_ids, label_ids,
                 token_ids, pos_left, pos_right) = expand_batch(batch, args.two_level_embeddings)

                if args.two_level_embeddings:
                    assert token_ids is not None
                else:
                    assert token_ids is None

                with torch.no_grad():
                    # cuda_events[i][0].record()
                    logits = model(input_ids, segment_ids, input_mask,
                                   token_ids=token_ids, pos_left=pos_left, pos_right=pos_right,
                                   use_token_embeddings=args.two_level_embeddings)
                    # cuda_events[i][1].record()
                    if args.do_eval:
                        total_eval_loss += loss_fct(
                            logits.view(-1, num_labels),
                            label_ids.view(-1),
                        ).mean().item()

                num_eval_steps += 1
                num_eval_examples += input_ids.size(0)

                # Get preds and output ids
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids,
                        label_ids.detach().cpu().numpy(),
                        axis=0,
                    )

            preds = np.argmax(preds, axis=1)

            # Log and update results
            eval_acc = compute_metrics(args.task_name, preds, out_label_ids)['acc']
            eval_loss = total_eval_loss / (num_eval_steps + 1e-10)
            
            # Log
            if is_main_process():
                logger.info("***** Results *****")
                logger.info(f'train loss: {train_loss}')
                logger.info(f'eval loss:  {eval_loss}')
                logger.info(f'eval acc:   {eval_acc}')

            eval_loss_history.append(eval_loss)
            eval_acc_history.append(eval_acc)

            with open(filename_scores, 'w') as f:
                f.write('\t'.join(['epoch', 'train_loss', 'dev_loss', 'dev_acc']) + '\n')
                for i in range(ep + 1):
                    train_loss = train_loss_history[i]
                    eval_loss = eval_loss_history[i]
                    eval_acc = eval_acc_history[i]
                    f.write(f"{i}\t{train_loss}\t{eval_loss}\t{eval_acc}\n")


        # Save model
        if is_main_process() and not args.skip_checkpoint:
            model_to_save = model.module if hasattr(model, 'module') else model
            model_dir = os.path.join(output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_filename = os.path.join(model_dir, modeling.WEIGHTS_NAME + '_' + str(ep))
            torch.save(
                {"model": model_to_save.state_dict()},
                model_filename,
            )

            # Check if it's best model
            if args.do_eval:
                if len(eval_acc_history) == 0 or eval_acc_history[-1] == max(eval_acc_history):
                    best_model_filename = os.path.join(output_dir, FILENAME_BEST_MODEL)
                    copyfile(model_filename, best_model_filename)
                    logger.info("New best model saved")

    logger.info('Training finished')


def test(args):
    # Setup output files
    output_dir = os.path.join(args.output_dir, str(args.seed))
    device = get_device(args)
    n_gpu = torch.cuda.device_count()
    utils.set_seed(args.seed)

    # Tokenizer and processor
    logger.info('Loading processor and tokenizer...')
    processor = PROCESSORS[args.task_name]()
    num_labels = len(processor.get_labels())
    tokenizer = utils.load_tokenizer(args)

    # Load test data
    logger.info('Loading test data from "{}"'.format(args.test_dir))
    examples = processor.get_test_examples(args.test_dir)
    eval_features, label_map = convert_examples_to_features(
        examples,
        processor.get_labels(),
        args.max_seq_length,
        tokenizer,
        two_level_embeddings=args.two_level_embeddings,
    )

    # Load best model
    if args.test_model:
        best_model_filename = args.test_model
    else:
        best_model_filename = os.path.join(output_dir, FILENAME_BEST_MODEL)
    logger.info('Loading model from "{}"...'.format(best_model_filename))
    model = load_model(args.config_file, best_model_filename, num_labels)
    logger.info('Loaded model from "{}"'.format(best_model_filename))
    model.to(device)


    eval_data = gen_tensor_dataset(eval_features, two_level_embeddings=args.two_level_embeddings)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
    )

    # Test
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size   = %d", args.eval_batch_size)
    loss_fct = torch.nn.CrossEntropyLoss()
    preds = None
    out_label_ids = None
    total_eval_loss = 0
    num_eval_steps, num_eval_examples = 0, 0
    cuda_events = [(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))
                    for _ in range(len(eval_dataloader))]

    model.eval()

    for i, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        (input_ids, input_mask, segment_ids, label_ids,
         token_ids, pos_left, pos_right) = expand_batch(batch, args.two_level_embeddings)

        if args.two_level_embeddings:
            assert token_ids is not None
        else:
            assert token_ids is None

        with torch.no_grad():
            cuda_events[i][0].record()
            logits = model(input_ids, segment_ids, input_mask,
                           token_ids=token_ids, pos_left=pos_left, pos_right=pos_right,
                           use_token_embeddings=args.two_level_embeddings)
            cuda_events[i][1].record()
            if args.do_eval:
                total_eval_loss += loss_fct(
                    logits.view(-1, num_labels),
                    label_ids.view(-1),
                ).mean().item()

        num_eval_steps += 1
        num_eval_examples += input_ids.size(0)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids,
                label_ids.detach().cpu().numpy(),
                axis=0,
            )
        # print('len(preds) =', len(preds))
    torch.cuda.synchronize()
    preds = np.argmax(preds, axis=1)

    # Save predictions
    dump_predictions(
        os.path.join(output_dir, 'predictions.json'),
        label_map,
        preds,
        examples,
    )

    loss = total_eval_loss / num_eval_steps
    result = compute_metrics(args.task_name, preds, out_label_ids)
    acc = result['acc']

    # Save result to file
    result_file = os.path.join(output_dir, FILENAME_TEST_RESULT)
    with open(result_file, 'w') as f:
        f.write(f'test_loss\ttest_acc\n')
        f.write(f'{loss}\t{acc}\n')

    # Log
    if is_main_process():
        logger.info("***** Results *****")
        logger.info(f'Test loss: {loss}')
        logger.info(f'Test acc:  {acc}')
        for key, val in result.items():
            logger.info(f'{key}: {val}')

    logger.info('Test finished')


def main(args):
    logger.info("Arguments:")
    logger.info(json.dumps(vars(args), indent=4))

    # Setup output files
    output_dir = os.path.join(args.output_dir, str(args.seed))
    os.makedirs(output_dir, exist_ok=True)
    filename_params = os.path.join(output_dir, consts.FILENAME_PARAMS)
    json.dump(vars(args), open(filename_params, 'w'), indent=4)


    # Sanity check on arguments
    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_test` must be True.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                             args.gradient_accumulation_steps))
    if args.gradient_accumulation_steps > args.train_batch_size:
        raise ValueError("gradient_accumulation_steps ({}) cannot be larger "
                         "train_batch_size ({}) - there cannot be a fraction "
                         "of one sample.".format(
                             args.gradient_accumulation_steps,
                             args.train_batch_size,
                         ))
    args.train_batch_size = (args.train_batch_size //
                             args.gradient_accumulation_steps)

    # Set seed
    if args.do_train:
        train(args)
    if args.do_test:
        test(args)
    print('DONE')


if __name__ == "__main__":
    main(parse_args())
