'''
Run BERT + CRF on NER tasks
'''
import glob
import logging
import os
import json
import time

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import consts
import modeling
import utils
from tokenization import ALL_TOKENIZERS
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler

from ner.callback.optimizater.adamw import AdamW
from ner.callback.lr_scheduler import get_linear_schedule_with_warmup
from ner.callback.progressbar import ProgressBar
from ner.tools.common import seed_everything,json_to_text
# from ner.tools.common import init_logger, logger
from ner.tools.finetuning_argparse import get_argparse

from ner.models.transformers import (
    # WEIGHTS_NAME, 
    BertConfig, 
    # AlbertConfig
)
from ner.models.bert_for_ner import BertCrfForNer
# from ner.models.albert_for_ner import AlbertCrfForNer
# from ner.processors.utils_ner import CNerTokenizer, get_entities
from ner.processors.utils_ner import get_entities, get_char_labels
from ner.processors.ner_seq import convert_examples_to_features
from ner.processors.ner_seq import ner_processors as processors
from ner.processors.ner_seq import get_collate_fn
from ner.metrics.ner_metrics import SeqEntityScore
from run_pretraining import pretraining_dataset, WorkerInitObj

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TWO_LEVEL_EMBEDDINGS = True
USE_TOKEN_EMBEDDINGS = True
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
PAD_TOKEN = '[PAD]'

collate_fn = get_collate_fn(TWO_LEVEL_EMBEDDINGS)


def train_old(args, train_dataset, model, tokenizer):
    """ Train the model """

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                  sampler=train_sampler, 
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    # if args.max_steps > 0:
    #     n_train_steps = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    # else:
    #     n_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    n_train_steps = len(train_dataloader) * args.epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ]
    args.warmup_steps = int(n_train_steps * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=n_train_steps)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    # # Distributed training (should be after apex fp16 initialization)
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    #                                                       output_device=args.local_rank,
    #                                                       find_unused_parameters=True)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(f'  Batch size = {args.train_batch_size}')
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             args.train_batch_size
    #             * args.gradient_accumulation_steps
    #             * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    #             )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", n_train_steps)

    global_step = 0
    # steps_trained_in_current_epoch = 0
    # # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
    #     # set global_step to gobal_step of last saved checkpoint from model path
    #     global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
    #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    train_loss, logging_loss = 0.0, 0.0

    train_loss_history = 0
    dev_loss_history = 0

    model.zero_grad()
    utils.set_seed(args.seed)
    # seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for ep in range(args.epochs):
        model.train()
        model.zero_grad()
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            # if steps_trained_in_current_epoch > 0:
            #     steps_trained_in_current_epoch -= 1
            #     continue
            # model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                'token_type_ids': batch[2],
                "labels": batch[3],
                'input_lens': batch[4],
            }
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}
            # if args.model_type != "distilbert":
            #     # XLM and RoBERTa don"t use segment_ids
            #     inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            # if args.n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # if args.gradient_accumulation_steps > 1:
            #     loss = loss / args.gradient_accumulation_steps
            # if args.fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     loss.backward()
            loss /= args.gradient_accumulation_steps
            loss.backward()

            pbar(step, {'loss': loss.item()})
            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1 == len(train_dataloader)):
                # if args.fp16:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                # else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     # Log metrics
                #     print(" ")
                #     if args.local_rank == -1:
                #         # Only evaluate when single GPU otherwise metrics may not average well
                #         evaluate(args, model, tokenizer)
                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = (
                #         model.module if hasattr(model, "module") else model
                #     )  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, "training_args.bin"))
                #     logger.info("Saving model checkpoint to %s", output_dir)
                #     tokenizer.save_vocabulary(output_dir)
                #     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                #     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                #     logger.info("Saving optimizer and scheduler states to %s", output_dir)
        logger.info("\n")

        # Evaluation
        logger.info('*** Evaluation ***')
        logger.info(f'  Epoch = {ep}')
        logger.info(f'  Num examples = {len(dev_examples)}')
        logger.info(f'  Batch size = {args.eval_batch_size}')

        

        if 'cuda' in str(device):
            torch.cuda.empty_cache()
    return global_step, train_loss / global_step


def get_char_preds(model, dataloader, device, id2label):
    char_preds = []
    token_preds = []
    for step, batch in enumerate(tqdm(dataloader, mininterval=8.0, desc='Evaluating')):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                'token_type_ids': batch[2],
                "labels": batch[3],
                'input_lens': batch[4],
            }
            outputs = model(**inputs)
            _, logits = outputs[:2]
            tags = model.crf.decode(logits, inputs['attention_mask'])
        # NOTE: labels of tokens might not match char labels from dataset
        labels = batch[5].cpu().numpy().tolist()
        tags = tags.squeeze(0).cpu().numpy().tolist()
        token_preds += tags

        # Convert to char labels, for exact comparison with labels in dataset
        left_index = batch[6].cpu().numpy().tolist()
        right_index = batch[7].cpu().numpy().tolist()
        old_tags = tags
        tags = [get_char_labels(tags[i], left_index[i], right_index[i], id2label) for i in range(len(tags))]
        
        # for i in range(len(tags)):
        #     if len(tags[i]) != len(labels[i]):
        #         print(tags[i])
        #         print(labels[i])
        #         exit()

        char_preds += tags
    return char_preds, token_preds


def get_examples(data_type, data_dir):
    processor = processors['cluener']()
    if data_type == 'train':
        examples = processor.get_train_examples(data_dir)
    elif data_type == 'dev':
        examples = processor.get_dev_examples(data_dir)
    else:
        examples = processor.get_test_examples(data_dir)
    return examples
 

def get_tokens(tokenizer, data_dir, data_type, max_seq_length):
    examples = get_examples(data_type, data_dir)
    all_tokens = []
    for ex in examples:
        tokens = tokenizer.tokenize(ex.text_a)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: (max_seq_length - 2)]
        all_tokens.append(['[CLS]'] + tokens + ['[SEP]'])
    return all_tokens


def get_truth(dataloader):
    char_labels = []
    for step, batch in enumerate(dataloader):
        out_label_ids = batch[5].cpu().numpy().tolist()  # Char labels
        char_labels += out_label_ids
    return char_labels


def diff_tokenizer(args, model, dataset, id2label, device, tokenizer):
    processor = processors['cluener']()
    tokenizer_bert = ALL_TOKENIZERS['BertZh'](
        "/home/chenyingfa/WubiBERT/tokenizers/bert_chinese_uncased_22675.vocab",
        "null")
    dataset_bert = get_dataset(
        args.task_name, 
        args.test_dir,
        tokenizer_bert, 
        tokenizer_name='bert', 
        data_type='test', 
        max_seq_len=args.eval_max_seq_length,
        two_level_embeddings=TWO_LEVEL_EMBEDDINGS)
    # Load best model
    best_model_filename = os.path.join('logs/cluener/bert/ckpt_8601/10/', consts.FILENAME_BEST_MODEL)
    logger.info(f'Loading model from "{best_model_filename}"')
    model_bert = load_model(args.config_file, best_model_filename, len(processor.get_labels()))
    logger.info(f'Loaded model')
    model_bert.to(device)

    dataloader_bert = DataLoader(
        dataset_bert,
        sampler=SequentialSampler(dataset_bert),
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn)

    metric = SeqEntityScore(id2label, markup=args.markup)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn)


    # Evaluation
    total_eval_loss = 0.0
    n_eval_steps = 0

    char_preds, token_preds = get_char_preds(model, dataloader, device, id2label)
    char_preds_bert, _ = get_char_preds(model_bert, dataloader_bert, device, id2label)
    truth = get_truth(dataloader_bert)
    examples = get_examples('test', args.test_dir)

    char_preds = char_preds[:50]
    char_preds_bert = char_preds_bert[:50]

    tokens = get_tokens(tokenizer, args.test_dir, 'test', args.eval_max_seq_length)
    tokens_bert = get_tokens(tokenizer_bert, args.test_dir, 'test', args.eval_max_seq_length)

    diff_idx = []
    for i in range(len(char_preds)):
        b = char_preds_bert[i][:len(char_preds[i])]
        if char_preds[i] != b:
            print('')
            print(f'Example {i}')
            print(f'  predictions:')
            print(f'    text:        {examples[i].text_a}')
            print(f'    truth:       {truth[i]}')
            print(f'    token preds: {token_preds[i]}')
            print(f'    pinyin:      {char_preds[i]}')
            print(f'    bert:        {b}\n')
            print(f'  tokens:')
            print(f'    pinyin:      {tokens[i]}')
            print(f'    bert:        {tokens_bert[i]}')
            diff_idx.append(i)
    
    print('')
    print(diff_idx, len(diff_idx))
    exit()


def evaluate(args, model, dataset, id2label, device, dump_preds=False, two_level_embeddings=True):
    metric = SeqEntityScore(id2label, markup=args.markup)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=args.eval_batch_size, 
        collate_fn=collate_fn)

    # Evaluation
    total_eval_loss = 0.0
    n_eval_steps = 0

    for step, batch in enumerate(tqdm(dataloader, mininterval=8.0, desc='Evaluating')):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                'token_type_ids': batch[2],
                "labels": batch[3],
                'input_lens': batch[4],
            }
            if two_level_embeddings:
                inputs['token_ids'] = batch[5]
                inputs['pos_right'] = batch[7]
                inputs['pos_left'] = batch[6]
                inputs['use_token_embeddings'] = USE_TOKEN_EMBEDDINGS
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags = model.crf.decode(logits, inputs['attention_mask'])
        # if args.n_gpu > 1:
        #     tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        total_eval_loss += tmp_eval_loss.item()
        n_eval_steps += 1

        # NOTE: labels of tokens might not match char labels from dataset
        out_label_ids = inputs['labels'].cpu().numpy().tolist()  # Token labels
        # out_label_ids = batch[5].cpu().numpy().tolist()  # Char labels

        # token_label_ids = inputs['labels'].cpu().numpy().tolist()  # Token labels
        # char_label_ids = batch[5].cpu().numpy().tolist()  # Char labels
        # for i in range(len(char_label_ids)):
        #     if token_label_ids[i] != char_label_ids[i]:
        #         print(token_label_ids[i])
        #         print(char_label_ids[i])
        #         exit()
        input_lens = inputs['input_lens'].cpu().numpy().tolist()
        tags = tags.squeeze(0).cpu().numpy().tolist()

        # Convert to char labels, for exact comparison with labels in dataset
        # inv_idx = inputs['inv_idx'].cpu().numpy().tolist()
        # left_index = batch[6].cpu().numpy().tolist()
        # right_index = batch[7].cpu().numpy().tolist()
        # old_tags = tags
        # tags = [get_char_labels(tags[i], left_index[i], right_index[i], id2label) for i in range(len(tags))]

        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    wrong_cnt = metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break                        
                else:
                    temp_1.append(id2label[out_label_ids[i][j]])
                    temp_2.append(id2label[tags[i][j]])

    logger.info("\n")
    eval_loss = total_eval_loss / n_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results *****")
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)

    return results


def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = get_dataset(args, args.task_name, tokenizer, data_type='test', two_level_embeddings=TWO_LEVEL_EMBEDDINGS)
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)
    results = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")

    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None, 'input_lens': batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask'])
            tags  = tags.squeeze(0).cpu().numpy().tolist()
        preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join([args.id2label[x] for x in preds])
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    logger.info("\n")
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    if args.task_name == 'cluener':
        output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
        test_text = []
        with open(os.path.join(args.data_dir,"test.json"), 'r') as fr:
            for line in fr:
                test_text.append(json.loads(line))
        test_submit = []
        for x, y in zip(test_text, results):
            json_d = {}
            json_d['id'] = x['id']
            json_d['label'] = {}
            entities = y['entities']
            words = list(x['text'])
            if len(entities) != 0:
                for subject in entities:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]
                    word = "".join(words[start:end + 1])
                    if tag in json_d['label']:
                        if word in json_d['label'][tag]:
                            json_d['label'][tag][word].append([start, end])
                        else:
                            json_d['label'][tag][word] = [[start, end]]
                    else:
                        json_d['label'][tag] = {}
                        json_d['label'][tag][word] = [[start, end]]
            test_submit.append(json_d)
        json_to_text(output_submit_file,test_submit)


def get_dataset(
    task,
    data_dir,
    tokenizer,
    tokenizer_name, 
    data_type,
    max_seq_len,
    two_level_embeddings=True,
    ):
    '''
    Generate dataset from data file.
    This will cache features using torch.save in data_dir.

    two_level_embeddings: Whether features should contain both split char tokens and ordinary tokens.
    '''
    # if args.local_rank not in [-1, 0] and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    use_cache = False
    
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_dir, 'cache_{}_{}_{}'.format(
        data_type,
        tokenizer_name,
        str(max_seq_len)))

    if use_cache and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(data_dir)
        else:
            examples = processor.get_test_examples(data_dir)
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            label_list=label_list,
            max_seq_length=max_seq_len,
            cls_token=CLS_TOKEN,
            sep_token=SEP_TOKEN,
            cls_token_at_end=False,
            cls_token_segment_id=0,
            # pad on the left for xlnet
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0],
            pad_token_segment_id=0,
            two_level_embeddings=two_level_embeddings,
        )
        # if args.local_rank in [-1, 0]:
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    # if args.local_rank == 0 and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)

    all_pos_left = torch.tensor([f.pos_left for f in features], dtype=torch.long)
    all_pos_right = torch.tensor([f.pos_right for f in features], dtype=torch.long)
    # all_subchar_pos = torch.tensor([f.subchar_pos for f in features], dtype=torch.long)
    if two_level_embeddings:
        all_token_ids = torch.tensor([f.token_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, 
                                all_lens, all_label_ids, all_token_ids,
                                all_pos_left, all_pos_right)
    else:
        all_char_label_ids = torch.tensor([f.char_label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, 
                                all_lens, all_label_ids, all_char_label_ids, 
                                all_pos_left, all_pos_right)

    return dataset


def get_optimizer_and_scheduler(model, lr, lr_crf, weight_decay, 
                                adam_eps, n_train_steps, n_warmup_steps):
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': lr},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': lr},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': lr_crf},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': lr_crf},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': lr_crf},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': lr_crf}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=n_warmup_steps,
                                                num_training_steps=n_train_steps)
    return optimizer, scheduler


def load_model(config_file, model_file, num_labels):
    config = modeling.BertConfig.from_json_file(config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = BertCrfForNer(config, num_labels=num_labels)
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict["model"], strict=False)
    return model


def train(args):
    logger.info('Training arguments:')
    logger.info(json.dumps(vars(args), indent=4))

    # Setup output dir
    tokenizer_name = utils.output_dir_to_tokenizer_name(args.output_dir)
    output_dir = os.path.join(args.output_dir, str(args.seed))
    os.makedirs(output_dir, exist_ok=True)
    filename_params = os.path.join(output_dir, consts.FILENAME_PARAMS)
    json.dump(vars(args), open(filename_params, 'w'), indent=4)

    args.train_batch_size //= args.gradient_accumulation_steps
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed
    logger.info(f'device: {device}')
    logger.info(f'Set seed: {args.seed}')
    utils.set_seed(args.seed)
    
    # logger.warning(
    #     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #     args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )

    # Load pretrained model and tokenizer
    tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}  # For evaluation
    num_labels = len(label_list)

    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # args.model_type = args.model_type.lower()
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
    #                                       num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    # tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #                                             do_lower_case=args.do_lower_case,
    #                                             cache_dir=args.cache_dir if args.cache_dir else None, )
    # model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
    #                                     config=config, cache_dir=args.cache_dir if args.cache_dir else None)
	# if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    logger.info(f'Loading model from "{args.init_checkpoint}"')
    model = load_model(args.config_file, args.init_checkpoint, num_labels)
    logger.info('Loaded model')
    model.to(device)

    # Save config
    filename_config = os.path.join(output_dir, modeling.CONFIG_NAME)
    with open(filename_config, 'w') as f:
        f.write(model.config.to_json_string())

    # Train and dev data
    # tokenizer_name = utils.output_dir_to_tokenizer_name(output_dir)
    train_dataset = get_dataset(
        args.task_name, 
        args.train_dir,
        tokenizer, 
        tokenizer_name=tokenizer_name,
        data_type='train',
        max_seq_len=args.train_max_seq_length,
        two_level_embeddings=TWO_LEVEL_EMBEDDINGS)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                  sampler=train_sampler, 
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    dev_dataset = get_dataset(
        # args, 
        args.task_name, 
        args.dev_dir,
        tokenizer,
        tokenizer_name=tokenizer_name,
        data_type='dev',
        max_seq_len=args.eval_max_seq_length,
        two_level_embeddings=TWO_LEVEL_EMBEDDINGS)

    # Optimizer
    n_train_steps = len(train_dataloader) * args.epochs
    n_warmup_steps = int(n_train_steps * args.warmup_proportion)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, 
        lr=args.learning_rate,
        lr_crf=args.crf_learning_rate,
        weight_decay=args.weight_decay,
        adam_eps=args.adam_epsilon,
        n_train_steps=n_train_steps,
        n_warmup_steps=n_warmup_steps,
    )

    # Training
    logger.info('*** Training ***')
    logger.info(f'  Num train examples = {len(train_dataset)}')
    logger.info(f'  Num dev examples = {len(dev_dataset)}')
    logger.info(f'  Num Epochs = {args.epochs}')
    logger.info(f'  Train batch size = {args.train_batch_size}')
    logger.info(f'  Dev batch size = {args.eval_batch_size}')
    logger.info(f'  Gradient Accumulation Steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total steps = {n_train_steps}')

    global_steps = 0
    cur_train_steps = 0
    total_train_loss = 0
    train_loss_history = []
    dev_loss_history = []
    dev_acc_history = []
    dev_f1_history = []
    model.zero_grad()
    utils.set_seed(args.seed)

    for ep in range(args.epochs):
        model.train()
        model.zero_grad()
        pbar = tqdm(train_dataloader, mininterval=8.0)
        for step, batch in enumerate(pbar):
            # if step == 0:
                # logger.info('Example:')
                # logger.info('input ids:')
                # logger.info(batch[0])
                # logger.info('labels:')
                # logger.info(batch[3])

            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                'token_type_ids': batch[2],
                "labels": batch[3],
                'input_lens': batch[4],
            }
            if TWO_LEVEL_EMBEDDINGS:
                inputs['token_ids'] = batch[5]
                inputs['pos_left'] = batch[6]
                inputs['pos_right'] = batch[7]
                inputs['use_token_embeddings'] = USE_TOKEN_EMBEDDINGS
                # 'left_indices': batch[6],
                # 'right_indices': batch[7],
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss /= args.gradient_accumulation_steps
            loss.backward()

            total_train_loss += loss.item()
            cur_train_steps += 1
            pbar.set_description(f'train_loss = {total_train_loss / cur_train_steps}')

            # BP
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1 == len(train_dataloader)):
                # if args.fp16:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                # else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_steps += 1

        train_loss = total_train_loss / (cur_train_steps + 1e-10)
        train_loss_history.append(train_loss)


        # Evaluation
        logger.info('*** Evaluation ***')
        logger.info(f'  Current epoch = {ep}')
        logger.info(f'  Num examples = {len(dev_dataset)}')
        logger.info(f'  Batch size = {args.eval_batch_size}')

        model.eval()
        total_dev_loss = 0
        cur_dev_steps = 0

        dev_result = evaluate(args, model, dev_dataset, id2label, device, TWO_LEVEL_EMBEDDINGS)
        dev_acc = dev_result['acc']
        dev_f1 = dev_result['f1']
        dev_loss = dev_result['loss']
        dev_acc_history.append(dev_acc)
        dev_f1_history.append(dev_f1)
        dev_loss_history.append(dev_loss)
    
        logger.info('*** Evaluation result ***')
        logger.info(f'  Current epoch = {ep}')
        logger.info(f'  Dev acc = {dev_acc}')
        logger.info(f'  Dev F1 = {dev_f1}')
        logger.info(f'  Dev loss = {dev_loss}')
        logger.info(f'  Train loss = {train_loss}')

        # Save to scores
        filename_scores = os.path.join(output_dir, consts.FILENAME_SCORES)
        with open(filename_scores, 'w') as f:
            f.write(f'epoch\ttrain_loss\tdev_loss\tdev_acc\tdev_f1\n')
            for i in range(ep + 1):
                train_loss = train_loss_history[i]
                dev_loss = dev_loss_history[i]
                dev_acc = dev_acc_history[i]
                dev_f1 = dev_f1_history[i]
                f.write(f'{i}\t{train_loss}\t{dev_loss}\t{dev_acc}\t{dev_f1}\n')
                

        # Save best model
        is_best = len(dev_f1_history) == 0 or dev_f1 == max(dev_f1_history)
        # is_best = len(dev_acc_history) == 0 or dev_acc == max(dev_acc_history)
        if is_best:
            model_dir = os.path.join(output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            best_model_filename = os.path.join(output_dir, consts.FILENAME_BEST_MODEL)
            torch.save(
                {'model': model.state_dict()},
                best_model_filename)
    
    logger.info('Training finished')


def test(args):
    logger.info('Testing start')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer_name = utils.output_dir_to_tokenizer_name(args.output_dir)
    output_dir = os.path.join(args.output_dir, str(args.seed))

    processor = processors[args.task_name]()
    tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load best model
    best_model_filename = os.path.join(output_dir, consts.FILENAME_BEST_MODEL)
    logger.info(f'Loading model from "{best_model_filename}"')
    model = load_model(args.config_file, best_model_filename, num_labels)
    logger.info(f'Loaded model')
    model.to(device)

    # Test data
    dataset = get_dataset(
        args.task_name,
        args.test_dir,
        tokenizer,
        tokenizer_name=tokenizer_name,
        data_type='test',
        max_seq_len=args.eval_max_seq_length,
        two_level_embeddings=TWO_LEVEL_EMBEDDINGS)

    # Test
    utils.set_seed(args.seed)
    logger.info('*** Testing ***')
    logger.info(f'  Num examples = {len(dataset)}')
    logger.info(f'  Batch size = {args.eval_batch_size}')
    # diff_tokenizer(args, model, dataset, id2label, device, tokenizer)
    result = evaluate(args, model, dataset, id2label, device, TWO_LEVEL_EMBEDDINGS)
    # exit()
    
    acc = result['acc']
    f1 = result['f1']
    loss = result['loss']

    logger.info('*** Test result ***')
    logger.info(f'  acc = {acc}')
    logger.info(f'  f1 = {f1}')
    logger.info(f'  loss = {loss}')

    # Save result
    file_test_result = os.path.join(output_dir, consts.FILENAME_TEST_RESULT)
    with open(file_test_result, 'w') as f:
        f.write(f'test_loss\ttest_acc\ttest_f1\n')
        f.write(f'{loss}\t{acc}\t{f1}\n')

    logger.info('Testing finished')


def main(args):
    assert args.do_train or args.do_test, 'At least one of `do_train` and `do_test` has to be true.'
    assert 1 <= args.gradient_accumulation_steps <= args.train_batch_size

    if args.do_train:
        train(args)
    
    if args.do_test:
        test(args)

    logger.info('DONE')


if __name__ == "__main__":
    main(get_argparse().parse_args())
