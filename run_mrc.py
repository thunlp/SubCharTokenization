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

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import modeling
from tokenization import BertTokenizer, ConcatSepTokenizer, WubiZhTokenizer, RawZhTokenizer, BertZhTokenizer
from optimization import BertAdam, warmup_linear, get_optimizer
from schedulers import LinearWarmUpScheduler
from utils import mkdir


from mrc.google_albert_pytorch_modeling import AlbertConfig, AlbertForMRC
from mrc.preprocess.cmrc2018_evaluate import get_eval
# from mrc.pytorch_modeling import BertConfig, BertForQuestionAnswering, ALBertConfig, ALBertForQA
from mrc.tools import official_tokenization, utils
# from mrc.tools.pytorch_optimization import get_optimization, warmup_linear


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


ALL_TOKENIZERS = {
    "ConcatSep": ConcatSepTokenizer,
    "WubiZh": WubiZhTokenizer,
    "RawZh": RawZhTokenizer,
    "BertZh": BertZhTokenizer,
    "Bert": BertTokenizer,
    "BertHF": BertTokenizer
}


def evaluate(
    model, 
    args, 
    eval_examples, 
    eval_features, 
    device, 
    # global_steps, 
    epoch,
    output_dir,
    # best_f1, 
    # best_em, 
    # best_f1_em,
    # filename_scores
    ):
    print("\n***** Eval *****")
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    output_prediction_file = os.path.join(output_dir,
                                          'predictions_' + str(epoch) + '.json')
                                        #   "predictions_steps" + str(global_steps) + ".json")
    output_nbest_file = output_prediction_file.replace('predictions', 'nbest')

    all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.train_batch_size, shuffle=False)

    model.eval()
    all_results = []
    print("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    write_predictions(eval_examples, eval_features, all_results,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)

    tmp_result = get_eval(args.dev_file, output_prediction_file)
    tmp_result['STEP'] = global_steps
    # with open(filename_scores, 'a') as aw:
    #     # aw.write(json.dumps(tmp_result) + '\n')
    #     aw.write('\t' + str(tmp_result['EM']))
    print('')
    print(tmp_result)
    print('')

    # if float(tmp_result['F1']) > best_f1:
    #     best_f1 = float(tmp_result['F1'])

    # if float(tmp_result['EM']) > best_em:
    #     best_em = float(tmp_result['EM'])

    # if float(tmp_result['F1']) + float(tmp_result['EM']) > best_f1_em:
    #     best_f1_em = float(tmp_result['F1']) + float(tmp_result['EM'])
    #     utils.torch_save_model(model, args.checkpoint_dir,
    #                            {'f1': float(tmp_result['F1']), 'em': float(tmp_result['EM'])}, max_save_num=1)

    model.train()
    # return best_f1, best_em, best_f1_em
    return float(tmp_result['EM']), float(tmp_result['F1'])


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')

    # training parameter
    parser.add_argument('--train_epochs', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_rate', type=float, default=0.05)
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument('--float16', action='store_true', default=False)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--n_best', type=int, default=20)
    parser.add_argument('--eval_epochs', type=float, default=0.5)
    parser.add_argument('--save_best', type=bool, default=True)
    # parser.add_argument('--vocab_size', type=int, default=21128)
    parser.add_argument('--max_seq_length', type=int, default=256)

    parser.add_argument('--seed', type=int, required=True)
    # data dir
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--dev_dir1', type=str, required=True)
    parser.add_argument('--dev_dir2', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, required=True)
    # parser.add_argument('--bert_config_file', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    # parser.add_argument('--init_restore_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--task_name', type=str, required=True)

    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')

    parser.add_argument('--tokenizer_type', type=str, required=True)
    parser.add_argument('--vocab_model_file', type=str, required=True)
    parser.add_argument('--init_checkpoint', type=str, required=True)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='logs/temp')
    parser.add_argument("--do_train", action='store_true', default=False, help="Whether to run training.")
    parser.add_argument('--do_test', action='store_true', default=False, help='Whether to test.')
    
    return parser.parse_args()


def train(args):
    # Determine task: DRCD or CMRC
    if args.task_name.lower() == 'drcd':
        from preprocess.DRCD_output import write_predictions
        from preprocess.DRCD_preprocess import json2features
    elif args.task_name.lower() == 'cmrc':
        from mrc.preprocess.cmrc2018_output import write_predictions
        from mrc.preprocess.cmrc2018_preprocess import json2features
    else:
        raise NotImplementedError

    # Prepare files
    output_dir = os.path.join(args.out_dir, str(args.seed))
    filename_scores = os.path.join(output_dir, 'scores.txt')
    logger.info("Arguments:")
    logger.info(json.dumps(vars(args), indent=4))
    filename_params = os.path.join(output_dir, 'params.json')
    json.dump(vars(args), open(filename_params, 'w'), indent=4)  # Save arguments

    # Because tokenizer_type is a part of the feature file name,
    # new features will be generated for every tokenizer type.
    feature_file_suffix = 'features_' + str(args.max_seq_length) + '_' + args.tokenizer_type + '.json'
    example_file_suffix = 'examples_' + str(args.max_seq_length) + '_' + args.tokenizer_type + '.json'
    args.train_dir = args.train_dir.replace('features.json', feature_file_suffix)
    args.dev_dir1 = args.dev_dir1.replace('examples.json', example_file_suffix)
    args.dev_dir2 = args.dev_dir2.replace('features.json', feature_file_suffix)

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # # Clear log file content
    # with open(args.log_file, 'w') as f:
    #     f.write('')


    # Manage files
    if (os.path.exists(output_dir) and os.listdir(output_dir) and
        args.do_train):
        logger.warning("Output directory ({}) already exists and is not "
                        "empty.".format(output_dir))
    mkdir(output_dir)
    with open(filename_scores, 'w') as f:
        f.write('epoch\ttrain_loss\tdev_acc\n')  # Column names

    # args = utils.check_args(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    logger.info("device %s n_gpu %d" % (device, n_gpu))
    logger.info("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    # Tokenizer
    logger.info('Loading tokenizer...')
    tokenizer = ALL_TOKENIZERS[args.tokenizer_type](args.vocab_file, args.vocab_model_file)
    # assert args.vocab_size == len(tokenizer.vocab)

    # Load data
    logger.info('Loading data and features...')
    if True:
    # if not os.path.exists(args.train_dir):
        json2features(args.train_file, [args.train_dir.replace('_features_', '_examples_'), args.train_dir],
                      tokenizer, is_training=True,
                      max_seq_length=args.max_seq_length)
    if True:
    # if not os.path.exists(args.dev_dir1) or not os.path.exists(args.dev_dir2):
        json2features(args.dev_file, [args.dev_dir1, args.dev_dir2], tokenizer, is_training=False,
                      max_seq_length=args.max_seq_length)

    train_features = json.load(open(args.train_dir, 'r'))
    dev_examples = json.load(open(args.dev_dir1, 'r'))
    dev_features = json.load(open(args.dev_dir2, 'r'))


    # TODO: for debugging, remove in the end
    # print(train_features[0])
    # train_features = train_features[:10]
    # dev_examples = dev_examples[:10]
    # dev_features = dev_features[:10]
    # exit(0)

    steps_per_epoch = len(train_features) // args.train_batch_size
    eval_steps = int(steps_per_epoch * args.eval_epochs)
    dev_steps_per_epoch = len(dev_features) // args.train_batch_size
    if len(train_features) % args.train_batch_size != 0:
        steps_per_epoch += 1
    if len(dev_features) % args.train_batch_size != 0:
        dev_steps_per_epoch += 1
    total_steps = steps_per_epoch * args.train_epochs

    logger.info('steps per epoch: ' + str(steps_per_epoch))
    logger.info('total steps: ' + str(total_steps))
    logger.info('warmup steps: ' + str(int(args.warmup_rate * total_steps)))

    F1s = []
    EMs = []
    # 存一个全局最优的模型
    # best_f1_em = 0
    # best_f1, best_em = 0, 0

    # Set seed
    logger.info('SEED: ' + str(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    # print('init model...')
    # utils.torch_show_all_params(model)
    # utils.torch_init_model(model, args.init_restore_dir)
    logger.info("USING CHECKPOINT from {}".format(args.init_checkpoint))
    if args.tokenizer_type == 'BertHF':
        # model.load_state_dict(
        #     torch.load(args.init_checkpoint, map_location='cpu'),
        #     strict=False,
        # )
        # model = modeling.BertForSequenceClassification.from_pretrained(args.init_checkpoint, num_labels=num_labels)
        model = modeling.BertForQuestionAnswering(config)
    else:
        # model = modeling.BertForSequenceClassification(
        #     config,
        #     num_labels=num_labels,
        # )
        model = modeling.BertForQuestionAnswering(config)
        model.load_state_dict(
            torch.load(args.init_checkpoint, map_location='cpu')["model"],
            strict=False,
        )
    logger.info("USED CHECKPOINT from {}".format(args.init_checkpoint))


    if args.float16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    optimizer = get_optimizer(
        model=model,
        float16=args.float16,
        learning_rate=args.lr,
        total_steps=total_steps,
        schedule=args.schedule,
        warmup_rate=args.warmup_rate,
        max_grad_norm=args.clip_norm,
        weight_decay_rate=args.weight_decay_rate)


    # Decode features
    all_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features], dtype=torch.long)

    seq_len = all_input_ids.shape[1]
    # 样本长度不能超过bert的长度限制
    assert seq_len <= config.max_position_embeddings

    # true label
    all_start_positions = torch.tensor([f['start_position'] for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f['end_position'] for f in train_features], dtype=torch.long)

    # Train and evaluation
    if args.do_train:
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

        # Save config
        model_to_save = model.module if hasattr(model, 'module') else model
        filename_config = os.path.join(output_dir, modeling.CONFIG_NAME)
        with open(filename_config, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        logger.info('***** Training *****')
        # model.train()
        global_steps = 1
        best_em = 0
        best_f1 = 0

        dev_acc_history = []
        dev_f1_history = []


        for ep in range(args.train_epochs):
            logger.info('Starting epoch %d' % (ep + 1))
            total_loss = 0
            model.train()
            model.zero_grad()
            num_train_steps = 1
            with tqdm(total=steps_per_epoch, desc='Epoch %d' % (ep + 1)) as pbar:
                for step, batch in enumerate(train_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    
                    # print('input_ids')
                    # print(input_ids)
                    # print('input_mask')
                    # print(input_mask)
                    # print('start_positions')
                    # print(start_positions)
                    # print('end_positions')
                    # print(end_positions)

                    loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    cur_loss = loss.item()
                    total_loss += cur_loss
                    # print('total_loss:', total_loss)
                    pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss / (num_train_steps + 1e-5))})
                    pbar.update(1)

                    if args.float16:
                        optimizer.backward(loss)
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.lr * warmup_linear(global_steps / total_steps, args.warmup_rate)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    else:
                        loss.backward()

                    num_train_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        # model.zero_grad()
                        optimizer.zero_grad()
                        global_steps += 1


                    # if global_steps % eval_steps == 0:
                    #     best_f1, best_em, best_f1_em = evaluate(model, args, dev_examples, dev_features, device,
                    #                                             global_steps, best_f1, best_em, best_f1_em)
            # best_f1, best_em, best_f1_em = evaluate(
            #     model, args, dev_examples, dev_features, device,
            #     best_f1, best_em, best_f1_em)
            dev_acc, dev_f1 = evaluate(
                model, args, dev_examples, dev_features, 
                device, ep, output_dir
            )

            dev_acc /= 100
            dev_f1 /= 100

            dev_acc_history.append(dev_acc)
            dev_f1_history.append(dev_f1)

            train_loss = total_loss / steps_per_epoch
            
            with open(filename_scores, 'a') as f:
                f.write(f'{ep}\t{train_loss}\t{dev_acc}\n')

            # Save model
            model_to_save = model.module if hasattr(model, 'module') else model
            dir_models = os.path.join(output_dir, 'models')
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

        mean_acc = sum(dev_acc_history) / len(dev_acc_history)
        max_acc = max(dev_acc_history)
        mean_f1 = sum(dev_f1_history) / len(dev_f1_history)
        max_f1 = max(dev_f1_history)

        logger.info(f'Mean F1: {mean_f1} Mean EM: {mean_acc}')
        logger.info(f'Max F1: {max_f1} Max EM: {max_acc}')

    # F1s.append(max(dev_f1_history))
    # EMs.append(max(dev_acc_history))

    # with open(args.log_file, 'a') as aw:
    #     aw.write('Mean(Best) F1:{}({})\n'.format(np.mean(F1s), np.max(F1s)))
    #     aw.write('Mean(Best) EM:{}({})\n'.format(mean_acc, np.max(EMs)))

    # release the memory
    del model
    del optimizer
    torch.cuda.empty_cache()

    print('Training finished')
    

def test(args):
    raise NotImplementedError


def main():
    args = parse_args()
    train(args)
    print('DONE')


if __name__ == '__main__':
    main()