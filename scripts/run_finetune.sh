#!/bin/bash

set -e

echo "Container nvidia build = " $NVIDIA_BUILD_ID

# Model
init_checkpoint=${init_checkpoint:-"results/checkpoints_raw_zh/ckpt_8601.pt"}
config_file=${config_file:-"configs/bert_config_vocab30k.json"}
vocab_file=${vocab_file:-"tokenizers/sp_raw_zh_30k.vocab"}
vocab_model_file=${vocab_model_file:-"tokenizers/sp_raw_zh_30k.model"}
tokenizer_type=${tokenizer_type:-"RawZh"}

# init_checkpoint=${init_checkpoint:-"results/checkpoints_bert_zh_22675/ckpt_8601.pt"}
# config_file=${config_file:-"configs/bert_config_vocab22675.json"}
# vocab_file=${vocab_file:-"tokenizers/bert_chinese_uncased_22675.vocab"}
# vocab_model_file=${vocab_model_file:-"tokenizers/bert_chinese_uncased_22675.model"}
# tokenizer_type=${tokenizer_type:-"BertZh"}

# Dataset
task_name=${task_name:-"tnews"}
data_dir=${data_dir:-"datasets/$task_name/split"}
train_dir=${train_dir:-"datasets/$task_name/split"}
dev_dir=${dev_dir:-"datasets/$task_name/split"}
test_dir=${test_dir:-"datasets/$task_name/split"}

seed=${seed:-"2"}
out_dir=${out_dir:-"logs/${task_name}/wubi_zh"}
# mode=${mode:-"prediction"}
mode=${mode:-"test"}
num_gpu=${num_gpu:-"8"}

# Hyperparameters
epochs=${epochs:-"6"}
max_steps=${13:-"-1.0"}
batch_size=${batch_size:-"32"}
gradient_accumulation_steps=${gradient_accumulation_steps:-"2"}
learning_rate=${10:-"2e-5"}
warmup_proportion=${11:-"0.1"}
# max_seq_length=${max_seq_length:-128}
fewshot=${fewshot:-"0"}
two_level_embeddings=${two_level_embeddings:-"0"}
test_model=${test_model:-""}
cws_vocab_file=${cws_vocab_file:-""}
# precision=${14:-"fp16"}

echo "mode = $mode"

mkdir -p $out_dir
mkdir -p "$out_dir/$seed"


if [ "$mode" = "eval" ] ; then
  num_gpu=1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16="--fp16"
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" -m torch.distributed.launch --master_port=423333 --nproc_per_node=$num_gpu"
fi

# CMD="python $mpi_command run_glue.py "
CMD="python3"
# CMD="python"
CMD+=" run_glue.py "
CMD+="--task_name ${task_name} "
if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
fi
if [[ $mode == *"eval"* ]] || [[ $mode == *"test"* ]]; then
  if [[ $mode == *"eval"* ]] ; then
    CMD+="--do_eval "
  fi
  if [[ $mode == *"test"* ]] ; then
    CMD+="--do_test "
  fi
  CMD+="--eval_batch_size=$batch_size "
fi

if [[ $two_level_embeddings == "1" ]] ; then
  CMD+="--two_level_embeddings "
fi

CMD+="--tokenizer_type $tokenizer_type "
CMD+="--vocab_file=$vocab_file "
CMD+="--vocab_model_file $vocab_model_file "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--config_file=$config_file "

CMD+="--output_dir $out_dir "
# CMD+="--data_dir $data_dir "
CMD+="--train_dir $train_dir "
CMD+="--dev_dir $dev_dir "
CMD+="--test_dir $test_dir "

# CMD+="--bert_model bert-tiny "
CMD+="--seed $seed "

CMD+="--epochs $epochs "
CMD+="--warmup_proportion $warmup_proportion "
# CMD+="--max_seq_length $max_seq_length "
CMD+="--learning_rate $learning_rate "
CMD+="--gradient_accumulation_steps=$gradient_accumulation_steps "
# CMD+="--max_steps $max_steps "
if [[ $fewshot == "1" ]] ; then
  CMD+="--fewshot "
fi
CMD+="--fewshot $fewshot "
# CMD+="--do_lower_case "
if [[ $test_model != "" ]] ; then
  CMD+="--test_model $test_model "
fi
if [[ $cws_vocab_file != "" ]] ; then
  CMD+="--cws_vocab_file $cws_vocab_file "
fi
# CMD+="$use_fp16"

LOGFILE=$out_dir/$seed/logfile

$CMD |& tee $LOGFILE
