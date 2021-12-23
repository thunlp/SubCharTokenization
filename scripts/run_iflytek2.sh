#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

set -e

echo "Container nvidia build = " $NVIDIA_BUILD_ID

# init_checkpoint=${1:-"results/checkpoints_bert_zh_22675/ckpt_8601.pt"}
init_checkpoint=${1:-"logs/iflytek_bert_zh/pytorch_model.bin_3"}
# init_checkpoint=${1:-"/data04/scl/bert-base-chinese"}
data_dir=${2:-"/mnt/nfs/home/scl/iflytek_public/iflytek"}
# data_dir=${2:-"/mnt/nfs/home/scl/iflytek_public/rare"}
vocab_file=${3:-"tokenizers/bert_chinese_uncased_22675.vocab"}
# vocab_file=${3:-"/data04/scl/bert-base-chinese/vocab.txt"}
config_file=${4:-"bert_config_vocab22675.json"}
# config_file=${4:-"/data04/scl/bert-base-chinese/config.json"}
out_dir=${5:-"logs/iflytek_bert_zh"}
task_name=${6:-"iflytek"}
num_gpu=${7:-"4"}
batch_size=${8:-"64"}
gradient_accumulation_steps=${9:-"1"}
learning_rate=${10:-"2e-5"}
warmup_proportion=${11:-"0.1"}
epochs=${12:-"4"}
max_steps=${13:-"-1.0"}
precision=${14:-"fp16"}
seed=${15:-"2"}
mode=${16:-"prediction"}
tokenizer_type=${17:-"BertZh"}
vocab_model_file=${18:-"tokenizers/sp_raw_zh_30k.model"}

mkdir -p $out_dir

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
  mpi_command=" -m torch.distributed.launch --master_port=23333 --nproc_per_node=$num_gpu"
fi

CMD="python3 $mpi_command run_glue.py "
CMD+="--task_name ${task_name} "
if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
fi
if [[ $mode == *"eval"* ]] || [[ $mode == *"prediction"* ]]; then
  if [[ $mode == *"eval"* ]] ; then
    CMD+="--do_eval "
  fi
  if [[ $mode == *"prediction"* ]] ; then
    CMD+="--do_predict "
  fi
  CMD+="--eval_batch_size=$batch_size "
fi

CMD+="--gradient_accumulation_steps=$gradient_accumulation_steps "
CMD+="--do_lower_case "
CMD+="--tokenizer_type $tokenizer_type "
CMD+="--vocab_model_file $vocab_model_file "
CMD+="--data_dir $data_dir "
CMD+="--bert_model bert-tiny "
CMD+="--seed $seed "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--max_seq_length 256 "
CMD+="--learning_rate $learning_rate "
CMD+="--num_train_epochs $epochs "
CMD+="--max_steps $max_steps "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$config_file "
CMD+="--output_dir $out_dir "
CMD+="$use_fp16"

LOGFILE=$out_dir/logfile

$CMD |& tee $LOGFILE
