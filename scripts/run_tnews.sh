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

# Model
# init_checkpoint=${1:-"results/checkpoints_concat_sep/ckpt_8601.pt"}
# init_checkpoint=${1:-"logs/tnews_concat_sep/pytorch_model.bin_2"}
# init_checkpoint=${1:-"/data04/scl/bert-base-chinese"}
init_checkpoint=${init_checkpoint:-"logs/tnews_concat_sep/pytorch_model.bin_2"}
# config_file=${4:-"/data04/scl/bert-base-chinese/config.json"}
config_file=${config_file:-"configs/bert_config_vocab30k.json"}
# vocab_file=${3:-"/data04/scl/bert-base-chinese/vocab.txt"}
vocab_file=${vocab_file:-"tokenizers/sp_concat_30k_sep.vocab"}
vocab_model_file=${vocab_model_file:-"tokenizers/sp_concat_30k_sep.model"}
tokenizer_type=${tokenizer_type:-"ConcatSep"}

# Dataset
# data_dir=${2:-"/mnt/nfs/home/scl/tnews_public/tnews"}
# data_dir=${2:-"/mnt/nfs/home/scl/iflytek_public/rare"}
data_dir=${data_dir:-"results/datasets/tnews_public/tnews"}
task_name=${task_name:-"tnews"}

seed=${seed:-"2"}
out_dir=${out_dir:-"logs/tnews_concat_sep"}
# mode=${mode:-"prediction"}
mode=${mode:-"train eval"}
num_gpu=${num_gpu:-"1"}

# Hyperparameters
batch_size=${batch_size:-"32"}
gradient_accumulation_steps=${gradient_accumulation_steps:-"2"}
learning_rate=${10:-"2e-5"}
warmup_proportion=${11:-"0.1"}
epochs=${epochs:-"4"}
max_steps=${13:-"-1.0"}
# precision=${14:-"fp16"}

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
  mpi_command=" -m torch.distributed.launch --master_port=423333 --nproc_per_node=$num_gpu"
fi

CMD="python3 $mpi_command run_glue.py "
CMD+="--task_name ${task_name} "
if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
fi
if [[ $mode == *"eval"* ]] || [[ $mode == *"pred"* ]]; then
  if [[ $mode == *"eval"* ]] ; then
    CMD+="--do_eval "
  fi
  if [[ $mode == *"pred"* ]] ; then
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

LOGFILE=$out_dir/$seed/logfile

$CMD |& tee $LOGFILE
