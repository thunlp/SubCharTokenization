#!/bin/bash

# Dataset
task_name=${task_name:-""}
train_dir=${train_dir:-""}
dev_dir=${dev_dir:-""}
test_dir=${test_dir:-""}

seed=${seed:-""}
out_dir=${out_dir:-""}
mode=${mode:-"test"}

# Hyperparameters
epochs=${epochs:-"12"}
max_steps=${13:-"-1.0"}
batch_size=${batch_size:-"32"}
gradient_accumulation_steps=${gradient_accumulation_steps:-"2"}
learning_rate=${10:-"2e-5"}
warmup_proportion=${11:-"0.1"}
max_seq_length=${max_seq_length:-512}
fewshot=${fewshot:-1}

CMD="python3"
CMD+=" run_ner.py "
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
CMD+="--gradient_accumulation_steps=$gradient_accumulation_steps "

CMD+="--tokenizer_type $tokenizer_type "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$config_file "
CMD+="--vocab_model_file $vocab_model_file "
CMD+="--init_checkpoint $init_checkpoint "

CMD+="--output_dir $out_dir "
CMD+="--train_dir $train_dir "
CMD+="--dev_dir $dev_dir "
CMD+="--test_dir $test_dir "
CMD+="--seed $seed "
CMD+="--epochs $epochs "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--train_max_seq_length $max_seq_length "
CMD+="--eval_max_seq_length $max_seq_length "
CMD+="--learning_rate $learning_rate "

CMD+="--do_lower_case "

LOGFILE="${out_dir}/${seed}/logfile"

$CMD |& tee $LOGFILE