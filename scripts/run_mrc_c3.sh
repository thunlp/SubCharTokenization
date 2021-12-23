#!/usr/bin/env bash
model_name="bert-tiny"

task_name=${task_name:-"c3"}

init_checkpoint=${init_checkpoint:-"results/checkpoints_wubi_zh/ckpt_8601.pt"}
config_file=${config_file:-"configs/bert_config_vocab30k.json"}
vocab_file=${vocab_file:-"tokenizers/sp_wubi_zh_30k_sep.vocab"}
vocab_model_file=${vocab_model_file:-"tokenizers/sp_wubi_zh_30k_sep.model"}
tokenizer_type=${tokenizer_type:-"WubiZh"}
# init_checkpoint=${init_checkpoint:-"results/checkpoints_bert_zh_22675/ckpt_8601.pt"}
# config_file=${config_file:-"configs/bert_config_vocab22675.json"}
# vocab_file=${vocab_file:-"tokenizers/bert_chinese_uncased_22675.vocab"}
# vocab_model_file=${vocab_model_file:-"tokenizers/bert_chinese_uncased_22675.model"}
tokenizer_type=${tokenizer_type:-"BertZh"}

data_dir=${data_dir:-"datasets/$task_name/split"}
out_dir=${out_dir:-"logs/$task_name/wubi_zh"}

mode=${mode:-"train eval test"}
seed=${seed:-2}
epochs=${epochs:-6}
# epochs=${epochs:-8}
test_model=${test_model:-""}
cws_vocab_file=${cws_vocab_file:-""}

mkdir -p $out_dir

CMD="python3 "
CMD+="run_c3.py "

# CMD+="--task_name ${task_name} "
if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
  # CMD+="--train_batch_size=$batch_size "
fi
if [[ $mode == *"eval"* ]] || [[ $mode == *"test"* ]]; then
  if [[ $mode == *"eval"* ]] ; then
    CMD+="--do_eval "
  fi
  if [[ $mode == *"test"* ]] ; then
    CMD+="--do_test "
  fi
  # CMD+="--eval_batch_size=$batch_size "
fi

if [[ $test_model != "" ]] ; then
  CMD+="--test_model=${test_model} "
fi



CMD+="--num_train_epochs=${epochs} "
CMD+="--seed=${seed} "
CMD+="--tokenizer_type=${tokenizer_type} "
CMD+="--vocab_file=${vocab_file} "
CMD+="--vocab_model_file=${vocab_model_file} "
CMD+="--config_file=${config_file} "
CMD+="--init_checkpoint=${init_checkpoint} "
CMD+="--data_dir=${data_dir} "
CMD+="--output_dir=${out_dir} "

CMD+="--train_batch_size=24 "
CMD+="--eval_batch_size=24 "
CMD+="--gradient_accumulation_steps=12 "
CMD+="--learning_rate=2e-5 "
CMD+="--warmup_proportion=0.05 "
CMD+="--max_seq_length=512 "
if [[ $cws_vocab_file != "" ]] ; then
  CMD+="--cws_vocab_file $cws_vocab_file "
fi

LOGFILE=$out_dir/$seed/logfile

$CMD |& tee $LOGFILE

#   --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
#   --gpu_ids="0,1,2,3" \
