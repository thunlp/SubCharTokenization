#!/usr/bin/env bash

model_name="bert-tiny"

task_name=${task_name:-"chid"}

# init_checkpoint=${init_checkpoint:-"results/checkpoints_raw_zh/ckpt_8601.pt"}
# config_file=${config_file:-"configs/bert_config_vocab30k.json"}
# vocab_file=${vocab_file:-"tokenizers/sp_raw_zh_30k.vocab"}
# vocab_model_file=${vocab_model_file:-"tokenizers/sp_raw_zh_30k.model"}
# tokenizer_type=${tokenizer_type:-"RawZh"}
init_checkpoint=${init_checkpoint:-""}
config_file=${config_file:-""}
vocab_file=${vocab_file:-""}
vocab_model_file=${vocab_model_file:-""}
tokenizer_type=${tokenizer_type:-""}


data_dir=${data_dir:-""}
# train_dir=${train_dir:-""}
# dev_dir=${dev_dir:-""}
# test_dir=${test_dir:-""}
out_dir=${out_dir:-"logs/$task_name"}

seed=${seed:-""}
epochs=${epochs:-""}
batch_size=${batch_size:-24}

CMD="python3 "
CMD+="run_multichoice_mrc.py "

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
  CMD+="--predict_batch_size=$batch_size "
fi

CMD+="--seed=${seed} "
CMD+="--vocab_file=${vocab_file} "
CMD+="--vocab_model_file=${vocab_model_file} "
CMD+="--tokenizer_type=${tokenizer_type} "
CMD+="--config_file=${config_file} "
CMD+="--init_checkpoint=${init_checkpoint} "
CMD+="--data_dir=${data_dir} "
CMD+="--output_dir=${out_dir} "

CMD+="--train_file=${data_dir}/train.json "
CMD+="--train_ans_file=${data_dir}/train_answer.json "
CMD+="--dev_file=${data_dir}/dev.json "
CMD+="--dev_ans_file=${data_dir}/dev_answer.json "

CMD+="--num_train_epochs=4 "
# CMD+="--train_batch_size=24 "
# CMD+="--predict_batch_size=24 "
CMD+="--gradient_accumulation_steps=12 "

CMD+="--learning_rate=2e-5 "
CMD+="--warmup_proportion=0.06 "
CMD+="--max_seq_length=96 "


LOGFILE="${out_dir}/${seed}/logfile"

$CMD |& tee $LOGFILE


# python3 run_multichoice_mrc.py \
#   --num_train_epochs=4 \
#   --train_batch_size=24 \
#   --predict_batch_size=24 \
#   --gradient_accumulation_steps=12 \
#   --learning_rate=2e-5 \
#   --warmup_proportion=0.06 \
#   --max_seq_length=64 \
#   --vocab_file=${vocab_file} \
#   --vocab_model_file=${vocab_model_file} \
#   --tokenizer_type=${tokenizer_type} \
#   --config_file=${config_file} \
#   --init_checkpoint=${init_checkpoint} \
#   --data_dir=${data_dir} \
#   --output_dir=${out_dir} \
#   --train_file=${data_dir}/train.json \
#   --train_ans_file=${data_dir}/train_answer.json \
#   --dev_file=${data_dir}/dev.json \
#   --dev_ans_file=${data_dir}/dev_answer.json \
#   --seed=${seed}
  # --gpu_ids="0,1,2,3" \

# python test_multichoice_mrc.py \
#   --gpu_ids="0" \
#   --predict_batch_size=24 \
#   --max_seq_length=64 \
#   --vocab_file=$BERT_DIR/vocab.txt \
#   --bert_config_file=$BERT_DIR/bert_config.json \
#   --input_dir=$GLUE_DIR/$TASK_NAME/ \
#   --init_restore_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
#   --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
#   --predict_file=$GLUE_DIR/$TASK_NAME/test.json \
