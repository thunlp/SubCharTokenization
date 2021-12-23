#!/usr/bin/env bash
model_name="bert-tiny"
task_name=${task_name:-"nulls"}
fewshot=${fewshot:-0}

init_checkpoint=${init_checkpoint:-""}
config_file=${config_file:-""}
vocab_file=${vocab_file:-""}
vocab_model_file=${vocab_model_file:-""}
tokenizer_type=${tokenizer_type:-""}

convert_to_simplified=${convert_to_simplified:-""}
two_level_embeddings=${two_level_embeddings:-""}
debug=${debug:-0}

data_dir=${data_dir:-""}

out_dir=${out_dir:-""}

mode=${mode:-"train test"}
seed=${seed:-""}
epochs=${epochs:-"6"}

mkdir -p $out_dir

CMD="python3 "
if [ "$task_name" = "drcd" ] ; then
  CMD+="run_drcd.py "
elif [ "$task_name" = "cmrc" ] ; then
  CMD+="run_cmrc.py "
else
  echo "INVALID task_name: $task_name"
fi

if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
fi

if [[ $mode == *"test"* ]] ; then
  CMD+="--do_test "
fi

if [ $two_level_embeddings -eq 1 ] ; then
  CMD+="--two_level_embeddings "
fi

if [ $debug -eq 1 ] ; then
  CMD+="--debug "
fi

# if [ $convert_to_simplified -eq 1 ] ; then
#   CMD+="--convert_to_simplified "
# fi

CMD+="--task_name=${task_name} "
# CMD+="--fewshot=${fewshot} "
CMD+="--tokenizer_type=${tokenizer_type} "
CMD+="--vocab_file=${vocab_file} "
CMD+="--vocab_model_file=${vocab_model_file} "
CMD+="--config_file=${config_file} "
CMD+="--init_checkpoint=${init_checkpoint} "
CMD+="--epochs=${epochs} "
CMD+="--seed=${seed} "
CMD+="--data_dir=${data_dir} "
CMD+="--output_dir=${out_dir} "

CMD+="--batch_size=32 "
CMD+="--gradient_accumulation_steps=8 "
CMD+="--lr=3e-5 "
CMD+="--warmup_rate=0.05 "
CMD+="--max_seq_length=512 "

LOGFILE=$out_dir/$seed/logfile

$CMD |& tee $LOGFILE

# python test_mrc.py \
#   --gpu_ids="0" \
#   --n_batch=32 \
#   --max_seq_length=512 \
#   --task_name=${task_name} \
#   --vocab_file=${vocab_file} \
#   --bert_config_file=${config_file} \
#   --init_restore_dir=$OUTPUT_DIR/$task_name/$MODEL_NAME/
#   --output_dir=$OUTPUT_DIR/$task_name/$MODEL_NAME/ \
#   --test_dir1=${data_dir}/test_examples.json \
#   --test_dir2=${data_dir}/test_features.json \
#   --test_file=${data_dir}/cmrc2018_test_2k.json \




