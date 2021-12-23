#!/bin/bash
#SBATCH
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -G 1
#SBATCH --no-requeue
#SSBATCH -o slurm_output/slurm-%j.out
#SSBATCH --gres=gpu:1

# cd /data/private/zhangzhengyan/projects/PLM-Task-Agnostic-Backdoor/src
# source /data/private/zhangzhengyan/miniconda3/bin/activate backdoor

dir_ckpts_sp=(
    "checkpoints/checkpoints_bert_zh_22675"
    "checkpoints/checkpoints_concat_sep"
    "checkpoints/checkpoints_raw_zh"
    "checkpoints/checkpoints_wubi_zh"
)

dir_ckpts=(
    "WubiBERT/results/checkpoints_cangjie_22675"
    "WubiBERT/wubi_results/checkpoints_pinyin_zh_22675"
    "WubiBERT/results/checkpoints_stroke_22675"
    "WubiBERT/wubi_results/checkpoints_wubi_zh_22675"
    "WubiBERT/wubi_results/checkpoints_zhengma_zh_22675"
    "WubiBERT/wubi_results/checkpoints_zhuyin_zh_22675"
    "WubiBERT/wubi_results/checkpoints_raw_zh_22675"
    "WubiBERT/results/checkpoints_bert_zh_22675"
    # cws
    "results/checkpoints_cws_raw_zh_22675"
    "results/checkpoints_cws_wubi_zh_22675"
    "results/checkpoints_cws_zhuyin_zh_22675"
)

dir_ckpts_long=(
    "" # cangjie
    "" # pinyin
    "" # stroke
    "" # wubi
    "" # zhengma
    "" # zhuyin
    "checkpoints_raw_zh_long" # raw
    "" # bert
    # cws
    ""
    ""
    ""
)

best_ckpts_long=(
    "" # cangjie
    "" # pinyin
    "" # stroke
    "" # wubi
    "" # zhengma
    "" # zhuyin

    # raw
    # "ckpt_6137"
    # "ckpt_7160"
    # "ckpt_8184"
    # "ckpt_9207"
    # "ckpt_10231"
    # "ckpt_11255"
    # "ckpt_12278"
    # "ckpt_13302"
    # "ckpt_14080"
    # "ckpt_15096"
    # "ckpt_16120"
    # "ckpt_17143"
    # "ckpt_18167"
    "ckpt_18200"

    "" # bert
)

best_ckpts_base=(
    "ckpt_8804" # cangjie
    "ckpt_8804" # pinyin
    "ckpt_8804" # stroke
    "ckpt_8804" # wubi
    "ckpt_8804" # zhengma
    "ckpt_8804" # zhuyin
    "ckpt_8804" # raw
    "ckpt_8804" # bert
)

best_ckpts=(
    # cangjie
    # "ckpt_7202"
    "ckpt_8804"

    "ckpt_8804" # pinyin
    "ckpt_8804" # stroke
    
    # wubi
    # "ckpt_7992"
    "ckpt_8804"  # This is best
    # "ckpt_8840"
    # "ckpt_8032"

    "ckpt_8804" # zhengma

    # zhuyin
    # "ckpt_8804"
    "ckpt_7992"
    
    # raw
    # "ckpt_7202"
    "ckpt_8804"

    "ckpt_8601" # bert
    # cws
    # "ckpt_7202" # cws_raw
    "ckpt_8804"
    # "ckpt_7993" # cws_wubi
    "ckpt_8804"
    "ckpt_8804" # cws_zhuyin
)

vocab_files_sp=(
    "/home/chenyingfa/WubiBERT/tokenizers/bert_chinese_uncased_22675.vocab"
    "/home/chenyingfa/WubiBERT/tokenizers/sp_concat_30k_sep.vocab"
    "/home/chenyingfa/WubiBERT/tokenizers/sp_raw_zh_30k.vocab"
    "/home/chenyingfa/WubiBERT/tokenizers/sp_wubi_zh_30k_sep.vocab"
)

vocab_model_files_sp=(
    "/home/chenyingfa/WubiBERT/tokenizers/bert_chinese_uncased_22675.model"
    "/home/chenyingfa/WubiBERT/tokenizers/sp_concat_30k_sep.model"
    "/home/chenyingfa/WubiBERT/tokenizers/sp_raw_zh_30k.model"
    "/home/chenyingfa/WubiBERT/tokenizers/sp_wubi_zh_30k_sep.model"
)

tokenizer_types_sp=(
    "BertZh"
    "ConcatSep"
    "RawZh"
    "WubiZh"
)

config_files_sp=(
    "configs/bert_config_vocab22675.json"
    "configs/bert_config_vocab30k.json"
    "configs/bert_config_vocab30k.json"
    "configs/bert_config_vocab30k.json"
)

tokenizer_names_sp=(
    "bert"
    "concat"
    "raw"
    "wubi"
)

vocab_files=(
    "/home/chenyingfa/WubiBERT/tokenizers/cangjie_zh_22675.vocab"
    "/home/chenyingfa/WubiBERT/tokenizers/pinyin_zh_22675.vocab"
    "/home/chenyingfa/WubiBERT/tokenizers/stroke_zh_22675.vocab"
    "/home/chenyingfa/WubiBERT/tokenizers/wubi_zh_22675.vocab"
    "/home/chenyingfa/WubiBERT/tokenizers/zhengma_zh_22675.vocab"
    "/home/chenyingfa/WubiBERT/tokenizers/zhuyin_zh_22675.vocab"
    "/home/chenyingfa/WubiBERT/tokenizers/raw_zh_22675.vocab"
    # "/home/chenyingfa/WubiBERT/tokenizers/sp_raw_zh_30k.vocab"
    "/home/chenyingfa/WubiBERT/tokenizers/bert_chinese_uncased_22675.vocab"
    # cws
    "/home/chenyingfa/WubiBERT/tokenizers/cws_raw_zh_22675.vocab"
    "/home/chenyingfa/WubiBERT/tokenizers/cws_wubi_zh_22675.vocab"
    "/home/chenyingfa/WubiBERT/tokenizers/cws_zhuyin_zh_22675.vocab"
)

vocab_model_files=(
    "/home/chenyingfa/WubiBERT/tokenizers/cangjie_zh_22675.model"
    "/home/chenyingfa/WubiBERT/tokenizers/pinyin_zh_22675.model"
    "/home/chenyingfa/WubiBERT/tokenizers/stroke_zh_22675.model"
    "/home/chenyingfa/WubiBERT/tokenizers/wubi_zh_22675.model"
    "/home/chenyingfa/WubiBERT/tokenizers/zhengma_zh_22675.model"
    "/home/chenyingfa/WubiBERT/tokenizers/zhuyin_zh_22675.model"
    "/home/chenyingfa/WubiBERT/tokenizers/raw_zh_22675.model"
    # "/home/chenyingfa/WubiBERT/tokenizers/sp_raw_zh_30k.model"
    "null"
    # cws
    "/home/chenyingfa/WubiBERT/tokenizers/cws_raw_zh_22675.model"
    "/home/chenyingfa/WubiBERT/tokenizers/cws_wubi_zh_22675.model"
    "/home/chenyingfa/WubiBERT/tokenizers/cws_zhuyin_zh_22675.model"
)

tokenizer_types=(
    "CommonZh"  # cangjie
    "CommonZh"  # pinyin
    "CommonZh"  # stroke
    "CommonZh"  # wubi
    "CommonZh"  # zhengma
    "CommonZh"  # zhuyin
    "RawZh"
    "BertZh"
    "CWSRawZh"
    "CWSCommonZh"
    "CWSCommonZh"
    "CommonZhNoIndex"   # pinyin_no_index
    "CommonZhNoIndex"   # wubi_no_index
    "PinyinConcatWubi"
)

# Will not be passed to script
tokenizer_names=(
    "cangjie"
    "pinyin"
    "stroke"
    "wubi"
    "zhengma"
    "zhuyin"
    "raw"
    # "raw8601"
    "bert"
    "cws_raw"
    "cws_wubi"
    "cws_zhuyin"
)


classification_tasks=(
    "tnews"
    "iflytek"
    "wsc"
    "afqmc"
    "csl"
    "ocnli"
)

# Change these
# task_name="tnews"
task_name="iflytek"
# task_name="wsc"
# task_name="afqmc"
# task_name="csl"
# task_name="ocnli"
task_name="cmrc"
# task_name="drcd"
# task_name="chid"
# task_name="c3"

# task_name="lcqmc"
# task_name="bq"
# task_name="thucnews"
# task_name="cluener"
# task_name="chinese_nli"  # Hu Hai's ChineseNLIProbing

# epochs=6  # All 6 classification tasks
# epochs=3  # cmrc
epochs=6
# epochs=6  # chid
# epochs=6  # C3
# batch_size=24

# Fewshot
fewshot=0       # 1 = true
# epochs=50
# batch_size=4

mode="train eval test"
# mode="train eval"
# mode="test"

debug=1
two_level_embeddings=1
dont_run=0
use_base=0
use_long=0
use_shuffled=0

run_in_bg=0
start_from_ckpt=0  # Not working yet
sleep_duration=10
use_sp=0
use_slurm=0

# Values of use_noise:
noise_suffix=""
# noise_suffix="_noise"
# noise_suffix="_noise_new"
# noise_suffix="_noise_random"
# noise_suffix="_noise_final"

if [ "$noise_suffix" != "" ] ; then
    noise_types=(
        # "glyph"
        "phonetic"
    )

    noise_train=(
        0
        # 20
        # 40
        # 60
        # 80
        # 100 # noise_random
    )
    noise_test=(
        # 0
        20
        40
        60
        80 # only applicable to iflytek
        # 100 # noise_random
    )
fi

classification_split_char=0

# CHID-only settings 
# NOTE: must manually change the same settings in "run_multichoice_mrc.py"
chid_use_shuffled_json=1
chid_split_char=0
chid_add_def=1

# DRCD-only settings
drcd_convert_to_simplified=1

seeds=(
    # "2"
    # "23"
    # "234"
    10 
    # 11 
    # 12 
    # 13 
    # 14 15 16 17 18 19 
    # 20 21 22 23 24 25 26 27 28 29
    # 30 31 32 33 34 35 36 37 38 39
)

# loop tokenizers
# (Use this range to choose tokenizers)
indices=(
    # 0 # cangjie
    1 # pinyin
    # 2 # stroke
    # 3 # wubi
    # 4 # zhengma
    # 5 # zhuyin
    # 6 # raw
    # 7 # bert
    # 8 # cws_raw
    # 9 # cws_wubi
    # 10 # cws_zhuyin
)

indices_sp=(
    # 0 # bert_chinese
    # 1 # concat_sp
    # 2 # raw_zh
    # 3 # wubi_zh
)


if [ $use_sp -eq 1 ] ; then
    arr_i=${indices_sp[@]}
else
    arr_i=${indices[@]}
fi

for i in ${arr_i[@]}
do
    # Don't change below
    for seed in ${seeds[@]}
    do
        # for noise_type in ${noise_types[@]}
        # do
        # for noise_amount_train in ${noise_train[@]}
        # do
        # for noise_amount_test in ${noise_test[@]}
        # do


        # Model
        if [ $use_sp -eq 1 ] ; then
            vocab_file="${vocab_files_sp[$i]}"
            vocab_model_file="${vocab_model_files_sp[$i]}"
            tokenizer_type="${tokenizer_types_sp[$i]}"
            tokenizer_name="${tokenizer_names_sp[$i]}"

        else
            vocab_file="${vocab_files[$i]}"
            vocab_model_file="${vocab_model_files[$i]}"
            tokenizer_type="${tokenizer_types[$i]}"
            tokenizer_name="${tokenizer_names[$i]}"
        fi


        # Get whether it is classification task
        is_classification=0
        for classification_task in ${classification_tasks[@]}
        do
            if [ "${task_name}" = "${classification_task}" ]  ; then
                is_classification=1
            fi
        done


        # Set train_dir, dev_dir, test_dir
        suf_split="/split"  # Whether data is manually split
        # LCQMC and BQ doesn't need data splitting, thus not split subdirectory
        if [ "$task_name" = "lcqmc" ] || [ "$task_name" = "bq" ] || [ "$task_name" = "thucnews" ] ; then
            suf_split=""
        fi
        # if [ $use_noise -ne 0 ] ; then
        if [ "$noise_suffix" != "" ] ; then

            # noise_suf="_noise"
            # if [ $use_noise -eq 3 ] ; then  # noise_random
            #     noise_suf="_noise_random"
            #     # if [ $noise_type = "phonetic" ] ; then
            #     #     if [ $tokenizer_name = "zhuyin" ] ; then
            #     #         noise_type="phonetic_zhuyin"
            #     #     elif [ $tokenizer_name = "pinyin" ] ; then
            #     #         noise_type="phonetic_pinyin"
            #     #     fi
            #     # fi 
            # elif [ $use_noise -eq 2 ] ; then  # noise_new
            #     noise_suf="_noise_new"
            # fi

            if [ $noise_amount_train -eq 0 ] ; then
                # if [ $use_noise -eq 3 ] ; then  # noise_random is a subset, must use dedicated clean data
                if [ "$noise_suffix" = "_noise_random" ] ; then
                    train_dir="datasets/${task_name}${noise_suffix}/${noise_type}_clean${suf_split}"
                else
                    train_dir="datasets/${task_name}${suf_split}"
                fi
            else
                train_dir="datasets/${task_name}${noise_suffix}/${noise_type}_${noise_amount_train}${suf_split}"
            fi
            
            # Same for test dir
            if [ $noise_amount_test -eq 0 ] ; then
                # if [ $use_noise -eq 3 ] ; then
                if [ "$noise_suffix" = "_noise_random" ] ; then
                    test_dir="datasets/${task_name}${noise_suffix}/${noise_type}_clean${suf_split}"
                else
                    test_dir="datasets/${task_name}${suf_split}"
                fi
            else
                test_dir="datasets/${task_name}${noise_suffix}/${noise_type}_${noise_amount_test}${suf_split}"
            fi
            dev_dir=${test_dir}
        else
            data_dir="datasets/${task_name}${suf_split}"
            train_dir="datasets/${task_name}${suf_split}"
            dev_dir="datasets/${task_name}${suf_split}"
            test_dir="datasets/${task_name}${suf_split}"
        fi
        if [ $fewshot -eq 1 ] ; then
            data_dir+="/fewshot"
            train_dir+="/fewshot"
            dev_dir+="/fewshot"
            test_dir+="/fewshot"
        fi

        # Set config_file, dir_ckpt and ckpt
        if [ $use_sp -eq 1 ] ; then
            # 分词模型
            config_file=${config_files_sp[$i]}
            dir_ckpt="${dir_ckpts_sp[$i]}"
            ckpt="ckpt_8601"
        else
            if [ $use_base -eq 1 ] ; then
                config_file="configs/bert_base_config.json"
                dir_ckpt="/home/chenyingfa/chinese_results/checkpoints_${tokenizer_name}_zh_base22675"
                ckpt=${best_ckpts_base[$i]}
            elif [ $use_long -eq 1 ] ; then
                config_file="configs/bert_config_vocab22675.json"
                dir_ckpt="/home/chenyingfa/${dir_ckpts_long[$i]}"
                ckpt=${best_ckpts_long[$i]}
            else 
                config_file="configs/bert_config_vocab22675.json"
                # config_file="configs/bert_config/vocab30k.json"
                dir_ckpt="/home/chenyingfa/${dir_ckpts[$i]}"
                if [ "$task_name" = "drcd" ] ; then
                    if [ $tokenizer_name == "bert" ] ; then
                        ckpt="ckpt_8601"
                    # elif [ $tokenizer_name = "zhuyin" ] ; then
                    #     ckpt="ckpt_7992"
                    else
                        ckpt="ckpt_7992"
                    fi
                else
                    ckpt="${best_ckpts[$i]}"
                fi
            fi
        fi

        # Set output_dir
        if [ $fewshot -eq 1 ] ; then
            task_name_in_output_dir="${task_name}_fewshot"
        else
            task_name_in_output_dir="${task_name}"
        fi 
        if [ $use_sp -eq 1 ] ; then  # 分词模型
            output_dir="logs/${task_name_in_output_dir}/sp/${tokenizer_name}"
        else
            if [ "$noise_suffix" = "" ] ; then # No noise
                output_dir="logs/${task_name_in_output_dir}/${tokenizer_name}"
            else
                if [ "$noise_suffix" = "_noise_random" ] ; then
                    if [ $noise_amount_train -eq 0 ] ; then
                        noise_amount_train="clean"
                    fi
                    if [ $noise_amount_test -eq 0 ] ; then
                        noise_amount_test="clean"
                    fi
                fi
                output_dir="logs/${task_name_in_output_dir}${noise_suffix}/${noise_type}_${noise_amount_train}_${noise_amount_test}/${tokenizer_name}"
            fi
            if [ ${is_classification} -eq 1 ] && [ ${classification_split_char} -eq 1 ] ; then
                output_dir+="_split_char"
            fi
            if [ $use_long -eq 1 ] ; then
                output_dir+="_long"
            fi
            if [ $use_base -eq 1 ] ; then
                output_dir+="_base"
            fi
            if [ $use_shuffled -eq 1 ] ; then
                output_dir+="_shuffled"
            fi

            if [ $two_level_embeddings -eq 1 ] ; then
                output_dir+="_twolevel"
            fi

            if [ "$task_name" = "chid" ] ; then
                if [ $chid_use_shuffled_json -eq 1 ] ; then
                    # output_dir="logs/${task_name}/${tokenizer_names[$i]}_pkl/$ckpt"
                    output_dir+="_shuffled"
                    # output_dir="logs/${task_name}/${tokenizer_name}_whole_shuffled/$ckpt"
                else
                    output_dir+="_unshuffled"
                fi

                if [ $chid_split_char -eq 0 ] ; then
                    output_dir+="_whole"
                fi

                if [ $chid_add_def -eq 1 ] ; then
                    output_dir+="_def"
                fi
            elif [ "$task_name" = "drcd" ] ; then
                if [ $drcd_convert_to_simplified -eq 0 ] ; then
                    output_dir+="_trad"
                else
                    output_dir+="_simp"
                fi
            fi
        fi
        output_dir+="/$ckpt"

        # Set checkpoint
        init_checkpoint="${dir_ckpt}/${ckpt}.pt"
        
        if [ $use_shuffled -eq 1 ] ; then
            init_checkpoint="/home/chenyingfa/checkpoints_shuffled_wubi/${ckpt}.pt"
        fi
        # init_checkpoint="checkpoints/checkpoints_raw_zh/ckpt_8601.pt"
        # init_checkpoint="checkpoints/checkpoints_bert_zh_22675/ckpt_8601.pt"

        # Task-specific settings
        if [ "$task_name" = "chid" ] || [ "$task_name" = "c3" ] ; then
            script="./scripts/run_mrc_${task_name}.sh"
        elif [ "$task_name" = "drcd" ] || [ "$task_name" = "cmrc" ] ; then
            script="./scripts/run_mrc_cmrc.sh"
        elif [ "$task_name" = "cluener" ] ; then
            script="./scripts/run_ner.sh"
            epochs=12
        else
            script="./scripts/run_finetune.sh"
            if [ "$task_name" = "wsc" ] ; then
                epochs=24       # WSC easily underfits
            elif [ "$task_name" = "thucnews" ] ; then
                epochs=4
            fi
        fi
    

        # Special cases. Should be removed?
        # if [ "${tokenizer_names[$i]}" = "raw8601" ] ; then
        #     # tokenizer_type="RawZh"
        #     vocab_file="tokenizers/sp_raw_zh_30k.vocab"
        #     vocab_model_file="tokenizers/sp_raw_zh_30k.model"
        #     config_file="configs/bert_config_vocab30k.json"
        #     init_checkpoint="checkpoints/checkpoints_raw_zh/ckpt_8601.pt"
        # elif [ "${tokenizer_name}" = "bert" ] ; then
        #     # tokenizer_type="BertZh"
        #     # vocab_file=${vocab_files[$i]}
        #     # vocab_model_file="null"
        #     if [ $use_base -eq 0 ] ; then
        #         init_checkpoint="checkpoints/checkpoints_bert_zh_22675/ckpt_8601.pt"
        #     fi
        # fi

        echo $script 
        echo "    Task:       $task_name"
        echo "    Vocab:      $vocab_file"
        echo "    Checkpoint: $init_checkpoint"
        echo "    Seed:       $seed"
        echo "    tokenizer_type: $tokenizer_type"
        echo "    train_dir:  $train_dir"
        echo "    dev_dir:    $dev_dir"
        echo "    test_dir:   $test_dir"
        echo "    out_dir:    $output_dir"
        echo "    epochs:     $epochs"
        # echo "    batch_size: $batch_size"

        # Export parameters
        export out_dir="$output_dir"
        export init_checkpoint="$init_checkpoint"
        export task_name="$task_name"
        export config_file="$config_file"
        export vocab_file="$vocab_file"
        export vocab_model_file="$vocab_model_file"
        export tokenizer_type="$tokenizer_type"
        export data_dir="$data_dir"
        export train_dir="$train_dir"
        export dev_dir="$dev_dir"
        export test_dir="$test_dir"
        export seed=$seed
        export epochs=$epochs
        export fewshot=$fewshot
        export convert_to_simplified=$drcd_convert_to_simplified
        export batch_size=$batch_size
        export mode="$mode"
        export classification_split_char=$classification_split_char
        export two_level_embeddings=$two_level_embeddings
        export debug=$debug

        # For testing
        if [ $dont_run -eq 1 ] ; then
            continue
        fi

        mkdir -p "$out_dir/$seed"

        if [ $use_slurm -eq 1 ] ; then
            mkdir -p "slurm_output/$task_name"
            mkdir -p "slurm_output/${task_name}/${tokenizer_names[$i]}"

            # # echo "$slurm_output_dir/slurm-%j.out"
            echo "    slurm output: slurm_output/${task_name}/${tokenizer_name}/seed${seed}-%j.out"
            sbatch -N 1 \
            -n 5 \
            -G 1 \
            --no-requeue \
            -o "slurm_output/${task_name}/${tokenizer_name}/seed${seed}-%j.out" \
            --export=init_checkpoint="$init_checkpoint",\
task_name="$task_name",\
config_file="$config_file",\
vocab_file="$vocab_file",\
vocab_model_file="$vocab_model_file",\
tokenizer_type="$tokenizer_type",\
out_dir="$output_dir",\
data_dir="$data_dir",\
train_dir="$train_dir",\
dev_dir="$dev_dir",\
test_dir="$test_dir",\
seed=$seed,\
epochs=$epochs,\
convert_to_simplified=$drcd_convert_to_simplified,\
batch_size=$batch_size,\
mode="$mode",\
classification_split_char=$classification_split_char,\
two_level_embeddings=$two_level_embeddings,\
fewshot=${fewshot},\
debug=${debug},\
            ${script}
        
        else
            LOGFILE="$out_dir/$seed/logfile.txt"
            
            if [ $run_in_bg -eq 1 ] ; then
                $script &> $LOGFILE &
                sleep $sleep_duration
            else
                $script
            fi

        fi


        # done
        # done
        # done
        # sleep 600
    done
done


            # if [ fewshot = 1 ] ; then
            #     slurm_output_dir="slurm_output/${task_name}/${tokenizer_names[$i]}/fewshot"
            # else
            #     slurm_output_dir="slurm_output/${task_name}/${tokenizer_names[$i]}"
            # fi

