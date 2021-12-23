# to_download=${1:-"wiki_only"}

# DATASET=wikicorpus_en

# # python3 bertPrepWiki.py --action create_hdf5_files --dataset $DATASET --max_seq_length 128 \
# # --max_predictions_per_seq 20 --vocab_file /home/sichenglei/WubiBERT/tokenizers/wiki_8k.vocab \
# # --model_file /home/sichenglei/WubiBERT/tokenizers/wiki_8k.model --do_macro

# python3 bertPrepWiki.py --action create_hdf5_files --dataset $DATASET --max_seq_length 128 \
# --max_predictions_per_seq 20 --vocab_file /home/sichenglei/WubiBERT/tokenizers/wiki_8k.vocab \
# --model_file /home/sichenglei/WubiBERT/tokenizers/wiki_8k.model 


DATASET=baidu_baike

python3 bertPrepWiki.py --action sharding --dataset $DATASET --vocab_file /home/sichenglei/WubiBERT/tokenizers/shuffled_pinyin_22675.vocab

python3 bertPrepWiki.py --action create_hdf5_files --dataset $DATASET --max_seq_length 128 \
--max_predictions_per_seq 20 --vocab_file /home/sichenglei/WubiBERT/tokenizers/shuffled_pinyin_22675.vocab --model_file /home/sichenglei/WubiBERT/tokenizers/shuffled_pinyin_22675.model

python3 bertPrepWiki.py --action create_hdf5_files --dataset $DATASET --max_seq_length 512 \
--max_predictions_per_seq 80 --vocab_file /home/sichenglei/WubiBERT/tokenizers/shuffled_pinyin_22675.vocab --model_file /home/sichenglei/WubiBERT/tokenizers/shuffled_pinyin_22675.model

