# Sub-Character Tokenization for Chinese Pretrained Language Models


This repo contains code for reproducing our results in the [paper](https://arxiv.org/abs/2106.00400) and using our models and tokenizers for your own tasks.
Model checkpoints are available at: https://huggingface.co/thunlp/SubCharTokenization/tree/main. That Huggingface repo only contains the model checkpoints, the config and tokenizer files are in this repo, which you can load locally. 

## Training the Tokenizers

### Transliterating Chinese Characters

Before training SubChar tokenizers, you need to first transliterate (i.e., encode) the Chinese characters. You can use the script `data/convert_corpus.py` to do so, note that you should specify in the script which transliration method you want to use (e.g., Pinyin, Wubi, etc.).


### Subword Tokenization on Transliterated Corpus

We use the SentencePiece library to train the tokenizers. The script we used is `tokenizers/train_sp.py`, which also contains all the hyper-parameters that we used. Notably, we used a vocab size of 22675 and character coverage of 1.00 for the training the SubChar tokenizers. 

The default subword tokenization implementation is unigram LM and you need to specify `model_type="bpe"` if you want to use the byte pair encoding implementation.


## Pretraining Transformers

### Pretraining Data Processing

The pretraining data that we used is from Baidu Baike, which consists of 2.2G raw text. You can download the raw text data from [this link](https://drive.google.com/file/d/1M9ipOApEDoIFpUHZxjb-HM9Zyj9vrsnn/view?usp=sharing) if you want to reproduce the results in the paper. Alternatively, you can use any other pretraining corpus you want, you should format the file by putting one document per line. Suppose the pretraining file is stored at `wubi_corpus/formatted/baidubaike_corpus.txt` (if not, substitute the directory in the data processing script).

Run `bash data/create_datasets_from_start.sh`. It consists of two data processing steps: 1) data sharding, and 2) creating HDF5 files. Note that we follow a two-stage pretraining pipeline where the first stage has max sequence length 128 and the second stage with max sequence length 512. 
You should also specify in that script the specific tokenizer vocab and model files that you want to use for tokenizing the corpus. Also, make sure that you are using the correct tokenizer class in `create_pretraining_data.py` (line 458-463) in the step of creating HDF5 files.

### Pretraining Commands

Once you have the processed data (the HDF5 files for the two stages of pretraining). Run `bash scripts/run_pretraining.sh` to run the pretraining. The default hyper-parameters in the script are used in the paper, you can also adjust them based on your own needs. Note that we used 8 A100 GPUs when doing the pretraining, you should adjust the batch sizes if you are using other GPUs.



## Finetuning Pretrained Models

You can run one of the following python code to do finetuning depending on which
task you want to finetune on. Note that different task/code might need different arguments. 

- `run_glue.py`: classification tasks such as TNews, IFlytek, OCNLI etc.
- `run_multichoice_mrc.py`: CHID
- `run_ner.py`: CLUENER
- `run_{cmrc, drcd, c3}.py`: CMRC, DRCD or C3

For example, for finetuning on TNews using pinyin tokenizer:

```bash
python3 run_glue.py \
  --task_name=tnews \
  --train_dir=datasets/tnews/split \
  --dev_dir=datasets/tnews/split \
  --test_dir=datasets/tnews/split \
  --do_train --do_eval --do_test \
  --init_checkpoint=checkpoints/checkpoints_pinyin_zh_22675/ckpt_8804.pt \
  --output_dir=logs/tnews/pinyin/ckpt_8804 \
  --tokenizer_type=CommonZh \
  --vocab_file=tokenizers/pinyin_zh_22675.vocab \
  --vocab_model_file=tokenizers/pinyin_zh_22675.model \
  --config_file=configs/bert_config_vocab22675.json \
  --epochs=6
```

Another example, finetuning on CMRC using wubi tokenizer:

```bash
python3 run_cmrc.py \
  --data_dir=datasets/cmrc/split \
  --init_checkpoint=checkpoints/checkpoints_wubi_zh_22675/ckpt_8804.pt \
  --config_file=configs/bert_config_vocab22675.json \
  --tokenizer_type=CommonZh \
  --vocab_file=tokenizers/wubi_zh_22675.vocab \
  --vocab_model_file=tokenizers/wubi_zh_22675.model \
  --output_dir=logs/cmrc/wubi_twolevel/ckpt_8804 \
  --do_train --do_test \
  --two_level_embeddings \
  --epochs=6
```


## References

Please consider citing our work if you found this code or our paper beneficial to your research.
```
@article{Si2021SubChar, 
  Author = {Chenglei Si and Zhengyan Zhang and Yingfa Chen and Fanchao Qi and Xiaozhi Wang and Zhiyuan Liu and Yasheng Wang and Qun Liu and Maosong Sun}, 
  Journal={arXiv preprint arXiv:2106.00400},  
  Year = {2021},  
  Title = {Sub-Character Tokenization for Chinese Pretrained Language Models} 
}    
