# Sub-Character Tokenization for Chinese Pretrained Language Models


This repo contains code for reproducing our results in the [paper](https://arxiv.org/abs/2106.00400) and using our models and tokenizers for your own tasks.
Model checkpoints are available at: https://huggingface.co/thunlp/SubCharTokenization/tree/main. That Huggingface repo only contains the model checkpoints, the config and tokenizer files are in this repo, which you can load locally. 

## Finetuning

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
