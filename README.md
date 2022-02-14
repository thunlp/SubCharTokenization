# Code for running pretraining and finetuning of Chinese BERT model

Model checkpoints available at: https://huggingface.co/thunlp/SubCharTokenization/tree/main . That repo only contains the model checkpoints, the config and tokenizer files are in this repo, which you should load locally. 

Note that we split a fraction of the original CLUE training set and use as the dev set, we choose checkpoints based on results of that dev set and evaluate on the original CLUE dev set as the test set.

You can use ```split_data.py``` to do the dev set splitting, but remember to keep the random seed so that we can all reproduce the same splitting and results.


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

