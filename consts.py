FILENAME_TEST_RESULT = 'result_test.txt'
FILENAME_SCORES = 'scores.txt'
FILENAME_BEST_MODEL = 'best_model.bin'
FILENAME_PARAMS = 'params.json'

VOCAB_FILES = {
    'cangjie': 'tokenizers/cangjie_zh_22675.vocab', 
    'pinyin': 'tokenizers/pinyin_zh_22675.vocab', 
    'stroke': 'tokenizers/stroke_zh_22675.vocab', 
    'wubi': 'tokenizers/wubi_zh_22675.vocab', 
    'zhengma': 'tokenizers/zhengma_zh_22675.vocab', 
    'zhuyin': 'tokenizers/zhuyin_zh_22675.vocab', 
    'raw': 'tokenizers/raw_zh_22675.vocab', 
    'bert': 'tokenizers/bert_chinese_uncased_22675.vocab', 
    'pinyin_concat_wubi': 'tokenizers/pinyin_concat_wubi_22675.vocab',
}

# Returns a string template
VOCAB_FILES_CWS = {
    'pinyin': 'cws_tokenizers/pinyin_cws_22675_{}.vocab',
    'wubi': 'cws_tokenizers/wubi_cws_22675_{}.vocab',
}

VOCAB_FILES_NO_INDEX = {
    'pinyin': 'tokenizers/pinyin_no_index_22675.vocab',
    'wubi': 'tokenizers/wubi_no_index_22675.vocab',
}

VOCAB_FILES_SHUFFLED = {
    'wubi': 'tokenizers/shuffled_wubi_22675.vocab',
    'pinyin': 'tokenizers/shuffled_pinyin_22675.vocab',
}

TOKENIZER_TYPES = {
    'cangjie': 'CommonZh',
    'pinyin': 'CommonZh',
    'stroke': 'CommonZh',
    'wubi': 'CommonZh',
    'zhengma': 'CommonZh',
    'zhuyin': 'CommonZh',
    'raw': 'RawZh',
    'bert': 'BertZh',
    'pinyin_no_index': 'CommonZhNoIndex',
    'wubi_no_index': 'CommonZhNoIndex',
    'pinyin_concat_wubi': 'PinyinConcatWubi',
}


MODEL_NAMES = [
    'cangjie',
    'pinyin',
    'stroke',
    'wubi',
    'zhengma',
    'zhuyin',
    'raw',
    'bert',
]

BEST_CKPTS = {
    # 'cangjie': "ckpt_7202",
    'cangjie': 'ckpt_8804',
    'pinyin': "ckpt_8804",
    'stroke': "ckpt_8804",
    # "ckpt_7992" # wubi
    'wubi': "ckpt_8804",
    'zhengma': "ckpt_8804",
    'zhuyin': "ckpt_7992",
    # 'zhuyin': 'ckpt_8804',
    # "ckpt_7202" # raw
    'raw': "ckpt_8804",
    'bert': "ckpt_8601", # bert
    # cws
    # "ckpt_7202" # cws_raw
    'cws_raw': "ckpt_8804",
    # "ckpt_7993" # cws_wubi
    'cws_wubi': "ckpt_8804",
    'cws_zhuyin': "ckpt_8804",
    'pinyin_concat_wubi': 'ckpt_8840',
}

BEST_CKPTS_NO_INDEX = {
    'pinyin': 'ckpt_8840',
    'wubi': 'ckpt_8840',
}

BEST_CKPTS_SHUFFLED = {
    'wubi': 'ckpt_8840',
    'pinyin': 'ckpt_8840',
}

BEST_CKPTS_CWS = {
    'pinyin': 'ckpt_8840',
    'wubi': 'ckpt_8840',
}

BEST_CKPT_BYTE = 'ckpt_8840'
BEST_CKPT_RANDOM_INDEX = 'ckpt_8840'
DIR_CKPTS_BYTE = 'checkpoints/ckpts_byte_22675'
DIR_CKPTS_RANDOM_INDEX = 'checkpoints/ckpts_random_index_22675'


DIR_CKPT_SP = {
    "bert": "checkpoints/checkpoints_bert_zh_22675",
    "concat_sep": "checkpoints/checkpoints_concat_sep",
    "raw": "checkpoints/checkpoints_raw_zh",
    "wubi": "checkpoints/checkpoints_wubi_zh",
}

DIR_CKPTS = {
    "cangjie": "checkpoints/checkpoints_cangjie_22675",
    "pinyin": "checkpoints/checkpoints_pinyin_zh_22675",
    "stroke": "checkpoints/checkpoints_stroke_22675",
    "wubi": "checkpoints/checkpoints_wubi_zh_22675",
    "zhengma": "checkpoints/checkpoints_zhengma_zh_22675",
    "zhuyin": "checkpoints/checkpoints_zhuyin_zh_22675",
    "raw": "checkpoints/checkpoints_raw_zh_22675",
    "bert": "checkpoints/checkpoints_bert_zh_22675",
    "pinyin_concat_wubi": "checkpoints/checkpoints_pinyin_concat_wubi",
}

DIR_CKPTS_CWS = {
    # "raw": "checkpoints/checkpoints_cws_raw_zh_22675",
    # "wubi": "checkpoints/checkpoints_cws_wubi_zh_22675",
    # "zhuyin": "checkpoints/checkpoints_cws_zhuyin_zh_22675",
    'pinyin': 'checkpoints/ckpts_pinyin_cws_22675',
    'wubi': 'checkpoints/ckpts_wubi_cws_22675',
}

DIR_CKPTS_LONG = {
    'raw': "checkpoints/checkpoints_raw_zh_long",
}

DIR_CKPTS_NO_INDEX = {
    'pinyin': 'checkpoints/checkpoints_pinyin_no_index',
    'wubi': 'checkpoints/checkpoints_wubi_no_index',
}

DIR_CKPTS_SHUFFLED = {
    'wubi': 'checkpoints/checkpoints_shuffled_wubi',
    'pinyin': 'checkpoints/checkpoints_shuffled_pinyin',
}