import collections
import logging
import os
import unicodedata
import six
from io import open
import pickle
import string

import sentencepiece as spm
import jieba
import oknlp

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def load_vocab_spm(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip().split()[0].strip()
            vocab[token] = index
            index += 1
    return vocab

# WUBI2CH = "/mnt/nfs/home/scl/LanguageModeling/BERT/data/wubi_to_chinese_unique.pkl"
# CH2WUBI = "/mnt/nfs/home/scl/LanguageModeling/BERT/data/chinese_to_wubi_unique.pkl"

# ENCODE2CH = "/home/ubuntu/WubiBERT/data/cangjie_to_chinese.pkl"
# CH2ENCODE = "/home/ubuntu/WubiBERT/data/chinese_to_cangjie.pkl"

cangjie2ch = "data/cangjie_to_chinese.pkl"
ch2cangjie = "data/chinese_to_cangjie.pkl"

stroke2ch = "data/stroke_to_chinese.pkl"
ch2stroke = "data/chinese_to_stroke.pkl"

zhengma2ch = "data/zhengma_to_chinese.pkl"
ch2zhengma = "data/chinese_to_zhengma.pkl"

wubi2ch = "data/wubi_to_chinese.pkl"
ch2wubi = "data/chinese_to_wubi.pkl"

pinyin2ch = "data/pinyin_to_chinese.pkl"
ch2pinyin = "data/chinese_to_pinyin.pkl"

zhuyin2ch = "data/zhuyin_to_chinese.pkl"
ch2zhuyin = "data/chinese_to_zhuyin.pkl"

# shuffle_map = "data/wubi_shuffle_dict.pkl"

# with open(shuffle_map, 'rb') as f:
#     shuffle_map = pickle.load(f)

control_char = u'0123456789abcdefghijklmnopqrstuvwxyz' 
control_uni = [chr(ord(c)+50000) for c in control_char]

CH2EN_PUNC = {f: t
              for f, t in zip(
                  u'，。！？【】（）％＃＠＆１２３４５６７８９０；：',
                  u',.!?[]()%#@&1234567890;:')}

puncs = list(CH2EN_PUNC.keys()) + list(CH2EN_PUNC.values()) + ['、', '“', '”', '》', '《', '·']

def load_dict(dict_path):
	return pickle.load(open(dict_path, "rb"))

vocab = load_vocab_spm('tokenizers/bert_chinese_uncased_22675.vocab')
vocab = list(vocab.keys())[5 : ]
vocab = [v for v in vocab if v not in puncs and (v not in control_char) and (v not in string.punctuation)]
vocab = [v for v in vocab if len(v) == 1]
# print (vocab[:257])

byte_char_map = {}
for i in range(257):
    byte_char_map[i] = vocab[i]

with open("byte_char_map.pkl", "wb") as f:
    pickle.dump(byte_char_map, f)

# with open("char_byte_map.pkl", "rb") as f:
#     char_byte_map = pickle.load(f)
# print (char_byte_map.keys())

# ## load some preprocessed dicts
# ch_chars = list(load_dict(ch2wubi).keys())
# ch_chars = list(set(ch_chars))
# SEP = chr(ord('_')+50000)

# with open("random_index_map.pkl", 'rb') as f:
#     random_index_map = pickle.load(f)

# ch_chars_inverse = {}
# for i in range(len(ch_chars)):
#     ch_chars_inverse[ch_chars[i]] = i

# def back2char(chars):
#     lst = []
#     for c in chars:
#         print (c, ch_chars_inverse[c])
#         lst.append(ch_chars_inverse[c])
#     byte = bytes(lst)
#     return byte.decode("utf-8")

# with open("tokenizers/byte_22675.vocab", "r") as f:
#     vocab = f.readlines()

# for line in vocab[5 : 10]:
#     print (back2char(line[:3]))