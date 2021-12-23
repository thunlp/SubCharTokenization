# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
import six
from io import open
import pickle

import sentencepiece as spm
import jieba
import oknlp

from file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
}
VOCAB_NAME = 'vocab.txt'

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

def load_dict(dict_path):
	return pickle.load(open(dict_path, "rb"))

## load some preprocessed dicts
with open("byte_char_map.pkl", "rb") as f:
    ch_chars = pickle.load(f)
SEP = chr(ord('_')+50000)

with open("random_index_map.pkl", 'rb') as f:
    random_index_map = pickle.load(f)

# map_dict = load_dict(CH2ENCODE)


class ByteTokenizer(object):

    def __init__(self, vocab_file, model_file, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)) or (not os.path.isfile(model_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))

        self.vocab = load_vocab_spm(vocab_file)
        self.spm_tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.max_len = max_len if max_len is not None else int(1e12)
    
    def convert_line(self, text):
        out_line = "" 
        for ch_char in text.strip():
            c = bytes(ch_char, 'utf-8')
            for byte_index in c:
                # print (byte_index)
                ch = ch_chars[byte_index]
                out_line += ch 
            out_line += SEP
        return out_line
    
    def tokenize(self, text):
        out_line = self.convert_line(text)
        return self.spm_tokenizer.encode(out_line, out_type=str)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
    




class RandomIndexTokenizer(object):

    def __init__(self, vocab_file, model_file, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)) or (not os.path.isfile(model_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))

        self.vocab = load_vocab_spm(vocab_file)
        self.spm_tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.max_len = max_len if max_len is not None else int(1e12)
    
    def convert_line(self, text):
        out_line = "" 
        for ch_char in text.strip():
            if ch_char in random_index_map:
                out_line += str(random_index_map[ch_char])
            else:
                out_line += ch_char
            out_line += SEP
        return out_line
    
    def tokenize(self, text):
        out_line = self.convert_line(text)
        return self.spm_tokenizer.encode(out_line, out_type=str)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
    


class BertZhTokenizer(object):
    "for bert_chinese_uncased_22675 tokenization"

    def __init__(self, vocab_file, model_file=None, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))
        self.vocab = load_vocab_spm(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.max_len = max_len if max_len is not None else int(1e12) 
    
    def tokenize(self, line):
        line = line.lower().replace(' ', '')
        line = list(line.strip())
        return line 
    
    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids
    
    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
        

class RawZhTokenizer(object):
    "for sp_raw_zh tokenization"

    def __init__(self, vocab_file, model_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)) or (not os.path.isfile(model_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))
        self.vocab = load_vocab_spm(vocab_file)
        self.spm_tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_lower_case = do_lower_case
        # self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
        #                                       never_split=never_split)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)     

    def tokenize(self, text):
        if self.do_lower_case:
            text = text.lower() ## lowercasing doesn't matter much here
        return self.spm_tokenizer.encode(text, out_type=str)
    
    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids
    
    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens





class CommonZhNoIndexTokenizer(object):
    "for cangjie_zh, wubi_zh, ... all such tokenization"

    def __init__(self, vocab_file, model_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)) or (not os.path.isfile(model_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))
        if 'cangjie' in vocab_file:
            self.map_dict = load_dict(ch2cangjie)
        elif 'stroke' in vocab_file:
            self.map_dict = load_dict(ch2stroke)
        elif 'zhengma' in vocab_file:
            self.map_dict = load_dict(ch2zhengma)
        elif 'wubi' in vocab_file:
            self.map_dict = load_dict(ch2wubi)
        elif 'pinyin' in vocab_file:
            self.map_dict = load_dict(ch2pinyin)
        elif 'zhuyin' in vocab_file:
            self.map_dict = load_dict(ch2zhuyin)

        self.vocab = load_vocab_spm(vocab_file)
        self.spm_tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        # self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
        #                                       never_split=never_split)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)
    
    def convert_line(self, text):
        text = text.lower() #  always lowercasing
        out_line = "" 
        for ch_word in text:
            ch_char = ch_word.strip()
            if len(ch_char) == 0:
                continue
                
            ## all convert to EN punctuations,
            ## to avoid mixture of different punctuations
            if ch_char in CH2EN_PUNC:
                ch_char = CH2EN_PUNC[ch_char]

            if ch_char in self.map_dict:
                mapped = ''.join([c for c in self.map_dict[ch_char].strip() if not c.isdigit()])
                out_line += mapped + chr(ord('_')+50000) ## for sp_concat
            else:
                if ch_char in control_char:
                    ch_char = chr(ord(ch_char)+50000)
                out_line += ch_char  ## sp_concat
        return out_line
    
    def tokenize(self, text):
        out_line = self.convert_line(text)
        return self.spm_tokenizer.encode(out_line, out_type=str)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
    
    ## TODO: implement the detokenizer!





class PinyinConcatWubiTokenizer(object):
    "for cangjie_zh, wubi_zh, ... all such tokenization"

    def __init__(self, vocab_file, model_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)) or (not os.path.isfile(model_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))

        self.map_dict_pinyin = load_dict(ch2pinyin)
        self.map_dict_wubi = load_dict(ch2wubi)

        self.vocab = load_vocab_spm(vocab_file)
        self.spm_tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        # self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
        #                                       never_split=never_split)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)
    
    def convert_line(self, text):
        text = text.lower() #  always lowercasing
        out_line = "" 
        for ch_word in text:
            ch_char = ch_word.strip()
            if len(ch_char) == 0:
                continue
                
            ## all convert to EN punctuations,
            ## to avoid mixture of different punctuations
            if ch_char in CH2EN_PUNC:
                ch_char = CH2EN_PUNC[ch_char]

            if (ch_char in self.map_dict_pinyin) or (ch_char in self.map_dict_wubi):
                mapped = ''
                if ch_char in self.map_dict_pinyin:
                    mapped += ''.join([c for c in self.map_dict_pinyin[ch_char].strip() if not c.isdigit()])
                if ch_char in self.map_dict_wubi:
                    mapped += ''.join([c for c in self.map_dict_wubi[ch_char].strip() if not c.isdigit()])
                out_line += mapped + chr(ord('_')+50000) ## for sp_concat
            else:
                if ch_char in control_char:
                    ch_char = chr(ord(ch_char)+50000)
                out_line += ch_char  ## sp_concat
        return out_line
    
    def tokenize(self, text):
        out_line = self.convert_line(text)
        return self.spm_tokenizer.encode(out_line, out_type=str)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
    
    ## TODO: implement the detokenizer!







class ShuffledTokenizer(object):
    "for cangjie_zh, wubi_zh, ... all such tokenization"

    def __init__(self, vocab_file, model_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)) or (not os.path.isfile(model_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))
        if 'cangjie' in vocab_file:
            self.map_dict = load_dict(ch2cangjie)
        elif 'stroke' in vocab_file:
            self.map_dict = load_dict(ch2stroke)
        elif 'zhengma' in vocab_file:
            self.map_dict = load_dict(ch2zhengma)
        elif 'wubi' in vocab_file:
            self.map_dict = load_dict(ch2wubi)
            shuffle_map = "data/wubi_shuffle_dict.pkl"
        elif 'pinyin' in vocab_file:
            self.map_dict = load_dict(ch2pinyin)
            shuffle_map = "data/pinyin_shuffle_dict.pkl"
        elif 'zhuyin' in vocab_file:
            self.map_dict = load_dict(ch2zhuyin)

        self.vocab = load_vocab_spm(vocab_file)
        self.spm_tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.max_len = max_len if max_len is not None else int(1e12)

        with open(shuffle_map, 'rb') as f:
            self.shuffle_map = pickle.load(f)

    
    def convert_line(self, line):
        # text = text.lower() #  always lowercasing
        text = ""
        for c in line.strip():
            if c in self.shuffle_map:
                newc = self.shuffle_map[c]
                # print (c, newc)
            else:
                newc = c 
            text += newc
        # print (text)
        out_line = "" 
        for ch_word in text:
            ch_char = ch_word.strip()
            if len(ch_char) == 0:
                continue
                
            ## all convert to EN punctuations,
            ## to avoid mixture of different punctuations
            if ch_char in CH2EN_PUNC:
                ch_char = CH2EN_PUNC[ch_char]

            if ch_char in self.map_dict:
                # add _ at the end of each ZH char as seperation
                out_line += self.map_dict[ch_char].strip() + chr(ord('_')+50000) ## for sp_concat
            else:
                if ch_char in control_char:
                    ch_char = chr(ord(ch_char)+50000)
                out_line += ch_char  ## sp_concat
        return out_line
    
    def tokenize(self, text):
        out_line = self.convert_line(text)
        return self.spm_tokenizer.encode(out_line, out_type=str)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
    
    ## TODO: implement the detokenizer!
    



class RawEnTokenizer(object):
    "for sp_raw_zh tokenization"

    def __init__(self, vocab_file, model_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)) or (not os.path.isfile(model_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))
        self.vocab = load_vocab_spm(vocab_file)
        self.spm_tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        # self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
        #                                       never_split=never_split)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.do_lower_case = do_lower_case
        self.max_len = max_len if max_len is not None else int(1e12)     

    def tokenize(self, text):
        if self.do_lower_case:
            text = text.lower() 
        return self.spm_tokenizer.encode(text, out_type=str)
    
    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids
    
    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens











class WubiZhTokenizer(object):
    "for sp_wubi_zh (also have char sep) tokenization"

    def __init__(self, vocab_file, model_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)) or (not os.path.isfile(model_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))
        self.vocab = load_vocab_spm(vocab_file)
        self.spm_tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        # self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
        #                                       never_split=never_split)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)
    
    def convert_line(self, text):
        text = text.lower() #  always lowercasing
        out_line = " " # actually space doesn't matter
        for ch_word in text:
            ch_word = ch_word.strip()
            for ch_char in ch_word:
                if ch_char in map_dict:
                    out_line += map_dict[ch_char].strip() + chr(ord('_')+50000)
                else:
                    if ch_char in control_char:
                        ch_char = chr(ord(ch_char)+50000)
                    out_line += ch_char + ''
            # out_line += ' '
        return out_line
    
    def tokenize(self, text):
        out_line = self.convert_line(text)
        return self.spm_tokenizer.encode(out_line, out_type=str)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
    
    ## TODO: implement the detokenizer!


class CangjieTokenizer(object):
    "for cangjie_zh tokenization"

    def __init__(self, vocab_file, model_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)) or (not os.path.isfile(model_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))
        self.vocab = load_vocab_spm(vocab_file)
        self.spm_tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        # self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
        #                                       never_split=never_split)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)
    
    def convert_line(self, text):
        text = text.lower() #  always lowercasing
        out_line = "" 
        for ch_word in text:
            ch_char = ch_word.strip()
            if len(ch_char) == 0:
                continue
                
            ## all convert to EN punctuations,
            ## to avoid mixture of different punctuations
            if ch_char in CH2EN_PUNC:
                ch_char = CH2EN_PUNC[ch_char]

            if ch_char in map_dict:
                # add _ at the end of each ZH char as seperation
                out_line += map_dict[ch_char].strip() + chr(ord('_')+50000) ## for sp_concat
            else:
                if ch_char in control_char:
                    ch_char = chr(ord(ch_char)+50000)
                out_line += ch_char  ## sp_concat
        return out_line
    
    def tokenize(self, text):
        out_line = self.convert_line(text)
        return self.spm_tokenizer.encode(out_line, out_type=str)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
    
    ## TODO: implement the detokenizer!
    

class CommonZhTokenizer(object):
    "for cangjie_zh, wubi_zh, ... all such tokenization"

    def __init__(self, vocab_file, model_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)) or (not os.path.isfile(model_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))
        if 'cangjie' in vocab_file:
            self.map_dict = load_dict(ch2cangjie)
        elif 'stroke' in vocab_file:
            self.map_dict = load_dict(ch2stroke)
        elif 'zhengma' in vocab_file:
            self.map_dict = load_dict(ch2zhengma)
        elif 'wubi' in vocab_file:
            self.map_dict = load_dict(ch2wubi)
        elif 'pinyin' in vocab_file:
            self.map_dict = load_dict(ch2pinyin)
        elif 'zhuyin' in vocab_file:
            self.map_dict = load_dict(ch2zhuyin)

        self.vocab = load_vocab_spm(vocab_file)
        self.spm_tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        # self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
        #                                       never_split=never_split)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)
    
    def convert_line(self, text):
        text = text.lower() #  always lowercasing
        out_line = "" 
        for ch_word in text:
            ch_char = ch_word.strip()
            if len(ch_char) == 0:
                continue
                
            ## all convert to EN punctuations,
            ## to avoid mixture of different punctuations
            if ch_char in CH2EN_PUNC:
                ch_char = CH2EN_PUNC[ch_char]

            if ch_char in self.map_dict:
                # add _ at the end of each ZH char as seperation
                out_line += self.map_dict[ch_char].strip() + chr(ord('_')+50000) ## for sp_concat
            else:
                if ch_char in control_char:
                    ch_char = chr(ord(ch_char)+50000)
                out_line += ch_char  ## sp_concat
        return out_line
    
    def tokenize(self, text):
        out_line = self.convert_line(text)
        return self.spm_tokenizer.encode(out_line, out_type=str)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
    
    ## TODO: implement the detokenizer!
    





class ConcatSepTokenizer(object):
    "for sp_concat_sep (wubi) tokenization"

    def __init__(self, vocab_file, model_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)) or (not os.path.isfile(model_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))
        self.vocab = load_vocab_spm(vocab_file)
        self.spm_tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        # self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
        #                                       never_split=never_split)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)
    
    def convert_line(self, text):
        text = text.lower() # always lowercasing
        in_line = jieba.lcut(text)
        out_line = " " # actually space doesn't matter
        for ch_word in in_line:
            ch_word = ch_word.strip()
            for ch_char in ch_word:
                if ch_char in map_dict:
                    out_line += map_dict[ch_char].strip() + chr(ord('_')+50000)
                else:
                    if ch_char in control_char:
                        ch_char = chr(ord(ch_char)+50000)
                    out_line += ch_char + ''
            out_line += ' '
        return out_line

    def tokenize(self, text):
        out_line = self.convert_line(text)
        return self.spm_tokenizer.encode(out_line, out_type=str)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
    
    ## TODO: implement the detokenizer!


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, model_file=None, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            vocab_file = pretrained_model_name_or_path
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    vocab_file))
            return None
        if resolved_vocab_file == vocab_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
        return tokenizer


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class CWSNewTokenizer(object):

    def __init__(self, vocab_file, model_file, cws_vocab_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if (not os.path.isfile(vocab_file)) or (not os.path.isfile(model_file)):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained ".format(vocab_file))
        if 'cangjie' in vocab_file:
            self.map_dict = load_dict(ch2cangjie)
        elif 'stroke' in vocab_file:
            self.map_dict = load_dict(ch2stroke)
        elif 'zhengma' in vocab_file:
            self.map_dict = load_dict(ch2zhengma)
        elif 'wubi' in vocab_file:
            self.map_dict = load_dict(ch2wubi)
        elif 'pinyin' in vocab_file:
            self.map_dict = load_dict(ch2pinyin)
        elif 'zhuyin' in vocab_file:
            self.map_dict = load_dict(ch2zhuyin)

        self.vocab = load_vocab_spm(vocab_file)
        self.spm_tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        self.cws_vocab = load_vocab(cws_vocab_file)
        self.seg = oknlp.algorithm.cws.get_by_name('thulac')

        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        # self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
        #                                       never_split=never_split)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)
    
    def convert_line(self, text):
        text = text.lower() #  always lowercasing
        out_line = "" 
        for ch_word in text:
            ch_char = ch_word.strip()
            if len(ch_char) == 0:
                continue
                
            ## all convert to EN punctuations,
            ## to avoid mixture of different punctuations
            if ch_char in CH2EN_PUNC:
                ch_char = CH2EN_PUNC[ch_char]

            if ch_char in self.map_dict:
                # add _ at the end of each ZH char as seperation
                out_line += self.map_dict[ch_char].strip() + chr(ord('_')+50000) ## for sp_concat
            else:
                if ch_char in control_char:
                    ch_char = chr(ord(ch_char)+50000)
                out_line += ch_char  ## sp_concat
        return out_line
    
    def tokenize(self, text):
        words = self.seg([text])[0]
        # print (words)
        tokens = []
        for word in words:
            if word in self.cws_vocab:
                tokens.append(word)
            else:
                for char in word:
                    if char in self.cws_vocab:
                        tokens.append(char)
                        continue
                    if char in control_char:
                        char = chr(ord('_')+50000) + chr(ord(char)+50000)
                        tokens.append(char)
                        continue
                    char = self.map_dict[char] if char in self.map_dict else char
                    tokens.extend([chr(ord('_')+50000) + x for x in self.spm_tokenizer.encode(char, out_type=str)])

        return tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.cws_vocab:
                ids.append(self.cws_vocab[token])
            elif token[0] == chr(ord('_')+50000) and token[1:] in self.vocab:
                ids.append(self.vocab[token[1:]]+len(self.cws_vocab))
            else:
                ids.append(self.vocab['[UNK]'])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
    
    ## TODO: implement the detokenizer!
    


ALL_TOKENIZERS = {
    "ConcatSep": ConcatSepTokenizer,
    "WubiZh": WubiZhTokenizer,
    "RawZh": RawZhTokenizer,
    'CommonZh': CommonZhTokenizer,
    "BertZh": BertZhTokenizer,
    "Bert": BertTokenizer,
    "BertHF": BertTokenizer,
    'CommonZhNoIndex': CommonZhNoIndexTokenizer,
    'Shuffled': ShuffledTokenizer,
    'PinyinConcatWubi': PinyinConcatWubiTokenizer, 
    'CWS': CWSNewTokenizer,
    'Byte': ByteTokenizer,
    'RandomIndex': RandomIndexTokenizer,
}