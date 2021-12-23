import collections
import logging
import os
import unicodedata
import six
from io import open
import pickle
# from tokenization import CommonZhTokenizer, BertZhTokenizer, RawZhTokenizer
from tqdm import tqdm 

import sentencepiece as spm

cangjie2ch = "../data/cangjie_to_chinese.pkl"
ch2cangjie = "../data/chinese_to_cangjie.pkl"

stroke2ch = "../data/stroke_to_chinese.pkl"
ch2stroke = "../data/chinese_to_stroke.pkl"

zhengma2ch = "../data/zhengma_to_chinese.pkl"
ch2zhengma = "../data/chinese_to_zhengma.pkl"

wubi2ch = "../data/wubi_to_chinese.pkl"
ch2wubi = "../data/chinese_to_wubi.pkl"

pinyin2ch = "../data/pinyin_to_chinese.pkl"
ch2pinyin = "../data/chinese_to_pinyin.pkl"

zhuyin2ch = "../data/zhuyin_to_chinese.pkl"
ch2zhuyin = "../data/chinese_to_zhuyin.pkl"

# control_char = u'0123456789abcdefghijklmnopqrstuvwxyz' 
control_char = u'0123456789abcdefghijklmnopqrstuvwxyz' 
pinyin_tones = '❁❄❂❃❅'
control_uni = [chr(ord(c)+50000) for c in control_char]
sep = chr(ord('_')+50000)

with open("/home/sichenglei/WubiBERT/random_index_map.pkl", 'rb') as f:
    random_index_map = pickle.load(f)

with open("/home/sichenglei/WubiBERT/byte_char_map.pkl", "rb") as f:
    ch_chars = pickle.load(f)
ch_chars = ch_chars.values()

def load_dict(dict_path):
	return pickle.load(open(dict_path, "rb"))

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
# vocab = load_vocab_spm('shuffled_wubi_22675.vocab')
# vocab = load_vocab_spm('shuffled_pinyin_22675.vocab')
# with open('../res/pinyin_cws_22675_0.8.cws_vocab', 'r') as f:
#     cws_vocab = f.readlines()
# cws_vocab = [v.strip() for v in cws_vocab]
# cws_vocab = ('../res/pinyin_cws_22675_0.7.cws_vocab')

# vocab = load_vocab_spm('../res/pinyin_cws_22675_0.8.vocab')
# vocab = load_vocab_spm('random_index_22675.vocab')
vocab = load_vocab_spm('wubi_bpe_5k.vocab')


v = list(vocab.keys())[5:]

d_wubi = load_dict(ch2wubi)
d_pinyin = load_dict(ch2pinyin)
d_zhuyin = load_dict(ch2zhuyin)

counter_char = 0
counter_sub = 0
counter_compo = 0
counter_total = 0

# for cand in cws_vocab:
#     if len(cand) < 1:
#         continue
#     counter_total += 1
#     if len(cand) == 1:
#         counter_char += 1
#     else:
#         counter_compo += 1

# print (counter_char, counter_compo, counter_total)

# pinyin_d = load_dict(pinyin2ch)
# for k in pinyin_d.keys():
#     i = -1
#     while k[i].isdigit():
#         i -= 1
#     c = k[i]
#     if c not in pinyin_tones:
#         print (c)

# d_no_index = []
# for s in d_wubi.keys():
#     s = ''.join([i for i in s if not i.isdigit()])
#     d_no_index.append(s)
# d_wubi = d_no_index

# d_no_index = []
# for s in d_wubi.keys():
#     s = ''.join([i for i in s if not i.isdigit()])
#     d_no_index.append(s)
# d_wubi = d_no_index


# d_no_index = {}
# for ch in d_wubi.keys():
#     d_no_index[ch] = ''.join([i for i in d_wubi[ch] if not i.isdigit()]) 

# for ch in d_wubi.keys():
#     if ch in d_no_index:
#         d_no_index[ch] += ''.join([i for i in d_wubi[ch] if not i.isdigit()]) 
#     else:
#         d_no_index[ch] = ''.join([i for i in d_wubi[ch] if not i.isdigit()]) 

# d = d_no_index.values()
# d = list(set(d))

# # print (d)
d = list(d_wubi.values())

# # counter_char = 0
# # counter_sub = 0
# # counter_compo = 0

def check_char(word):
    for c in word:
        if c.lower() not in (control_char + pinyin_tones):
            return False 
    return True

for c in v:
    if len(c) < 1:
        continue
    counter_total += 1

    ## punctuations:
    if len(c) == 1 and (c.lower() not in control_char):
        counter_char += 1
        print (c)
    
    ## char
    elif (c[-1] == sep and c[:-1] in d) or (c in d):
        counter_char += 1
        print (c)

    ## sub-char 
    elif (c[-1] == sep) and (sep not in c[:-1]) and (c[:-1] not in d) and check_char(c[:-1]):
        counter_sub += 1
        # print (c)
    elif check_char(c):
        counter_sub += 1
        # print (c)

    else:
        counter_compo += 1
        # print (c)

# d = random_index_map.values()
# d = [str(s) for s in d]

# for c in v:
#     if len(c) < 1:
#         continue
#     counter_total += 1

#     # ## punctuations:
#     # if len(c) == 1 and (c.lower() not in control_char):
#     #     counter_char += 1
#     #     # print (c)
    
#     ## char
#     if (c[-1] == sep and len(c) == 4) or (c[-1] != sep and len(c) == 3):
#         counter_char += 1
#         # print (c)
#     elif len(c) == 1 and (not c in ch_chars):
#         counter_char += 1
#         # print (c)
#     ## sub-char 
#     elif ((c[-1] == sep and len(c) < 4) or (c[-1] != sep and len(c) < 3)) and (c.count(sep) <= 1):
#         counter_sub += 1
#         # print (c)
#     else:
#         counter_compo += 1
#         print (c)
    

# for c in v:
#     if len(c) < 1:
#         continue 
    
#     counter_total += 1
#     if len(c) == 1:
#         counter_char += 1
#     elif len(c) > 1:
#         counter_compo += 1


print (counter_total)
print (counter_sub)
print ("sub: ", counter_sub / counter_total)
print (counter_char)
print ("char: ", counter_char / counter_total)
print (counter_compo)
print ("compo: ", counter_compo / counter_total)