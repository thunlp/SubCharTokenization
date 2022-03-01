import collections
import logging
import os
import unicodedata
import pickle

import sentencepiece as spm


logger = logging.getLogger(__name__)


wubi2ch = "/home/sichenglei/WubiBERT/data/wubi_to_chinese.pkl"
ch2wubi = "/home/sichenglei/WubiBERT/data/chinese_to_wubi.pkl"

pinyin2ch = "/home/sichenglei/WubiBERT/data/pinyin_to_chinese.pkl"
ch2pinyin = "/home/sichenglei/WubiBERT/data/chinese_to_pinyin.pkl"

zhuyin2ch = "/home/sichenglei/WubiBERT/data/zhuyin_to_chinese.pkl"
ch2zhuyin = "/home/sichenglei/WubiBERT/data/chinese_to_zhuyin.pkl"

control_char = u'0123456789abcdefghijklmnopqrstuvwxyz' 
control_uni = [chr(ord(c)+50000) for c in control_char]

CH2EN_PUNC = {f: t
              for f, t in zip(
                  u'，。！？【】（）％＃＠＆１２３４５６７８９０；：',
                  u',.!?[]()%#@&1234567890;:')}


def load_dict(dict_path):
    return pickle.load(open(dict_path, "rb"))

## choose accordingly
map_dict = load_dict(ch2pinyin)

## 
# for k,v in map_dict.items():
#     # convert digits to special symbols
#     map_dict[k] = ""
#     for c in v:
#         if c.isdigit():
#             map_dict[k] += chr(ord(c)+50000)
#         else:
#             map_dict[k] += c

print ("Dict loaded with size: {}".format(str(len(map_dict.keys()))))

puncs = '，。！？【】（）％＃＠＆１２３４５６７８９０；：,.!?[]()%#@&1234567890;:'

def convert(line):
    line = line.strip().lower()
    out_line = ""
    for char in line:
        if char in CH2EN_PUNC:
            char = CH2EN_PUNC[char]

        if char in map_dict:
            ## append transliterated char and separation symbol
            out_line += map_dict[char] + chr(ord('_')+50000)
        else:
            if char.isalpha():
                char = chr(ord(char)+50000)
            out_line += char
    return out_line

new_lines = 0
with open('/data1/private/clsi/wubi_corpus_orig/formatted/baidubaike_corpus.txt', 'r') as f:
    with open('/data1/private/clsi/wubi_corpus_orig/formatted_pinyin/baidubaike_corpus.txt', 'w+') as fw:
        line = f.readline()
        counter = 1
        while line:
            line = line.strip()
            if len(line) <= 3:
                line = f.readline()
                continue
            try:
                new_line = convert(line)
                fw.write(new_line + '\n')
                new_lines += 1
                
                if counter % 300000 == 0:
                    print (counter)
                    print (line)
                    print (new_line)
                    print ()
                
                line = f.readline()
                counter += 1
            except:
                line = f.readline()
                counter += 1

print ("total new lines: ", new_lines)


