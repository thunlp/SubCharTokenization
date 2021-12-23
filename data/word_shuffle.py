from tqdm import tqdm 
import pickle
import collections
import random 
random.seed(2021)

# d = [] ## store all unique characters 
# with open("/data2/private/clsi/wubi_corpus_orig/formatted/baidubaike_corpus.txt", 'r') as f:
#     line = f.readline()
#     counter = 0
#     while line:
#         counter += 1
#         if counter % 10000 == 0:
#             print (counter)
#         line = line.strip()
#         for c in line:
#             if c not in d:
#                 d.append(c)
#         line = f.readline()

# print ("#unique chars: ", len(d))

# d_copy = d[:]
# random.shuffle(d_copy)

# final = {}
# for i in range(len(d)):
#     final[d[i]] = d_copy[i]

# with open("word_shuffle_dict.pkl", 'wb') as f:
#     pickle.dump(final, f)

# wubi2ch = "wubi_to_chinese.pkl"
# ch2wubi = "chinese_to_wubi.pkl"

'''
wubi2ch = "pinyin_to_chinese.pkl"
ch2wubi = "chinese_to_pinyin.pkl"

def load_dict(dict_path):
	return pickle.load(open(dict_path, "rb"))

wubi_dict = load_dict(ch2wubi)

same_len_dict = {}
for k,v in wubi_dict.items():
    lv = len(v)
    if lv not in same_len_dict:
        same_len_dict[lv] = [k]
    else:
        same_len_dict[lv].append(k)

mapping_dict = {}
for lv, lst in same_len_dict.items():
    lst_2 = lst[:]
    random.shuffle(lst_2)
    for k in range(len(lst)):
        mapping_dict[lst[k]] = lst_2[k]


with open("pinyin_shuffle_dict.pkl", 'wb') as f:
    pickle.dump(mapping_dict, f)

# print (mapping_dict['我'])
# print (mapping_dict['好'])
# print (mapping_dict['疯'])

# for k,v in same_len_dict.items():
#     print (k,v)


'''

# def load_vocab_spm(vocab_file):
#     """Loads a vocabulary file into a dictionary."""
#     vocab = collections.OrderedDict()
#     index = 0
#     with open(vocab_file, "r", encoding="utf-8") as reader:
#         while True:
#             token = reader.readline()
#             if not token:
#                 break
#             token = token.strip().split()[0].strip()
#             vocab[token] = index
#             index += 1
#     return vocab


# raw_zh_vocab = load_vocab_spm()

# print (len(raw_zh_vocab))


# counter = 0
# for v in list(raw_zh_vocab.keys())[:7]:
#     print (v)

# raw_zh_vocab = load_vocab_spm("../tokenizers/raw_zh_22675.vocab")

# d = list(raw_zh_vocab.keys())[5 : ]
# d_copy = d[:]
# random.shuffle(d_copy)

# final = {}
# for i in range(len(d)):
#     final[d[i].strip()] = d_copy[i].strip()

# with open("word_shuffle_dict.pkl", 'wb') as f:
#     pickle.dump(final, f)

control_char = u'0123456789abcdefghijklmnopqrstuvwxyz' 
control_uni = [chr(ord(c)+50000) for c in control_char]

CH2EN_PUNC = {f: t
              for f, t in zip(
                  u'，。！？【】（）％＃＠＆１２３４５６７８９０；：',
                  u',.!?[]()%#@&1234567890;:')}


ch2pinyin = "chinese_to_pinyin.pkl"
ch2wubi = "chinese_to_wubi.pkl"

def load_dict(dict_path):
	return pickle.load(open(dict_path, "rb"))

def convert_line(line, shuffle_map, map_dict):
        # text = text.lower() #  always lowercasing
        text = ""
        for c in line.strip():
            if c in shuffle_map:
                newc = shuffle_map[c]
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

            if ch_char in map_dict:
                # add _ at the end of each ZH char as seperation
                out_line += map_dict[ch_char].strip() + chr(ord('_')+50000) ## for sp_concat
            else:
                if ch_char in control_char:
                    ch_char = chr(ord(ch_char)+50000)
                out_line += ch_char  
        # print (out_line)
        return out_line

with open("wubi_shuffle_dict.pkl", 'rb') as f:
    mapping = pickle.load(f)

wubi_map_dict = load_dict(ch2wubi)

with open("/data2/private/clsi/wubi_corpus_orig/formatted/baidubaike_corpus.txt", 'r') as fr:
    with open("/data2/private/clsi/wubi_corpus_shuffled/formatted/baidubaike_corpus.txt", 'w') as fw:
        line = fr.readline()
        counter = 0
        while line:
            counter += 1
            newline = convert_line(line, mapping, wubi_map_dict) + '\n'
            fw.write(newline)
            if counter % 40000 == 0:
                print (counter)
                print (newline)
            line = fr.readline()


# for k,v in mapping.items():
#     print (k + ': ' + v)

# with open("/data2/private/clsi/wubi_corpus_orig/formatted/baidubaike_corpus.txt", 'r') as fr:
#     with open("/data2/private/clsi/wubi_corpus_shuffled/formatted/baidubaike_corpus.txt", 'w') as fw:
#         line = fr.readline()
#         counter = 0
#         while line:
#             counter += 1
#             line = line.strip()
#             newline = ''
#             for c in line:
#                 if c in mapping:
#                     newc = mapping[c]
#                 else:
#                     newc = c 
#                 newline += newc
#             newline += '\n'
#             fw.write(newline)
#             if counter % 40000 == 0:
#                 print (counter)
#                 print (newline)
#             line = fr.readline()




# with open("pinyin_shuffle_dict.pkl", 'rb') as f:
#     mapping = pickle.load(f)

# # for k,v in mapping.items():
# #     print (k + ': ' + v)

# with open("/data2/private/clsi/wubi_corpus_orig/formatted/baidubaike_corpus.txt", 'r') as fr:
#     with open("/data2/private/clsi/pinyin_corpus_shuffled/formatted/baidubaike_corpus.txt", 'w') as fw:
#         line = fr.readline()
#         counter = 0
#         while line:
#             counter += 1
#             line = line.strip()
#             newline = ''
#             for c in line:
#                 if c in mapping:
#                     newc = mapping[c]
#                 else:
#                     newc = c 
#                 newline += newc
#             newline += '\n'
#             fw.write(newline)
#             if counter % 40000 == 0:
#                 print (counter)
#                 print (newline)
#             line = fr.readline()