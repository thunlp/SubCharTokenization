from io import open
import pickle
import string
import random 
random.seed(2021)

wubi2ch = "data/wubi_to_chinese.pkl"
ch2wubi = "data/chinese_to_wubi.pkl"

pinyin2ch = "data/pinyin_to_chinese.pkl"
ch2pinyin = "data/chinese_to_pinyin.pkl"


def load_dict(dict_path):
	return pickle.load(open(dict_path, "rb"))

ch_chars = list(load_dict(ch2wubi).keys()) + list(load_dict(ch2pinyin).keys()) + list(string.punctuation)
ch_chars = list(set(ch_chars))
random.shuffle(ch_chars)
# print (len(ch_chars))
SEP = chr(ord('_')+50000)

# random_index_map = {}
# for i in range(len(ch_chars)):
#     random_index_map[ch_chars[i]] = i + 10000

# print (list(random_index_map.keys())[:10])
# print (list(random_index_map.values())[:10])

# with open("random_index_map.pkl", 'wb') as f:
#     pickle.dump(random_index_map, f)

with open("random_index_map.pkl", 'rb') as f:
    random_index_map = pickle.load(f)

# print (list(random_index_map.keys())[:10])
# print (list(random_index_map.values())[:10])

with open('/data2/private/clsi/wubi_corpus_orig/formatted/baidubaike_corpus.txt', 'r') as f:
    with open('/data2/private/clsi/wubi_corpus_random_index/formatted/baidubaike_corpus.txt', 'w+') as fw:
        line = f.readline()
        idx = 0
        while line:
            idx += 1       
            newline = ''
            for c in line.strip():
                if c in random_index_map:
                    newline += str(random_index_map[c])
                else:
                    newline += c
                newline += SEP
            newline += '\n'
            fw.write(newline)
            line = f.readline()

            if idx % 400000 == 0:
                print (idx)     
                print (newline)    


## tmux 10