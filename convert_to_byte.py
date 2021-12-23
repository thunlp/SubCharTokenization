from io import open
import pickle

wubi2ch = "data/wubi_to_chinese.pkl"
ch2wubi = "data/chinese_to_wubi.pkl"

def load_dict(dict_path):
	return pickle.load(open(dict_path, "rb"))

with open("byte_char_map.pkl", "rb") as f:
    byte_char_map = pickle.load(f)
SEP = chr(ord('_')+50000)

with open('/data2/private/clsi/wubi_corpus_orig/formatted/baidubaike_corpus.txt', 'r') as f:
    with open('/data2/private/clsi/wubi_corpus_byte/formatted/baidubaike_corpus.txt', 'w+') as fw:
        line = f.readline()
        idx = 0
        while line:
            idx += 1             
            newline = ''
            for c in line.strip():
                c = bytes(c, 'utf-8')
                for byte_index in c:
                    ch = byte_char_map[byte_index]
                    newline += ch
                newline += SEP
            newline += '\n'
            fw.write(newline)
            line = f.readline()

            if idx % 400000 == 0:
                print (idx) 
                print (newline)  



# with open('/data2/private/clsi/wubi_corpus_byte/formatted/baidubaike_corpus.txt', 'r') as f:
#     print (f.readline())

## tmux 12