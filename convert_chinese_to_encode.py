## this script is used to convert cws_raw_zh to cws_wubi_zh and cws_zhuyin_zh
import pickle

with open('/mnt/datadisk0/scl/baike/formatted_cws/baidubaike_corpus.txt', 'r') as f:
    orig = f.readlines()

print ('orig', len(orig))

ch2zhuyin = "/home/ubuntu/WubiBERT/data/chinese_to_zhuyin.pkl"
ch2wubi = "/home/ubuntu/WubiBERT/data/chinese_to_wubi.pkl"
control_char = u'0123456789abcdefghijklmnopqrstuvwxyz' 


CH2EN_PUNC = {f: t
              for f, t in zip(
                  u'，。！？【】（）％＃＠＆１２３４５６７８９０；：',
                  u',.!?[]()%#@&1234567890;:')}


def load_dict(dict_path):
	return pickle.load(open(dict_path, "rb"))

wubi_dict = load_dict(ch2wubi)
zhuyin_dict = load_dict(ch2zhuyin)

with open('/mnt/datadisk0/scl/baike/formatted_cws_wubi/baidubaike_corpus.txt', 'w+') as f:
    map_dict = wubi_dict
    i = 0
    for line in orig:
        ## removing single char lines which are irrelevant
        if len(line) > 2:
            out_line = ""

            line = line.lower()
            for char in line:
                if char in CH2EN_PUNC:
                    char = CH2EN_PUNC[char]

                if char in map_dict:
                    out_line += map_dict[char].strip() + chr(ord('_')+50000)
                else:
                    if char in control_char:
                        char = chr(ord(char)+50000)
                    out_line += char  
            
            f.write(out_line)
            i += 1
        
            if (i + 1) % 200000 == 0:
                print (i, out_line)


with open('/mnt/datadisk0/scl/baike/formatted_cws_zhuyin/baidubaike_corpus.txt', 'w+') as f:
    map_dict = zhuyin_dict
    i = 0
    for line in orig:
        ## removing single char lines which are irrelevant
        if len(line) > 2:
            out_line = ""

            line = line.lower()
            for char in line:
                if char in CH2EN_PUNC:
                    char = CH2EN_PUNC[char]

                if char in map_dict:
                    out_line += map_dict[char].strip() + chr(ord('_')+50000)
                else:
                    if char in control_char:
                        char = chr(ord(char)+50000)
                    out_line += char  
            
            f.write(out_line)
            i += 1
        
            if (i + 1) % 200000 == 0:
                print (i, out_line)