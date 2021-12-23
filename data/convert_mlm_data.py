import csv
import sys 

csv.field_size_limit(sys.maxsize)

def read_tsv(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return list(csv.reader(f, delimiter="\t"))

with open("/home/sichenglei/LM-BFF/MLM_data/IMDB/train.txt", "w") as f:
    train_text = read_tsv('/home/sichenglei/LM-BFF/data/original/IMDB/train.tsv')
    for line in train_text[1:]:
        f.write(line[0] + '\n')


# print (len(train_text))
# for line in train_text:
#     print (line)
#     break 
