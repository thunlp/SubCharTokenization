from tqdm import tqdm

with open('data/baike/formatted/baidubaike_corpus.txt', 'r') as f:
    data = f.readlines()

vocab = {}
for line in tqdm(data):
    line = line.lower().replace(' ', '')
    line = list(line.strip())
    for ch in line:
        if ch not in vocab:
            vocab[ch] = 1
        else:
            vocab[ch] += 1

vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1], reverse=True)}

counter = 5
with open('tokenizers/bert_chinese_uncased_30k.vocab', 'w+') as f:
    f.write('[UNK]	0\n')
    f.write('[PAD]	0\n')
    f.write('[CLS]	0\n')
    f.write('[SEP]	0\n')
    f.write('[MASK]	0\n')
    for k,v in vocab.items():
        if counter >= 30000:
            break 
        f.write(k+' '+str(v)+'\n')
        counter += 1


