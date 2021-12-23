import pickle 
import os 
import collections

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

def load_vocab_txt(vocab_file):
    d = {}
    with open(vocab_file, 'r') as f:
        vocab = f.readlines()
    for line in vocab:
        line = line.strip().split()
        if len(line) != 2:
            print (line)
        d[line[0]] = int(line[1])
    return d

# vocab = load_vocab_spm("../tokenizers/wiki_64k.vocab")

# all_stats = {}
# for i in range(256):
#     with open("stats_macro_64k/wikicorpus_en_training_"+str(i)+".pkl", "rb") as f:
#         d = pickle.load(f)
#     for k,v in d.items():
#         if k not in vocab:
#             k = "[UNK]"
#         if k not in all_stats:
#             all_stats[k] = v 
#         else:
#             all_stats[k] += v

# all_stats = {k: v for k, v in sorted(all_stats.items(), key=lambda item: -item[1])}
# # for k,v in all_stats.items():
# #     print (k,v)

# with open("stats_macro_64k.txt", "w+") as f:
#     for k,v in all_stats.items():
#         f.write(k + " " + str(v) + '\n')


vocab = load_vocab_txt("stats_macro_64k.txt")
# print (len(vocab.keys()))
# counter = 0
# for k,v in vocab.items():
#     counter += v
# print (counter)
print (vocab["[UNK]"])