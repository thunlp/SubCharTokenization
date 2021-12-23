import pickle

VOCAB_FILES = {
	'cangjie': '/home/chenyingfa/WubiBERT/tokenizers/cangjie_zh_22675.vocab', 
	'pinyin': '/home/chenyingfa/WubiBERT/tokenizers/pinyin_zh_22675.vocab', 
	'stroke': '/home/chenyingfa/WubiBERT/tokenizers/stroke_zh_22675.vocab', 
	'wubi': '/home/chenyingfa/WubiBERT/tokenizers/wubi_zh_22675.vocab', 
	'zhengma': '/home/chenyingfa/WubiBERT/tokenizers/zhengma_zh_22675.vocab', 
	'zhuyin': '/home/chenyingfa/WubiBERT/tokenizers/zhuyin_zh_22675.vocab', 
	'raw': '/home/chenyingfa/WubiBERT/tokenizers/raw_zh_22675.vocab', 
	'bert': '/home/chenyingfa/WubiBERT/tokenizers/bert_chinese_uncased_22675.vocab', 
}

cangjie2ch = "/home/chenyingfa/WubiBERT/data/cangjie_to_chinese.pkl"
ch2cangjie = "/home/chenyingfa/WubiBERT/data/chinese_to_cangjie.pkl"

stroke2ch = "/home/chenyingfa/WubiBERT/data/stroke_to_chinese.pkl"
ch2stroke = "/home/chenyingfa/WubiBERT/data/chinese_to_stroke.pkl"

zhengma2ch = "/home/chenyingfa/WubiBERT/data/zhengma_to_chinese.pkl"
ch2zhengma = "/home/chenyingfa/WubiBERT/data/chinese_to_zhengma.pkl"

wubi2ch = "/home/chenyingfa/WubiBERT/data/wubi_to_chinese.pkl"
ch2wubi = "/home/chenyingfa/WubiBERT/data/chinese_to_wubi.pkl"

pinyin2ch = "/home/chenyingfa/WubiBERT/data/pinyin_to_chinese.pkl"
ch2pinyin = "/home/chenyingfa/WubiBERT/data/chinese_to_pinyin.pkl"

zhuyin2ch = "/home/chenyingfa/WubiBERT/data/zhuyin_to_chinese.pkl"
ch2zhuyin = "/home/chenyingfa/WubiBERT/data/chinese_to_zhuyin.pkl"

def load_vocab(file):
	vocab = {}
	with open(file, 'r') as f:
		for line in f:
			line = line.strip().split('\t')
			if len(line) != 2:
				continue
			k = line[0]
			v = line[1]
			vocab[k] = v
	return vocab


SEP_CHAR = '쎯'
SPECIAL_TOKENS = [
	'[UNK]',
	'[PAD]',
	'[CLS]',
	'[SEP]',
	'[MASK]',
]


def get_makeup(vocab, encodeds):
	'''Return num of punctuation, 单字，词，subchar'''
	def is_punc(c):
		return len(c) == 1 and c in '.,;:\'\"|?=+-_*&^%$#@!~`<>()\{\}\\/[]，。《》、：；‘“）——……￥，。！？【】（）％＃＠＆１２３４５６７８９０；：'

	def is_letter(c):
		return len(c) == 1 and ord('a') <= ord(c) <= ord('z')

	def is_chinese_char(cp):
		"""Checks whether CP is the codepoint of a CJK character."""
		# This defines a "chinese character" as anything in the CJK Unicode block:
		#   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
		#
		# Note that the CJK Unicode block is NOT all Japanese and Korean characters,
		# despite its name. The modern Korean Hangul alphabet is a different block,
		# as is Japanese Hiragana and Katakana. Those alphabets are used to write
		# space-separated words, so they are not treated specially and handled
		# like the all of the other languages.
		if len(cp) != 1:
			return False
		cp = ord(cp)
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

	n_punc = 0
	n_letter = 0
	n_chinese_char = 0
	n_subchar = 0
	n_phrase = 0
	n_other = 0

	for k, v in vocab.items():
		if k in SPECIAL_TOKENS:
			continue
		if is_punc(k):
			n_punc += 1
		elif is_letter(k):
			n_letter += 1
		elif is_chinese_char(k):  # This shouldn't be triggered
			print(k)
			# raise ValueError
			n_chinese_char += 1
		else:
			# Get number of corresponding character
			if SEP_CHAR not in k:
				n_subchar += 1
			else:
				cnt = k.count(SEP_CHAR)
				if cnt == 0:
					n_subchar += 1
				elif cnt == 1:
					n_chinese_char += 1
				else:
					n_phrase += 1
	return n_punc, n_letter, n_chinese_char, n_subchar, n_phrase
				
			



def main():
	for t in VOCAB_FILES:
		map_dict = pickle.load(open(ch2cangjie, 'rb'))
		encodeds = set(map_dict.values())
		# print(list(encodeds)[:10])
		# exit(0)
		vocab_file = VOCAB_FILES[t]
		vocab = load_vocab(vocab_file)
		n_punc, n_letter, n_chinese_char, n_subchar, n_phrase = get_makeup(vocab, encodeds)
		print(t)
		print(n_punc, n_letter, n_chinese_char, n_subchar, n_phrase)


if __name__ == '__main__':
	main()