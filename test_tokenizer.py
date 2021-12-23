import random

import consts
from tokenization import ALL_TOKENIZERS
from processors.glue import PROCESSORS, convert_examples_to_features


def get_tokenizer(tokenizer_name, suffix):
	if suffix == 'no_index':
		vocab_file = consts.VOCAB_FILES_NO_INDEX
		model_file = vocab_file.replace('.vocab', '.model')
		tokenizer_type = consts.TOKENIZER_TYPES[tokenizer_name]


def main():
	random.seed(123)

	task = 'iflytek'
	tokenizer_name = 'pinyin'

	# Load examples

	tokenizers = load_tokenizers()
	tokenizer_names = list(consts.VOCAB_FILES.keys())
	print('\t'.join(tokenizer_names))
	tokenizer = tokenizers[tokenizer_name]
	text = '就读于清华大学 CS Department，的陈英发。'
	

	line = tokenizer.convert_line(text)
	
	print(line)
	tokens = tokenizer.spm_tokenizer.encode(line, out_type=str)
	print(tokens)
	print(get_inverse_index(tokens, tokenizer, text))


if __name__ == '__main__':
	main()