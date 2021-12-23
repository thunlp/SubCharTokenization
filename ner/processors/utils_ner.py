import csv
import json
import torch
# from ..models.transformers import BertTokenizer

# class CNerTokenizer(BertTokenizer):
#     def __init__(self, vocab_file, do_lower_case=False):
#         super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
#         self.vocab_file = str(vocab_file)
#         self.do_lower_case = do_lower_case

#     def tokenize(self, text):
#         _tokens = []
#         for c in text:
#             if self.do_lower_case:
#                 c = c.lower()
#             if c in self.vocab:
#                 _tokens.append(c)
#             else:
#                 _tokens.append('[UNK]')
#         return _tokens

class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r", encoding="utf-8-sig") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines

	@classmethod
	def _read_text(self,input_file):
		lines = []
		with open(input_file,'r') as f:
			words = []
			labels = []
			for line in f:
				if line.startswith("-DOCSTART-") or line == "" or line == "\n":
					if words:
						lines.append({"words":words,"labels":labels})
						words = []
						labels = []
				else:
					splits = line.split(" ")
					words.append(splits[0])
					if len(splits) > 1:
						labels.append(splits[-1].replace("\n", ""))
					else:
						# Examples could have no label for mode = "test"
						labels.append("O")
			if words:
				lines.append({"words":words,"labels":labels})
		return lines

	@classmethod
	def _read_json(self,input_file):
		lines = []
		with open(input_file, 'r', encoding='utf8') as f:
			for line in f:
				line = json.loads(line.strip())
				text = line['text']
				label_entities = line.get('label',None)
				words = text
				labels = ['O'] * len(words)
				if label_entities is not None:
					for key,value in label_entities.items():
						for sub_name,sub_index in value.items():
							for start_index,end_index in sub_index:
								assert  ''.join(words[start_index:end_index+1]) == sub_name
								if start_index == end_index:
									labels[start_index] = 'S-'+key
								else:
									labels[start_index] = 'B-'+key
									labels[start_index+1:end_index+1] = ['I-'+key]*(len(sub_name)-1)
				lines.append({"words": words, "labels": labels})
		return lines

def get_entity_bios(seq,id2label):
	"""Gets entities from sequence.
	note: BIOS
	Args:
		seq (list): sequence of labels.
	Returns:
		list: list of (chunk_type, chunk_start, chunk_end).
	Example:
		# >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
		# >>> get_entity_bios(seq)
		[['PER', 0,1], ['LOC', 3, 3]]
	"""
	chunks = []
	chunk = [-1, -1, -1]
	for indx, tag in enumerate(seq):
		if not isinstance(tag, str):
			tag = id2label[tag]
		if tag.startswith("S-"):
			if chunk[2] != -1:
				chunks.append(chunk)
			chunk = [-1, -1, -1]
			chunk[1] = indx
			chunk[2] = indx
			chunk[0] = tag.split('-')[1]
			chunks.append(chunk)
			chunk = (-1, -1, -1)
		if tag.startswith("B-"):
			if chunk[2] != -1:
				chunks.append(chunk)
			chunk = [-1, -1, -1]
			chunk[1] = indx
			chunk[0] = tag.split('-')[1]
		elif tag.startswith('I-') and chunk[1] != -1:
			_type = tag.split('-')[1]
			if _type == chunk[0]:
				chunk[2] = indx
			if indx == len(seq) - 1:
				chunks.append(chunk)
		else:
			if chunk[2] != -1:
				chunks.append(chunk)
			chunk = [-1, -1, -1]
	return chunks

def get_entity_bio(seq,id2label):
	"""Gets entities from sequence.
	note: BIO
	Args:
		seq (list): sequence of labels.
	Returns:
		list: list of (chunk_type, chunk_start, chunk_end).
	Example:
		seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
		get_entity_bio(seq)
		#output
		[['PER', 0,1], ['LOC', 3, 3]]
	"""
	chunks = []
	chunk = [-1, -1, -1]
	for indx, tag in enumerate(seq):
		if not isinstance(tag, str):
			tag = id2label[tag]
		if tag.startswith("B-"):
			if chunk[2] != -1:
				chunks.append(chunk)
			chunk = [-1, -1, -1]
			chunk[1] = indx
			chunk[0] = tag.split('-')[1]
			chunk[2] = indx
			if indx == len(seq) - 1:
				chunks.append(chunk)
		elif tag.startswith('I-') and chunk[1] != -1:
			_type = tag.split('-')[1]
			if _type == chunk[0]:
				chunk[2] = indx

			if indx == len(seq) - 1:
				chunks.append(chunk)
		else:
			if chunk[2] != -1:
				chunks.append(chunk)
			chunk = [-1, -1, -1]
	return chunks

def get_entities(seq,id2label,markup='bios'):
	'''
	:param seq:
	:param id2label:
	:param markup:
	:return:
	'''
	assert markup in ['bio','bios']
	if markup =='bio':
		return get_entity_bio(seq,id2label)
	else:
		return get_entity_bios(seq,id2label)

def bert_extract_item(start_logits, end_logits):
	S = []
	start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
	end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
	for i, s_l in enumerate(start_pred):
		if s_l == 0:
			continue
		for j, e_l in enumerate(end_pred[i:]):
			if s_l == e_l:
				S.append((s_l, i, i + j))
				break
	return S


def get_inverse_index(tokens, tokenizer, text):
	'''
	Return inverse index of tokens in the original text.
	NOTE:
	1. This assumes that the tokenization result of multiple
	   characters is always equal to the concatenation of the
	   tokenization result of each characters separately.
	2. Last index is equal to the length of the text.

	Example:
	>>> Input:
	>>> text = '今天天气很好'
	>>> tokens = ['今天', '天气', '很', '好']
	>>> inverse_indices = [0, 2, 4, 5, 6]
	'''
	split = [None] * len(text)  # Tokenization result of each character
	for i, ch in enumerate(text):
		split[i] = ''.join(tokenizer.tokenize(ch))
	indices = [None] * len(tokens)
	idx_t = 0
	idx_s = 0
	len_t = 0
	len_s = 0
	while idx_t < len(tokens):
		if len_t == len_s:
			while len(split[idx_s]) == 0: # Skip all whitespaces
				idx_s += 1
			indices[idx_t] = idx_s
			len_s += len(split[idx_s])
			len_t += len(tokens[idx_t])
			idx_s += 1
			idx_t += 1
		elif len_t < len_s:
			indices[idx_t] = idx_s - 1
			len_t += len(tokens[idx_t])
			idx_t += 1
		else:
			len_s += len(split[idx_s])
			idx_s += 1
	indices.append(len(text))
	return indices


def get_token_pos(tokens, subchars):
	'''
	Return starting index of each token in subchars
	NOTE:
	1. This assumes that the concatenation of tokens is equal to the 
	   concatenation of subchars, and that for each token and subchar, the 
	   token either doesn't contain the subchar, or contains it entirely.

	Example:
	>>> Input:
	>>> subchars  = ['jin+', 'ti', 'an+', 'ti', 'an+', 'qi+', 'hen+', 'hao+']
	>>> tokens    = ['jin+', 'tian+', 'tian+qi+', 'hen+hao+']
	>>> token_pos = [0, 1, 3, 6]
	'''
	assert ''.join(subchars) == ''.join(tokens), 'The concatenation of subchars and tokens must be equal.'
	pos = [0] * len(tokens)
	i = 0
	idx_s = 0
	while i < len(tokens):
		pos[i] = idx_s
		sum_len = 0
		j = idx_s
		while sum_len < len(tokens[i]):
			sum_len += len(subchars[j])
			j += 1
		i += 1
	# pos.append(len(text))
	return pos


def get_subchar_pos(tokens, subchars):
	'''
	Return starting index of each subchar in tokens.
	NOTE: This assumes that the concatenation of tokens is equal to the 
	concatenation of subchars.

	Example:
	>>> Input:
	>>> subchars  = ['jin+', 'ti', 'an+', 'ti', 'an+', 'qi+', 'hen+', 'hao+']
	>>> tokens    = ['jin', '+', 'tian+', 'tian+qi', '+', 'hen+hao+']
	>>> token_pos = [0, 2, 2, 3, 3, 3, 5, 5]
	'''
	pos = [None] * len(subchars)
	len_t = 0
	len_s = 0
	j = -1  # idx of last token that was added to len_t
	for i, subchar in enumerate(subchars):
		while len_t <= len_s:
			j += 1
			len_t += len(tokens[j])
		pos[i] = j
		len_s += len(subchar)
	return pos
	


def get_subchar_to_token(tokens, subchars):
	'''
	Return index mapping from subchar to token, the i'th element is the index 
	of the token that contains the i'th subchar.
	NOTE:
	1. This assumes that the concatenation of tokens is equal to the 
	   concatenation of subchars, and that for each token and subchar, the 
	   token either doesn't contain the subchar, or contains it entirely.

	Example:
	>>> Input:
	>>> subchars  = ['jin+', 'ti', 'an+', 'ti', 'an+', 'qi+', 'hen+', 'hao+']
	>>> tokens    = ['jin+', 'tian+', 'tian+qi+', 'hen+hao+']
	>>> subchar_to_token = [0, 1, 1, 2, 2, 2, 3, 3]
	'''
	subchar_to_token = [None] * len(subchars)
	j = 0
	for i, token in enumerate(tokens):
		total_len = 0
		while total_len < len(token):
			try:
				subchar_to_token[j] = i
			except:
				print(subchar_to_token)
				print(tokens)
				print(subchars)
				raise ValueError
			total_len += len(subchars[j])
			j += 1
	return subchar_to_token


def get_labels_of_tokens(labels, tokens, inv_idx):
	'''
	Given labels to each character in a text, its tokenization result,
	and the inverse index of the tokens, return the labels of each token.
	
	Labeling scheme:
		text:   这是北京的清华大学
		labels: O O B I O B I I I

		tokens: 		这是  北京的  清华  大学
		token labels: 	O     S       B     I
	'''
	token_labels = ['O'] * len(tokens)  # 'O' by default
	
	def get_inclusive_left(lo, inv_idx):
		for i in range(len(inv_idx) - 1, 0, -1):
			if inv_idx[i] <= lo and inv_idx[i-1] < inv_idx[i]:
				return i
		return 0

	def get_inclusive_right(hi, inv_idx):
		for i in range(len(inv_idx)):
			if inv_idx[i] >= hi:
				return i
		raise ValueError

	i = 0
	while i < len(labels):
		if labels[i] == 'O':
			i += 1
		else:
			if labels[i][0] == 'S':
				lo, hi = i, i + 1
				left = get_inclusive_left(lo, inv_idx)
				right = get_inclusive_right(hi, inv_idx)
			else:
				if labels[i][0] != 'B':
					print(labels)
					print(i, labels[i][0])
					raise ValueError()
				hi = i + 1
				while hi < len(labels):
					if labels[hi][0] != 'I':
						break
					hi += 1
				lo = i
				left = get_inclusive_left(lo, inv_idx)
				right = get_inclusive_right(hi, inv_idx)
			assert right > left
			length = right - left
			if length == 1:
				token_labels[left] = 'S' + labels[i][1:]
			else:
				token_labels[left] = 'B' + labels[i][1:]
				for j in range(left+1, right):
					token_labels[j] = 'I' + labels[i][1:]
			i = hi
	return token_labels


def get_char_labels(token_label_ids, left_index, right_index, id2label):
	'''
	Return list character label ids given a token label ids, and 
	inverse index of it.
	'''
	max_len = len(token_label_ids)
	# Truncate lists
	hi = len(left_index)
	while left_index[hi-1] == -1:
		hi -= 1
	left_index = left_index[:hi]
	hi = len(right_index)
	while right_index[hi-1] == -1:
		hi -= 1
	right_index = right_index[:hi]
	inv_idx = left_index[:] + right_index[-1:]
	hi = len(token_label_ids)
	while token_label_ids[hi-1] == 0:
		hi -= 1
	token_label_ids = token_label_ids[1:hi-1]


	if len(inv_idx) != len(token_label_ids) + 1:
		print(inv_idx, len(inv_idx))
		print(token_label_ids, len(token_label_ids))
		print(left_index)
		print(right_index)
		print(id2label)
		raise ValueError('Token labels and inverse index do not match.')

	label2id = {v: k for k, v in id2label.items()}
	id_o = label2id['O']
	char_label_ids = []
	n_tokens = len(inv_idx) - 1
	i = 0
	while i < n_tokens:
		tli = token_label_ids[i]
		tag = id2label[tli]

		if tli == id_o:
			length = inv_idx[i+1] - inv_idx[i]
			for _ in range(length):
				char_label_ids.append(id_o)
			i += 1
			continue

		# Get the label ids and count to add
		if tag.startswith('S-') or tag.startswith('B-'):
			if tag.startswith('S-'):
				length = inv_idx[i+1] - inv_idx[i]
				id_s = tli
				i += 1
			elif tag.startswith('B-'):
				hi = i + 1
				# assert hi < n_tokens
				id_i = label2id['I' + tag[1:]]
				while hi < n_tokens and token_label_ids[hi] == id_i:
					hi += 1
				
				length = inv_idx[hi] - inv_idx[i]
				i = hi

			id_b = label2id['B' + tag[1:]]
			id_i = label2id['I' + tag[1:]]
			id_s = label2id['S' + tag[1:]]
			# Append char label ids
			if length > 0:
				if length == 1:
					char_label_ids.append(id_s)
				else:
					char_label_ids.append(id_b)
					for _ in range(length - 1):
						char_label_ids.append(id_i)
		elif tag.startswith('I-'):
			length = inv_idx[i+1] - inv_idx[i]
			for _ in range(length):
				char_label_ids.append(tli)
			i += 1
		else:
			print(token_label_ids)
			print(inv_idx)
			print(id2label)
			print(i)
			print(tag)
			print(tli)
			raise ValueError('Got label O, when it should have skipped them.')
	char_label_ids = [31] + char_label_ids + [31]
	pad_len = max_len - len(char_label_ids)
	if pad_len > 0:
		char_label_ids += [0] * pad_len
	return char_label_ids
