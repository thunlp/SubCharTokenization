import collections
import json
import os
from copy import deepcopy

from tqdm import tqdm

from ..tools import official_tokenization as tokenization

SPIECE_UNDERLINE = '▁'
DEBUG = True


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
						 orig_answer_text):
	"""Returns tokenized answer spans that better match the annotated answer."""

	# The SQuAD annotations are character based. We first project them to
	# whitespace-tokenized words. But then after WordPiece tokenization, we can
	# often find a "better match". For example:
	#
	#   Question: What year was John Smith born?
	#   Context: The leader was John Smith (1895-1943).
	#   Answer: 1895
	#
	# The original whitespace-tokenized answer will be "(1895-1943).". However
	# after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
	# the exact answer, 1895.
	#
	# However, this is not always possible. Consider the following:
	#
	#   Question: What country is the top exporter of electornics?
	#   Context: The Japanese electronics industry is the lagest in the world.
	#   Answer: Japan
	#
	# In this case, the annotator chose "Japan" as a character sub-span of
	# the word "Japanese". Since our WordPiece tokenizer does not split
	# "Japanese", we just use "Japanese" as the annotation. This is fairly rare
	# in SQuAD, but does happen.
	tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

	for new_start in range(input_start, input_end + 1):
		for new_end in range(input_end, new_start - 1, -1):
			text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
			if text_span == tok_answer_text:
				return (new_start, new_end)

	return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
	"""Check if this is the 'max context' doc span for the token."""

	# Because of the sliding window approach taken to scoring documents, a single
	# token can appear in multiple documents. E.g.
	#  Doc: the man went to the store and bought a gallon of milk
	#  Span A: the man went to the
	#  Span B: to the store and bought
	#  Span C: and bought a gallon of
	#  ...
	#
	# Now the word 'bought' will have two scores from spans B and C. We only
	# want to consider the score with "maximum context", which we define as
	# the *minimum* of its left and right context (the *sum* of left and
	# right context will always be the same, of course).
	#
	# In the example the maximum context for 'bought' would be span C since
	# it has 1 left context and 3 right context, while span B has 4 left context
	# and 0 right context.
	best_score = None
	best_span_index = None
	for (span_index, doc_span) in enumerate(doc_spans):
		end = doc_span.start + doc_span.length - 1
		if position < doc_span.start:
			continue
		if position > end:
			continue
		num_left_context = position - doc_span.start
		num_right_context = end - position
		score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
		if best_score is None or score > best_score:
			best_score = score
			best_span_index = span_index

	return cur_span_index == best_span_index


def json2features(input_file, output_files, tokenizer, is_training=False, repeat_limit=3, max_query_length=64,
				  max_seq_length=512, doc_stride=128):
	with open(input_file, 'r') as f:
		train_data = json.load(f)
		train_data = train_data['data']

	def _is_chinese_char(cp):
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

	def is_fuhao(c):
		if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
				or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
				or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
				or c == '‘' or c == '’':
			return True
		return False

	def _tokenize_chinese_chars(text):
		"""Adds whitespace around any CJK character."""
		output = []
		for char in text:
			cp = ord(char)
			if _is_chinese_char(cp) or is_fuhao(char):
				if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
					output.append(SPIECE_UNDERLINE)
				output.append(char)
				output.append(SPIECE_UNDERLINE)
			else:
				output.append(char)
		return "".join(output)

	def is_whitespace(c):
		if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
			return True
		return False

	# to examples
	examples = []
	mis_match = 0
	for article in tqdm(train_data):
		for para in article['paragraphs']:
			context = para['text']
			context_chs = _tokenize_chinese_chars(context)
			doc_tokens = []
			char_to_word_offset = []
			prev_is_whitespace = True

			for c in context_chs:
				if is_whitespace(c):
					prev_is_whitespace = True
				else:
					if prev_is_whitespace:
						doc_tokens.append(c)
					else:
						doc_tokens[-1] += c
					prev_is_whitespace = False
				if c != SPIECE_UNDERLINE:
					char_to_word_offset.append(len(doc_tokens) - 1)

			for qas in para['qas']:
				qid = qas['id']
				ques_text = qas['question']
				ans_text = qas['answers'][0]['text']

				start_position_final = None
				end_position_final = None
				if is_training:
					count_i = 0
					start_position = qas['answers'][0]['answer_start']

					end_position = start_position + len(ans_text) - 1
					while context[start_position:end_position + 1] != ans_text and count_i < repeat_limit:
						start_position -= 1
						end_position -= 1
						count_i += 1

					while context[start_position] == " " or context[start_position] == "\t" or \
							context[start_position] == "\r" or context[start_position] == "\n":
						start_position += 1

					start_position_final = char_to_word_offset[start_position]
					end_position_final = char_to_word_offset[end_position]

					if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
						start_position_final += 1

					actual_text = "".join(doc_tokens[start_position_final:(end_position_final + 1)])
					cleaned_answer_text = "".join(tokenization.whitespace_tokenize(ans_text))

					if actual_text != cleaned_answer_text:
						print(actual_text, 'V.S', cleaned_answer_text)
						mis_match += 1

				examples.append({'doc_tokens': doc_tokens,
								 'orig_answer_text': ans_text,
								 'qid': qid,
								 'question': ques_text,
								 'answer': ans_text,
								 'start_position': start_position_final,
								 'end_position': end_position_final})

	print('examples num:', len(examples))
	print('mis_match:', mis_match)
	os.makedirs('/'.join(output_files[0].split('/')[0:-1]), exist_ok=True)
	json.dump(examples, open(output_files[0], 'w'), ensure_ascii=False)
	print('saving to', output_files[0])

	# to features
	features = []
	unique_id = 1000000000
	for (example_index, example) in enumerate(tqdm(examples)):
		query_tokens = tokenizer.tokenize(example['question'])
		if len(query_tokens) > max_query_length:
			query_tokens = query_tokens[0:max_query_length]

		tok_to_orig_index = []
		orig_to_tok_index = []
		all_doc_tokens = []
		for (i, token) in enumerate(example['doc_tokens']):
			orig_to_tok_index.append(len(all_doc_tokens))
			# print(token)
			sub_tokens = tokenizer.tokenize(token)
			# print(sub_tokens)
			for sub_token in sub_tokens:
				tok_to_orig_index.append(i)
				all_doc_tokens.append(sub_token)

		tok_start_position = None
		tok_end_position = None
		if is_training:
			tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
			if example['end_position'] < len(example['doc_tokens']) - 1:
				tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
			else:
				tok_end_position = len(all_doc_tokens) - 1
			(tok_start_position, tok_end_position) = _improve_answer_span(
				all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
				example['orig_answer_text'])

		print('orig_to_tok_index')
		print(orig_to_tok_index)
		print('all_doc_tokens')
		print(all_doc_tokens)
		print(example)
		print('tok_start_position')
		print(tok_start_position)
		print('tok_end_position')
		print(tok_end_position)
		exit(0)

		# The -3 accounts for [CLS], [SEP] and [SEP]
		max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

		doc_spans = []
		_DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
		start_offset = 0
		while start_offset < len(all_doc_tokens):
			length = len(all_doc_tokens) - start_offset
			if length > max_tokens_for_doc:
				length = max_tokens_for_doc
			doc_spans.append(_DocSpan(start=start_offset, length=length))
			if start_offset + length == len(all_doc_tokens):
				break
			start_offset += min(length, doc_stride)

		for (doc_span_index, doc_span) in enumerate(doc_spans):
			tokens = []
			token_to_orig_map = {}
			token_is_max_context = {}
			segment_ids = []
			tokens.append("[CLS]")
			segment_ids.append(0)
			for token in query_tokens:
				tokens.append(token)
				segment_ids.append(0)
			tokens.append("[SEP]")
			segment_ids.append(0)

			for i in range(doc_span.length):
				split_token_index = doc_span.start + i
				token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
				is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
				token_is_max_context[len(tokens)] = is_max_context
				tokens.append(all_doc_tokens[split_token_index])
				segment_ids.append(1)
			tokens.append("[SEP]")
			segment_ids.append(1)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)

			# The mask has 1 for real tokens and 0 for padding tokens. Only real
			# tokens are attended to.
			input_mask = [1] * len(input_ids)

			# Zero-pad up to the sequence length.
			while len(input_ids) < max_seq_length:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)

			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(segment_ids) == max_seq_length

			start_position = None
			end_position = None
			if is_training:
				# For training, if our document chunk does not contain an annotation
				# we throw it out, since there is nothing to predict.
				if tok_start_position == -1 and tok_end_position == -1:
					start_position = 0  # 问题本来没答案，0是[CLS]的位子
					end_position = 0
				else:  # 如果原本是有答案的，那么去除没有答案的feature
					out_of_span = False
					doc_start = doc_span.start  # 映射回原文的起点和终点
					doc_end = doc_span.start + doc_span.length - 1

					if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
						out_of_span = True
					if out_of_span:
						start_position = 0
						end_position = 0
					else:
						doc_offset = len(query_tokens) + 2
						start_position = tok_start_position - doc_start + doc_offset
						end_position = tok_end_position - doc_start + doc_offset

			features.append({'unique_id': unique_id,
							 'example_index': example_index,
							 'doc_span_index': doc_span_index,
							 'tokens': tokens,
							 'token_to_orig_map': token_to_orig_map,
							 'token_is_max_context': token_is_max_context,
							 'input_ids': input_ids,
							 'input_mask': input_mask,
							 'segment_ids': segment_ids,
							 'start_position': start_position,
							 'end_position': end_position})
			unique_id += 1

	print('features num:', len(features))
	json.dump(features, open(output_files[1], 'w'), ensure_ascii=False)


def _convert_index(index, pos, M=None, is_start=True):
	if pos >= len(index):
		pos = len(index) - 1
	if index[pos] is not None:
		return index[pos]
	N = len(index)
	rear = pos
	while rear < N - 1 and index[rear] is None:
		rear += 1
	front = pos
	while front > 0 and index[front] is None:
		front -= 1
	assert index[front] is not None or index[rear] is not None
	if index[front] is None:
		if index[rear] >= 1:
			if is_start:
				return 0
			else:
				return index[rear] - 1
		return index[rear]
	if index[rear] is None:
		if M is not None and index[front] < M - 1:
			if is_start:
				return index[front] + 1
			else:
				return M - 1
		return index[front]
	if is_start:
		if index[rear] > index[front] + 1:
			return index[front] + 1
		else:
			return index[rear]
	else:
		if index[rear] > index[front] + 1:
			return index[rear] - 1
		else:
			return index[front]


def read_cmrc_examples(input_file, is_training, two_level_embeddings):
	# if two_level_embeddings:
	#     return read_cmrc_examples_twolevel(input_file, is_training)

	with open(input_file, 'r') as f:
		train_data = json.load(f)
	train_data = train_data['data']

	def _is_chinese_char(cp):
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

	def is_fuhao(c):
		if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
				or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
				or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
				or c == '‘' or c == '’':
			return True
		return False

	def _tokenize_chinese_chars(text):
		"""Adds whitespace around any CJK character."""
		output = []
		for char in text:
			cp = ord(char)
			if _is_chinese_char(cp) or is_fuhao(char):
				if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
					output.append(SPIECE_UNDERLINE)
				output.append(char)
				output.append(SPIECE_UNDERLINE)
			else:
				output.append(char)
		return "".join(output)

	def is_whitespace(c):
		if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
			return True
		return False

	# to examples
	examples = []
	mis_match = 0
	for article in tqdm(train_data):
		for para in article['paragraphs']:
			context = para['context']
			if two_level_embeddings:
				context = context.replace('\u200b', '')
				context = context.replace(u'\xa0', u'')
				# Adjust answer position accordingly
				for i, qas in enumerate(para['qas']):
					ans_text = qas['answers'][0]['text']
					ans_start = qas['answers'][0]['answer_start']
					if ans_text != context[ans_start:ans_start + len(ans_text)]:
						lo = None
						for offset in range(-3, 4):
							lo = ans_start + offset
							if context[lo:lo+len(ans_text)] == ans_text:
								break
						para['qas'][i]['answers'][0]['answer_start'] = lo

			context_chs = _tokenize_chinese_chars(context)
			doc_tokens = []
			char_to_word_offset = []
			prev_is_whitespace = True
			for c in context_chs:
				if is_whitespace(c):
					prev_is_whitespace = True
				else:
					if prev_is_whitespace:
						doc_tokens.append(c)
					else:
						doc_tokens[-1] += c
					prev_is_whitespace = False
				if c != SPIECE_UNDERLINE:
					char_to_word_offset.append(len(doc_tokens) - 1)

			for qas in para['qas']:
				qid = qas['id']
				ques_text = qas['question']
				ans_text = qas['answers'][0]['text']

				start_position_final = None
				end_position_final = None
				# if is_training:
				if True:
					count_i = 0
					start_position = qas['answers'][0]['answer_start']

					end_position = start_position + len(ans_text) - 1
					repeat_limit = 3
					while context[start_position:end_position + 1] != ans_text and count_i < repeat_limit:
						start_position -= 1
						end_position -= 1
						count_i += 1

					while context[start_position] == " " or context[start_position] == "\t" or \
							context[start_position] == "\r" or context[start_position] == "\n":
						start_position += 1

					start_position_final = char_to_word_offset[start_position]
					end_position_final = char_to_word_offset[end_position]

					if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
						start_position_final += 1

					actual_text = "".join(doc_tokens[start_position_final:(end_position_final + 1)])
					cleaned_answer_text = "".join(tokenization.whitespace_tokenize(ans_text))

					if actual_text != cleaned_answer_text:
						print(actual_text, 'V.S', cleaned_answer_text)
						mis_match += 1
						# ipdb.set_trace()

				examples.append({'doc_tokens': doc_tokens,
								 'orig_answer_text': ans_text,
								 'qid': qid,
								 'question': ques_text,
								 'answer': ans_text,
								 'start_position': start_position_final,
								 'end_position': end_position_final})
	return examples, mis_match


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
	if ''.join(tokens) != ''.join(subchars):
		print(tokens)
		print(subchars)
		print('\n\n')
	assert ''.join(tokens) == ''.join(subchars)
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


def convert_examples_to_features_twolevel(
	examples,
	tokenizer,
	max_seq_length=512,
	max_query_length=64,
	doc_stride=128,
 	include_long_tokens=False):

	is_training = True  # TODO: Just remove this parameter altogether.

	features = []
	unique_id = 1000000000
	for (example_index, example) in enumerate(tqdm(examples)):
		question = example['question']
		query_tokens = tokenizer.tokenize(question)
		# Subchars (technically, it's subwords)
		# query_subchars = []
		# for c in question:
		#     query_subchars += tokenizer.tokenize(c)

		query_tokens = query_tokens[:max_query_length]  # Truncate too long questions

		# query_subchar_pos = get_subchar_pos(query_tokens, query_subchars)
		# if len(query_subchars) > max_query_length:
			# query_tokens = query_tokens[0:max_query_length]
		# query_subchars = query_subchars[:max_query_length]
		# query_subchar_pos = query_subchar_pos[:max_query_length]

		doc_tokens = example['doc_tokens']

		tok_to_orig_index = []
		orig_to_tok_index = []
		all_doc_tokens = []
		for (i, token) in enumerate(doc_tokens):
			orig_to_tok_index.append(len(all_doc_tokens))
			sub_tokens = tokenizer.tokenize(token)
			for sub_token in sub_tokens:
				tok_to_orig_index.append(i)
				all_doc_tokens.append(sub_token)

		long_tokens = tokenizer.tokenize(''.join(doc_tokens))
		
		# Zhuyin long tokenization result might have excessive characters: "▁" (UNICODE: 9601)
		# So, remove it.
		for i in range(len(long_tokens)):
			long_tokens[i] = long_tokens[i].replace('▁', '')
		
		try:
			doc_tok_to_long_index = get_subchar_pos(long_tokens, all_doc_tokens)
		except:
			print(doc_tokens)
			print(long_tokens)
			print(all_doc_tokens)
			# a = tokenizer.tokenize('俄罗斯圣彼得堡的模特儿。')
			# s = '圣彼得堡的'
			# b = tokenizer.tokenize(s)
			# for i in range(len(b)):
			# 	b[i] = b[i].replace('▁', '')
			# bs = sum([tokenizer.tokenize(c) for c in s], [])
			# # print(a)
			# print(b)
			# print(bs)
			# # print(''.join(a))
			# a = ''.join(b)
			# b = ''.join(bs)
			# for i in range(len(a)):
			# 	print(i, a[i], ord(a[i]), b[i], ord(b[i]))
			exit()

		# print(doc_tokens)
		# exit()

		tok_start_position = None
		tok_end_position = None
		if is_training:
			tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
			if example['end_position'] < len(example['doc_tokens']) - 1:
				tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
			else:
				tok_end_position = len(all_doc_tokens) - 1
			(tok_start_position, tok_end_position) = _improve_answer_span(
				all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
				example['orig_answer_text'])

		# text = example['text']
		# doc_index_char_to_subchar = []  # index of the beginning subchar of each character
		# doc_index_subchar_to_char = []  # index of each subchar to the char that contains it
		# # Get subchars
		# doc_subchars = []
		# for i, c in enumerate(text):
		#     doc_index_char_to_subchar.append(len(doc_subchars))
		#     this_subchars = tokenizer.tokenize(c)
		#     for subchar in this_subchars:
		#         doc_index_subchar_to_char.append(i)
		#         doc_subchars.append(subchar)
		
		# # Get tokens
		# doc_tokens = tokenizer.tokenize(text)
		# doc_subchar_pos = get_subchar_pos(doc_tokens, doc_subchars)
		# doc_subchar_pos.append(len(doc_tokens))

		# start_pos = None  # start index of answer in subchars
		# end_pos = None    # end index of answer in subchars

		# start_pos = doc_index_char_to_subchar[example['start_position']]  # 原来token到新token的映射，这是新token的起点
		# if example['end_position'] < len(text) - 1:
		# 	end_pos = doc_index_char_to_subchar[example['end_position'] + 1] - 1
		# else:
		# 	end_pos = len(doc_subchars) - 1
		# (start_pos, end_pos) = _improve_answer_span(
		#     doc_subchars, start_pos, end_pos, tokenizer,
		#     example['orig_answer_text'])

		tok_start_position = None
		tok_end_position = None
		if is_training:
			tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
			if example['end_position'] < len(example['doc_tokens']) - 1:
				tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
			else:
				tok_end_position = len(all_doc_tokens) - 1
			(tok_start_position, tok_end_position) = _improve_answer_span(
				all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
				example['orig_answer_text'])

		# The -3 accounts for [CLS], [SEP] and [SEP]
		# max_tokens_for_doc = max_seq_length - len(query_subchars) - 3
		max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

		doc_spans = []
		_DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
		start_offset = 0
		while start_offset < len(all_doc_tokens):
			length = len(all_doc_tokens) - start_offset
			if length > max_tokens_for_doc:
				length = max_tokens_for_doc
			doc_spans.append(_DocSpan(start=start_offset, length=length))
			if start_offset + length == len(all_doc_tokens):
				break
			start_offset += min(length, doc_stride)

		# Each doc span may be a new feature
		for (doc_span_index, doc_span) in enumerate(doc_spans):
			#
			# Feature format:
			#    tokens      = [CLS] + <query tokens> + [SEP] + <all_doc_token> + [SEP]
			#    segment_id  =   0   0  ...               0   0   1   1   ...    1  1
			#    long_tokens = ...
			#    subchar_pos = ...
			#
			cls_token = '[CLS]'
			sep_token = '[SEP]'
			tokens = []
			token_to_orig_map = {}
			token_is_max_context = {}
			segment_ids = []

			tokens.append(cls_token)
			segment_ids.append(0)
			for token in query_tokens:
				tokens.append(token)
				segment_ids.append(0)
			tokens.append(sep_token)
			segment_ids.append(0)
			token_to_long_index = [-1] * len(tokens)  # Map index of token to long_tokens

			# subchars = [cls_token] + query_subchars + [sep_token]
			# tokens = [cls_token] + query_tokens + [sep_token]
			# segment_ids = [0]
			# subchar_pos = [0] + [x+1 for x in query_subchar_pos] + [len(tokens) - 1]
			# segment_ids = [0] * len(subchars)
			
			# index_subchar_to_char = {}
			# subchar_is_max_context = {}

			for i in range(doc_span.length):
				split_token_index = doc_span.start + i
				token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
				is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
				token_is_max_context[len(tokens)] = is_max_context
				tokens.append(all_doc_tokens[split_token_index])
				segment_ids.append(1)
				
				token_to_long_index.append(doc_tok_to_long_index[split_token_index])
			tokens.append("[SEP]")
			segment_ids.append(1)
			token_to_long_index.append(-1)


			# Extract tokens that overlap with tokens in this span
			long_index_lo = doc_tok_to_long_index[doc_span.start]
			# long_index_hi = doc_tok_to_long_index[doc_span.start + doc_span.length]
			this_long_tokens = long_tokens[long_index_lo:]  		# Excessive long tokens on the right end doesn't matter
			this_long_tokens = this_long_tokens[:max_seq_length]  	# But can't be too long
			# Tweak index mapping of tokens accordingly
			for i in range(len(token_to_long_index)):
				if token_to_long_index[i] != -1:
					token_to_long_index[i] -= long_index_lo

			# print(this_long_tokens)
			# print(token_to_long_index, len(token_to_long_index))
			# print(tokens, len(tokens))
			# exit()

			# subchar_pos_offset = len(tokens)
			# right_pos = None
			# for i in range(doc_span.length):
			# 	split_subchar_index = doc_span.start + i
			# 	# token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
			# 	index_subchar_to_char[len(subchars)] = doc_index_subchar_to_char[split_subchar_index]
			# 	is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_subchar_index)
			# 	# token_is_max_context[len(tokens)] = is_max_context
			# 	subchar_is_max_context[len(subchars)] = is_max_context
			# 	# tokens.append(all_doc_tokens[split_token_index])
			# 	subchars.append(doc_subchars[split_subchar_index])
			# 	segment_ids.append(1)
			# 	subchar_pos.append(len(tokens))

			# 	# Add all tokens that overlap with this subchar
			# 	prev_right_pos = right_pos if right_pos is not None else doc_span.start
			# 	# left_pos = doc_subchar_pos[split_subchar_index]
			# 	right_pos = doc_subchar_pos[split_subchar_index + 1]
			# 	for token_index in range(prev_right_pos, right_pos):
			# 		tokens.append(doc_tokens[token_index])
			# subchars.append(sep_token)
			# tokens.append(sep_token)
			# subchar_pos.append(len(tokens) - 1)
			# segment_ids.append(1)

			# Adjust start_pos and end_pos according to prefixed query
			# start_pos += len(query_subchars) + 2
			# end_pos += len(query_subchars) + 2

			# input_ids = tokenizer.convert_tokens_to_ids(subchars)
			# token_ids = tokenizer.convert_tokens_to_ids(tokens)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)
			token_ids = tokenizer.convert_tokens_to_ids(long_tokens)

			# The mask has 1 for real tokens and 0 for padding tokens. Only real
			# tokens are attended to.
			input_mask = [1] * len(input_ids)

			# Turn token_to_long_index into left and right pos
			pos_left = []
			pos_right = []
			for i, index in enumerate(token_to_long_index):
				if index == -1:
					pos_left.append(-1)
					pos_right.append(-1)
				else:
					pos_left.append(index)
					# Make sure right index is at least greater than left index. 
					# next_index will be -1 for the rightmost token, and next_index will 
					# be equal to this index when next token map to same long token.
					next_index = token_to_long_index[i+1]
					pos_right.append(max(index + 1, next_index))

			# subchar_pos.append(len(token_ids))
			# pos_left = [subchar_pos[i] for i in range(len(subchar_pos) - 1)]
			# pos_right = [subchar_pos[i+1] for i in range(len(subchar_pos) - 1)]

			# # When multiple subchars map to one token, corresponding pos_left and pos_right will be equal
			# # except for the rightmost subchar. We add one to make sure the token embedding will be added
			# # to all subchars.
			# for i in range(len(pos_left) - 1):
			# 	if pos_left[i] == pos_right[i]:
			# 		pos_right[i] += 1

			# Zero-pad up to the sequence length.
			padding_len = max_seq_length - len(input_ids)
			token_padding_len = max_seq_length - len(token_ids)

			input_ids += [0] * padding_len
			pos_left += [-1] * padding_len  # Doesn't matter
			pos_right += [-1] * padding_len
			input_mask += [0] * padding_len
			segment_ids += [0] * padding_len
			token_ids += [0] * token_padding_len

			# There might be more tokens than subchars, so we need to truncate it.
			token_ids = token_ids[:max_seq_length]

			if len(token_ids) != max_seq_length:
				print(len(token_ids))
				print(max_seq_length)

			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(segment_ids) == max_seq_length
			assert len(token_ids) == max_seq_length
			assert len(pos_left) == len(pos_right) == max_seq_length

			start_position = None
			end_position = None
			is_training = True
			if is_training:
				# For training, if our document chunk does not contain an annotation
				# we throw it out, since there is nothing to predict.
				if tok_start_position == -1 and tok_end_position == -1:
					start_position = 0  # 问题本来没答案，0是[CLS]的位子
					end_position = 0
				else:  # 如果原本是有答案的，那么去除没有答案的feature
					out_of_span = False
					doc_start = doc_span.start  # 映射回原文的起点和终点
					doc_end = doc_span.start + doc_span.length - 1

					if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
						out_of_span = True
					if out_of_span:
						start_position = 0
						end_position = 0
					else:
						doc_offset = len(query_tokens) + 2
						start_position = tok_start_position - doc_start + doc_offset
						end_position = tok_end_position - doc_start + doc_offset

			feature = {'unique_id': unique_id,
					   'example_index': example_index,
					   'doc_span_index': doc_span_index,
					#    'text': text,
					#    'subchars': subchars,
					   'tokens': tokens,
					   'token_to_orig_map': token_to_orig_map,
					   'token_is_max_context': token_is_max_context,
					   'input_ids': input_ids,
					   'input_mask': input_mask,
					   'segment_ids': segment_ids,
					   'start_position': start_position,
					   'end_position': end_position,

					   # For token embeddings
					   'token_ids': token_ids,
					   'pos_left': pos_left,
					   'pos_right': pos_right,
					#    'index_subchar_to_char': index_subchar_to_char,
					#    'subchar_is_max_context': subchar_is_max_context,
					   }
			if include_long_tokens:
				feature['long_tokens'] = long_tokens
			# print(token_ids)
			# exit()
			features.append(feature)
			unique_id += 1

	return features


def convert_examples_to_features(
	examples,
	tokenizer,
	max_seq_length=512,
	max_query_length=64,
	doc_stride=128,
	two_level_embeddings=False,
 	include_long_tokens=False):
	if two_level_embeddings:
		return convert_examples_to_features_twolevel(
			examples,
			tokenizer,
			max_seq_length=max_seq_length,
			max_query_length=max_query_length,
			doc_stride=doc_stride,
			include_long_tokens=include_long_tokens,
		)

	is_training = True # TODO: Just remove this parameter altogether?

	features = []
	unique_id = 1000000000
	for (example_index, example) in enumerate(tqdm(examples)):
		query_tokens = tokenizer.tokenize(example['question'])
		if len(query_tokens) > max_query_length:
			query_tokens = query_tokens[0:max_query_length]

		tok_to_orig_index = []
		orig_to_tok_index = []
		all_doc_tokens = []
		for (i, token) in enumerate(example['doc_tokens']):
			orig_to_tok_index.append(len(all_doc_tokens))
			sub_tokens = tokenizer.tokenize(token)
			for sub_token in sub_tokens:
				tok_to_orig_index.append(i)
				all_doc_tokens.append(sub_token)

		tok_start_position = None
		tok_end_position = None
		if is_training:
			tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
			if example['end_position'] < len(example['doc_tokens']) - 1:
				tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
			else:
				tok_end_position = len(all_doc_tokens) - 1
			(tok_start_position, tok_end_position) = _improve_answer_span(
				all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
				example['orig_answer_text'])

		# The -3 accounts for [CLS], [SEP] and [SEP]
		max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

		doc_spans = []
		_DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
		start_offset = 0
		while start_offset < len(all_doc_tokens):
			length = len(all_doc_tokens) - start_offset
			if length > max_tokens_for_doc:
				length = max_tokens_for_doc
			doc_spans.append(_DocSpan(start=start_offset, length=length))
			if start_offset + length == len(all_doc_tokens):
				break
			start_offset += min(length, doc_stride)

		for (doc_span_index, doc_span) in enumerate(doc_spans):
			tokens = []
			token_to_orig_map = {}
			token_is_max_context = {}
			segment_ids = []

			tokens.append("[CLS]")
			segment_ids.append(0)
			for token in query_tokens:
				tokens.append(token)
				segment_ids.append(0)
			tokens.append("[SEP]")
			segment_ids.append(0)

			for i in range(doc_span.length):
				split_token_index = doc_span.start + i
				token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
				is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
				token_is_max_context[len(tokens)] = is_max_context
				tokens.append(all_doc_tokens[split_token_index])
				segment_ids.append(1)
			tokens.append("[SEP]")
			segment_ids.append(1)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)

			# The mask has 1 for real tokens and 0 for padding tokens. Only real
			# tokens are attended to.
			input_mask = [1] * len(input_ids)

			# Zero-pad up to the sequence length.
			while len(input_ids) < max_seq_length:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)

			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(segment_ids) == max_seq_length

			start_position = None
			end_position = None
			is_training = True
			if is_training:
				# For training, if our document chunk does not contain an annotation
				# we throw it out, since there is nothing to predict.
				if tok_start_position == -1 and tok_end_position == -1:
					start_position = 0  # 问题本来没答案，0是[CLS]的位子
					end_position = 0
				else:  # 如果原本是有答案的，那么去除没有答案的feature
					out_of_span = False
					doc_start = doc_span.start  # 映射回原文的起点和终点
					doc_end = doc_span.start + doc_span.length - 1

					if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
						out_of_span = True
					if out_of_span:
						start_position = 0
						end_position = 0
					else:
						doc_offset = len(query_tokens) + 2
						start_position = tok_start_position - doc_start + doc_offset
						end_position = tok_end_position - doc_start + doc_offset

			features.append({'unique_id': unique_id,
							 'example_index': example_index,
							 'doc_span_index': doc_span_index,
							 'tokens': tokens,
							 'token_to_orig_map': token_to_orig_map,
							 'token_is_max_context': token_is_max_context,
							 'input_ids': input_ids,
							 'input_mask': input_mask,
							 'segment_ids': segment_ids,
							 'start_position': start_position,
							 'end_position': end_position})
			unique_id += 1

	return features
 
 
