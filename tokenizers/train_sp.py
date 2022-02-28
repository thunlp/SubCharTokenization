import sentencepiece as spm

# spm.SentencePieceTrainer.train(input='/data2/private/clsi/wubi_corpus/baidubaike_corpus_cangjie.txt', model_prefix='cangjie_zh_22675', vocab_size=22675, \
# user_defined_symbols=['[PAD]', '[CLS]', '[SEP]', '[MASK]'], bos_id=-1, eos_id=-1, pad_id=-1, unk_piece='[UNK]', character_coverage=1.0, \
# train_extremely_large_corpus=True, add_dummy_prefix=False, split_by_number=False, split_by_unicode_script=False)

# spm.SentencePieceTrainer.train(input='/data2/private/clsi/wubi_corpus/baidubaike_corpus_stroke.txt', model_prefix='stroke_zh_22675', vocab_size=22675, \
# user_defined_symbols=['[PAD]', '[CLS]', '[SEP]', '[MASK]'], bos_id=-1, eos_id=-1, pad_id=-1, unk_piece='[UNK]', character_coverage=1.0, \
# train_extremely_large_corpus=True, add_dummy_prefix=False, split_by_number=False, split_by_unicode_script=False)

# spm.SentencePieceTrainer.train(input='/data2/private/clsi/wubi_corpus/baidubaike_corpus_zhengma.txt', model_prefix='zhengma_zh_22675', vocab_size=22675, \
# user_defined_symbols=['[PAD]', '[CLS]', '[SEP]', '[MASK]'], bos_id=-1, eos_id=-1, pad_id=-1, unk_piece='[UNK]', character_coverage=1.0, \

spm.SentencePieceTrainer.train(input='/data2/private/clsi/wubi_corpus/baidubaike_corpus_pinyin.txt', model_prefix='pinyin_zh_22675_bpe', vocab_size=22675, \
user_defined_symbols=['[PAD]', '[CLS]', '[SEP]', '[MASK]'], bos_id=-1, eos_id=-1, pad_id=-1, unk_piece='[UNK]', character_coverage=1.0, \
train_extremely_large_corpus=True, add_dummy_prefix=False, split_by_number=False, split_by_unicode_script=False, model_type="bpe")

# spm.SentencePieceTrainer.train(input='/data2/private/clsi/wubi_corpus/baidubaike_corpus_zhuyin.txt', model_prefix='zhuyin_zh_22675', vocab_size=22675, \
# user_defined_symbols=['[PAD]', '[CLS]', '[SEP]', '[MASK]'], bos_id=-1, eos_id=-1, pad_id=-1, unk_piece='[UNK]', character_coverage=1.0, \
# train_extremely_large_corpus=True, add_dummy_prefix=False, split_by_number=False, split_by_unicode_script=False)

# spm.SentencePieceTrainer.train(input='/data2/private/clsi/wubi_corpus/baidubaike_corpus.txt', model_prefix='raw_zh_22675', vocab_size=22675, \
# user_defined_symbols=['[PAD]', '[CLS]', '[SEP]', '[MASK]'], bos_id=-1, eos_id=-1, pad_id=-1, unk_piece='[UNK]', character_coverage=1.0, \
# train_extremely_large_corpus=True, add_dummy_prefix=False, split_by_number=False, split_by_unicode_script=False)
