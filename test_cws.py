from tokenization import CWSNewTokenizer

tokenizer = CWSNewTokenizer('res/wubi_cws_22675_0.65.vocab', 'res/wubi_cws_22675_0.65.model', 'res/wubi_cws_22675_0.65.cws_vocab')

print (0.65)
print(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！'))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！')))
print ()



tokenizer = CWSNewTokenizer('res/wubi_cws_22675_0.7.vocab', 'res/wubi_cws_22675_0.7.model', 'res/wubi_cws_22675_0.7.cws_vocab')

print (0.7)
print(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！'))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！')))
print ()



tokenizer = CWSNewTokenizer('res/wubi_cws_22675_0.75.vocab', 'res/wubi_cws_22675_0.75.model', 'res/wubi_cws_22675_0.75.cws_vocab')

print (0.75)
print(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！'))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！')))
print ()


tokenizer = CWSNewTokenizer('res/wubi_cws_22675_0.8.vocab', 'res/wubi_cws_22675_0.8.model', 'res/wubi_cws_22675_0.8.cws_vocab')

print (0.8)
print(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！'))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！')))
print ()


tokenizer = CWSNewTokenizer('res/pinyin_cws_22675_0.65.vocab', 'res/pinyin_cws_22675_0.65.model', 'res/pinyin_cws_22675_0.65.cws_vocab')

print (0.65)
print(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！'))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！')))
print ()


tokenizer = CWSNewTokenizer('res/pinyin_cws_22675_0.7.vocab', 'res/pinyin_cws_22675_0.7.model', 'res/pinyin_cws_22675_0.7.cws_vocab')

print (0.7)
print(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！'))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！')))
print ()


tokenizer = CWSNewTokenizer('res/pinyin_cws_22675_0.75.vocab', 'res/pinyin_cws_22675_0.75.model', 'res/pinyin_cws_22675_0.75.cws_vocab')

print (0.75)
print(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！'))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！')))
print ()


tokenizer = CWSNewTokenizer('res/pinyin_cws_22675_0.8.vocab', 'res/pinyin_cws_22675_0.8.model', 'res/pinyin_cws_22675_0.8.cws_vocab')

print (0.8)
print(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！'))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('如今把事实指出，愈使魑魅魍魉无所遁形于光天化日之下了！')))
print ()

