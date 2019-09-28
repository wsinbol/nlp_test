import numpy as np

data = list(open("./quora.txt", encoding='utf-8'))


from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()
# print([line for line in data[:2]])
# exit()
# data_tok = [item.lower() for item in tokenizer.tokenize(data[10])]
data_tok = []
for line in data:
	if line:
		data_tok.append((item.lower() for item in tokenizer.tokenize(line)))

# data_tok = 
# exit()

# print(data_tok)
# assert all(isinstance(row, (list, tuple)) for row in data_tok)
# assert all(isinstance(row, (list)) for row in data_tok)
# print([''.join(row) for row in data_tok[:]])


from gensim.models import Word2Vec
model = Word2Vec(data_tok, 
                 size=32,      # embedding vector size
                 min_count=5,  # consider words that occured at least 5 times
                 window=5).wv  # define context as a 5-word window around the target word

model.get_vector('which')
print(model.most_similar('bread'))