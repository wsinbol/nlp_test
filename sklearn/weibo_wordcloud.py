#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import jieba

df = pd.read_csv('./weibo_text_score.csv', header=0)
corpus = df['content'].values

stop_words = set()
with open('stopwords_cn.txt',encoding='utf-8') as rf:
    for line in rf:
        stop_words.add(line.strip())


# In[ ]:


pos_words = []
with open('sentiment_words_cn.txt', encoding='utf-8') as rf:
	for line in rf.readlines():
		word, score = line.strip().split(' ')
		if float(score) < 0:
			pos_words.append(word)


# In[ ]:


word_corpus = []
for text in corpus:
    word_corpus.append(" ".join([word for word in jieba.lcut(text) if len(word) > 1 and                                  str(word).isalpha() and word not in stop_words and word in pos_words]))
word_corpus


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(word_corpus)
feature_names = vectorizer.get_feature_names()
# feature_names
print(X)

'''
doc_item_mat = X.toarray()
rows,cols = np.shape(doc_item_mat)
for row_index, row in enumerate(doc_item_mat):
    print('No.',row_index)
    print([feature_names[i] for i in row.argsort()[:-10:-1]])
'''


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(word_corpus)
word = vectorizer.get_feature_names()
count = np.sum(X.toarray(), axis=0)
# print(len(count))
combine_dict = dict()
combile = zip(word, count)

combine_dict = {word:count for word,count in combile}
print(combine_dict)
# combine_dict = sorted(combine_dict.items(), key=lambda x:x[1], reverse=True)
# print(combine_dict)


# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
font = r'C:\Windows\Fonts\simfang.ttf'
wc = WordCloud(background_color="white", max_words=10000,font_path=font)
wc.generate_from_frequencies(combine_dict)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

