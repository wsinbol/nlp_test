#!/usr/bin/env python
# coding: utf-8

# In[23]:


neg_words = set()
with open('./vader_lexicon.txt',encoding='utf-8') as f:
    for line in f.readlines():
        item = line.strip().split('\t')
        word = item[0]
        score = item[1]
        if float(score) > 0:
            neg_words.add(word)
print(len(neg_words))


# In[ ]:


import pandas as pd
import jieba

df = pd.read_csv('./twitter_text_score_5_10.csv', header=0)
word_corpus = df['content'].values
word_corpus


# In[24]:


from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import numpy as np
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(word_corpus)
word = vectorizer.get_feature_names()
count = np.sum(X.toarray(), axis=0)
# print(len(count))
combine_dict = dict()
combile = zip(word, count)

combine_dict = {word:count for word,count in combile if word in neg_words}
print(combine_dict)


# In[26]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
font = r'C:\Windows\Fonts\simfang.ttf'
wc = WordCloud(background_color="white", max_words=10000,font_path=font)
wc.generate_from_frequencies(combine_dict)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

