
import numpy as np
import pandas as pd
import re
from gensim import corpora, models, similarities
import gensim
from nltk.corpus import stopwords

df = pd.read_csv('./input/HillaryEmails.csv')
# 取ID、ExtractedBodyText两列内容，并将空值的行去掉
df = df[['Id', 'ExtractedBodyText']].dropna(axis = 0)

def clean_email_text(text):
	text = text.replace('\n', " ")
	text = re.sub(r'-', " ", text)
	text = re.sub(r"\d+/\d+/\d+", "", text)
	text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text)
	text = re.sub(r"[\w]+@[\.\w]+", "", text)
	text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text)

	pure_text = ''
	for letter in text:
		if letter.isalpha() or letter == ' ':
			pure_text += letter
	text = " ".join(word for word in pure_text.split() if len(word) > 1)
	return text

docs = df['ExtractedBodyText']
docs = docs.apply(lambda s: clean_email_text(s))

doclist = docs.values

stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours', 
'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their', 
'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once', 
'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you', 
'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will', 
'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be', 
'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself', 
'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both', 
'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn', 
'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about', 
'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn', 
'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in docs]

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]
# print(corpus)
# exit()

# 训练过程
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)

# 查看#10分类
# print(lda.print_topic(18, topn=200))

# 打印所有主题,20个主题的前5个词
lda.print_topics(num_topics=20, num_words=5)

print('-------------------预测开始---------------------')
print(dictionary[99],dictionary[331])
sentence = "I love our country, and I believe in our people, and I will never, ever quit on you. No matter what"

print(lda.get_document_topics([(99,1),(331,1)]))




