# -*- coding:utf-8 -*-
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

origin_data_path = './news_data.txt'
stopwords_file = './stopwords_cn.txt'

stopwords_set = set()
with open(stopwords_file, 'r', encoding='utf-8') as f:
	for word in f.readlines():
		if len(word) > 0 and word not in stopwords_set:
				word = word.strip()
				stopwords_set.add(word)

def text_processing(file_path):
	with open(file_path, 'r', encoding='utf-8') as f:
		dataset = f.read()

	pattern = re.compile(r'<content>(.*?)</content>', re.S)
	content_list = re.findall(pattern, dataset)
	content_data = []
	for item in content_list:
		if len(item) > 1:
			content_data.append(item)
	
	clean_words = []
	for item in content_data:
		words = list(jieba.cut(item))
		clean_words.append([word for word in words if word not in stopwords_set and not word.isdigit() and len(word) > 1])

	corpus = []
	for word in clean_words:
		corpus.append(' '.join(word))

	return corpus

def make_count_vectorizer(corpus):
	vectorizer = CountVectorizer(stop_words=None)
	tf = vectorizer.fit_transform(corpus)
	return tf


corpus = text_processing(origin_data_path)
tf = make_count_vectorizer(corpus)
lda = LatentDirichletAllocation(n_topics=10,
                                    max_iter=8000,
                                    learning_method='batch',
                                    evaluate_every=200,
                                    perp_tol=0.01)
model = lda.fit(tf)
for topic_idx, topic in enumerate(model.components_):
	print(topic_idx, topic)
