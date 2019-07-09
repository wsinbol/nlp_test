# -*- coding:utf-8 -*-
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

origin_data_path = './news_data.txt'
stopwords_file = './stopwords_cn.txt'
n_top_words = 40

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
	# 未进行归一化
	tf = vectorizer.fit_transform(corpus)
	feature_names = vectorizer.get_feature_names()
	return tf,feature_names


corpus = text_processing(origin_data_path)
tf, feature_names = make_count_vectorizer(corpus)
lda = LatentDirichletAllocation(n_components=10,
                                    max_iter=8000,
                                    learning_method='batch',
                                    evaluate_every=200,
                                    perp_tol=0.01)
model = lda.fit(tf)
# print(model.components_)

# for topic_idx, topic in enumerate(model.components_):
	# print(topic_idx, topic)

with open('res_topic_word.txt', 'w') as f:
        f.write("Topic, Top Word\n")
        # components_ : array, [n_components, n_features]
        # components_[i, j] can be viewed as pseudocount that represents the number of times word j was assigned to topic i
        for topic_idx, topic in enumerate(model.components_):
        # for topic_idx, topic in model.components_:
            f.write(str(topic_idx) + ',')
            # topic.argsort()[:-n_top_words - 1:-1] 降序返回特征词在词袋中的索引
            # (feature_names[i], topic[i]) 特征词向量自始至终都是一致的
            topic_word_dist = [(feature_names[i], topic[i]) for i in topic.argsort()[:-n_top_words - 1:-1]]
            print(topic_word_dist)
            for word, weight in topic_word_dist:
                f.write(word + '#' + str(weight) + ';')
            f.write('\n')

