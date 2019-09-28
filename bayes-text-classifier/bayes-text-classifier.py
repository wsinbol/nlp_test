# -*- coding:utf-8 -*-
import os
import jieba
import random
import sklearn
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

folder_path = './Sample'
stopwords_file = './stopwords_cn.txt'
log_file = './bayes-text-classifier.log'

def make_word_set(words_file):
	words_set = set()
	with open(words_file, 'r', encoding = 'utf-8') as f:
		for line in f.readlines():
			word = line.strip()
			if len(word) > 0 and word not in words_set:
				words_set.add(word)
	return words_set

# 文本处理过程OR样本生成过程
def text_processing(folder_path, text_size=0.2):
	folder_list = os.listdir(folder_path)
	data_list = []
	class_list = []

	for folder in folder_list:
		new_folder_path = os.path.join(folder_path, folder)
		files = os.listdir(new_folder_path)
		num = 1
		for file in files:
			if num > 2:
				break

			with open(os.path.join(new_folder_path, file), 'r', encoding = 'utf-8') as f:
				raw = f.read()
			word_cut = jieba.cut(raw, cut_all=False) # 返回generator结构
			word_list = list(word_cut)

			data_list.append(word_list)
			class_list.append(folder)
			num += 1

	# 手动划分训练街和测试集
	data_class_list = zip(data_list, class_list) # 将数据集和分类集一一对应
	data_class_list = list(data_class_list)
	# random.shuffle(data_class_list)	# 打乱数据

	# 按照2-8占比划分数据
	index = int(len(list(data_class_list)) * text_size) + 1 
	train_list = data_class_list[:index]
	test_list = data_class_list[index:]
	
	train_data_list, train_class_list = zip(*train_list)
	test_data_list, test_class_list = zip(*test_list)

	all_words_dict = {}
	for word_list in train_data_list:
		for word in word_list:
			if all_words_dict.get(word):
				all_words_dict[word] += 1
			else:
				all_words_dict[word] = 1

	# 根据词频降序排列
	all_words_tuple_list = sorted(all_words_dict.items(), key = lambda f:f[1], reverse=True)
	# 获得排列后的分词结果
	all_words_list = list(zip(*all_words_tuple_list))[0]
	return all_words_list, train_data_list, train_class_list, test_data_list, test_class_list
	

# 获得特征词，竟然用这种方式？
def words_dict(all_words_list, deleteN, stopwords_set=set()):
	features_words = []
	n = 1
	# deleteN为0-980中间隔为20的数，生成的特征词维度基本都可以满足1000维
	for t in range(deleteN, len(all_words_list), 1):
		if n > 1000:
			break
		# 特征词不为小数,不为停用词，长度小于5
		if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
			features_words.append(all_words_list[t])			
			n += 1

	with open(log_file, 'a', encoding = 'utf-8') as log:
		log.write(' '.join(features_words))
		log.write('【共个'+ str(len(features_words)) +'特征词】')

	return features_words

# 文本特征
def text_features(train_data_list, test_data_list, features_words):
	def text_features(text, features_words):
		text_words = set(text)
		features = [1 if word in text_words else 0 for word in features_words]
		return features
	train_feature_list = [text_features(text, features_words) for text in train_data_list]
	test_feature_list = [text_features(text, features_words) for text in test_data_list]
	return train_feature_list, test_feature_list

def text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
	classifier = MultinomialNB().fit(train_feature_list, train_class_list)
	# 测试 测试集文本的分类
	print(classifier.predict(test_feature_list))
	test_accuracy = classifier.score(test_feature_list, test_class_list)
	return test_accuracy



if __name__ == '__main__':
	stopwords_set = make_word_set(stopwords_file)
	all_words_list, train_data_list, train_class_list, test_data_list, test_class_list = text_processing(folder_path)
	
	if os.path.exists(log_file):
		os.remove(log_file)
	
	deleteNs = range(0, 2, 20)
	test_accuracy_list = []

	for deleteN in deleteNs:
		features_words = words_dict(all_words_list, deleteN, stopwords_set)
		text_features(train_data_list, test_data_list, features_words)
		train_feature_list, test_feature_list = text_features(train_data_list, test_data_list, features_words)
		test_accuracy = text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
		test_accuracy_list.append(test_accuracy)
	plt.plot(deleteNs, test_accuracy_list)
	plt.xlabel('deleteNs')
	plt.ylabel('test_accuracy')
	plt.show()

	