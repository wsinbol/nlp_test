# -*- coding:utf-8 -*-
import os
import jieba
import random

folder_path = './Sample'
stopwords_file = './stopwords_cn.txt'

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
			if num > 100:
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
	

# 获得特征词
def words_dict(all_words_list, deleteN, stopwords_set=set()):
	features_words = []
	n = 1
	for t in range(deleteN, len(all_words_list), 1):
		if n > 1000:
			break
		if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
			features_words.append(all_words_list[t])			
			n += 1
		return features_words




if __name__ == '__main__':
	stopwords_set = make_word_set(stopwords_file)
	all_words_list, train_data_list, train_class_list, test_data_list, test_class_list = text_processing(folder_path)
	features_words = words_dict(all_words_list, 0, stopwords_set)