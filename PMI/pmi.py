# -*- coding:utf-8 -*-

import math
import re
corpus = [
 '支持 华为',
 '华为 加油',
 '加油 中国',
]

# print(math.log(10,10))
doc_count = len(corpus)

all_words_set = dict()

pattern = '\s'

text_words = []
for text in corpus:
	words = re.split(pattern, text)
	text_words.append(words)
# print(text_words)

for text in corpus:
	words = re.split(pattern, text)
	for word in words:
		if word in all_words_set:
			all_words_set[word] += 1
		else:
			all_words_set[word] = 1

print(all_words_set)



exit()
all_words_list = []
for k,v in all_words_set.items():
	all_words_list.append([k,v])
# print(all_words_list)

word1_word2 = []
for i,k_v in enumerate(all_words_list):
	for j in range(i+1,len(all_words_list)):
		k_1, k_2 = k_v[0], all_words_list[j][0]
		word1_word2.append(k_1 + '&' + k_2)
	# print()
# print(word1_word2)

calc_word1_word2 = dict()
for word in word1_word2:
	word1, word2 = re.split('&', word)
	calc_word1_word2[word] = 0
	for text in text_words:
		if word1 in text and word2 in text:
			calc_word1_word2[word] += 1
print(calc_word1_word2)

for word, count in calc_word1_word2.items():
	# print(item)
	word1, word2 = re.split('&', word)
	if count:
		print(word,word1,word2,'',math.log((count/doc_count)/((all_words_set[word1]/doc_count) * (all_words_set[word2]/doc_count)),2))
