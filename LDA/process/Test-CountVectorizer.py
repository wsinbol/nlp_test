from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    '我 爱 北京 天安门',
    '天安门 广场',
    '北京 天安门 北京',
]

test = [
	'哈哈 新年',
	'北京 王府井',
]

stop_words=['北京','广场']
vectorizer = CountVectorizer(stop_words=None)
# print(vectorizer.get_stop_words())
X = vectorizer.fit_transform(corpus)
print(X)

X = vectorizer.fit(corpus)
print(X.transform(corpus))
exit()

# 词袋向量
print(vectorizer.get_feature_names())
# (文档编号,词语编号)[从0开始] 频率
print(X)
# 转成矩阵形式
print(X.toarray())

print('-'*4)

# 新数据集在词袋向量中的统计
Y = vectorizer.transform(test)
# 计数形式
print(Y)
# 矩阵形式
print(Y.toarray())
