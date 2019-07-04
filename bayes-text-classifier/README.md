### bayes-text-classifier [贝叶斯文本分类]
- 20190704 生成测试集test_feature_list过程中错误地遍历训练集数据train_data_list导致ValueError: Found input variables with inconsistent numbers of samples，怀疑原因是文本特征大小和文本容量大小不一致