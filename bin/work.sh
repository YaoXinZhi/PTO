# 分割文件
python pubtator_split.py
# 批量预处理
python batch_corpus.py
# 连接语料库
cat ../data/abs_corpus/* ../data/ner_corpus/** ../data/pto_corpus/* > ../data/embedding/triple_corpus
# 训练embedding
python SGNS.py -cp "../data/embedding/triple_corpus" -sg 1 -hs 0
# 计算最相似的10个词
python Similarity_calculation.py
