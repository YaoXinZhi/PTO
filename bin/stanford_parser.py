# -*- coding:utf-8 -*-
#! usr/bin/env python3
"""
Created on 28/03/2020 下午1:10 
@Author: xinzhi yao 
"""

from nltk.parse.stanford import StanfordParser
import time

eng_parser = StanfordParser(model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

print (list(eng_parser.parse("the quick brown fox jumps over the lazy dog".split())))


sent1 = """the objective of this study is to examine the frequency development concomitants and risk factors of falls in a population-based incident parkinson s disease pd cohort"""
sent2 = """one hundred eighty-one drug-na ve patients with incident pd and NBR normal controls recruited from the norwegian parkwest study were prospectively monitored over NBR years"""
sent_list = [sent1, sent2]

# 1.2 s

for sent in sent_list:
    start_time = time.time()
    parse_result = eng_parser.parse(sent.split())
    end_time = time.time()
    print('解析一个句子, 花费 {0:.2f} s.'.format(end_time - start_time))



