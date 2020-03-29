# -*- coding:utf-8 -*-
#! usr/bin/env python3
"""
Created on 11/03/2020 下午8:26 
@Author: xinzhi yao
"""

import re
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import nltk
# from tqdm import tqdm


def read_pto(pto_file):
    opt_dic = defaultdict(list)
    count_term = 0
    count_word = 0
    with open(pto_file) as f:
        for line in f:
            l = line.strip()
            if l.startswith('id'):
                count_term += 1
                id = l.split()[1]
            if l.startswith('name'):
                count_word += 1
                name = ' '.join(l.split(' ')[1:])
                opt_dic[id].append(name)
            if l.startswith('synonym'):
                count_word += 1
                pattern = re.compile('"(.*)"')
                synonym = pattern.findall(l)
                synonym = ' '.join(synonym[0].split(' ')[:-1])
                opt_dic[id].append(synonym)
    print('一共 {0} terms, 包含 {1} 个名字及同义名.'.format(count_term, count_word))
    return opt_dic

def abs_read(abstract_file: str):
    title_abstract_list = []
    count_abs = 0
    with open(abstract_file) as f:
        line = f.readline()
        for line in f:
            l = line.strip().split('\t')
            try:
                title = l[0]
                abs = l[4]
                title_abstract_list.append((title, abs))
                count_abs += 1
            except:
                print(l)
                continue
    print('共 {0} 篇摘要.'.format(count_abs))
    return title_abstract_list


def save_result(ta_list: list, pto_dic: dict, out_file: str):
    wf = open(out_file, 'w')
    wf.write('PTO\tsentence\ttitle\tterm\n')
    count = 0
    count_result = 0
    for ta in ta_list:
        count += 1
        if count % 600 == 0:
            print('{0}/{1} abstracts process done.'.format(count, len(ta_list)))
        title = ta[0]
        abs = ta[1]
        sent_list = [i for i in sent_tokenize(abs)]
        for sent in sent_list:
            word_list = nltk.word_tokenize(sent)
            for term, name_list in pto_dic.items():
                for name in name_list:
                    if len(name.split(' ')) > 1:
                        if name.lower() in sent.lower():
                            wf.write('{0}\t{1}\t{2}\t{3}\n'. \
                                     format(term, sent, title, '|'.join(name_list)))
                            count_result += 1
                    else:
                        if name in word_list:
                            wf.write('{0}\t{1}\t{2}\t{3}\n'. \
                                     format(term, sent, title, '|'.join(name_list)))
                            count_result += 1
    print('共找到 {0} 个句子包含pto条目.'.format(count_result))

    wf.close()

def main(pto_file, abs_file, out_file):
    pto_dic = read_pto(pto_file)
    ta_list = abs_read(abs_file)
    save_result(ta_list, pto_dic, out_file)


if __name__ == '__main__':
    pto_file = '../data/to-basic.obo'
    abs_file = '../data/reference.table.txt'
    out_file = '../data/match_result.txt'
    print('-'*50)
    print('running.')
    main(pto_file, abs_file, out_file)
    print('-'*50)