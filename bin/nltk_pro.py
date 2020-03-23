# -*- coding:utf-8 -*-
#! usr/bin/env python3
"""
Created on 11/03/2020 下午8:26 
@Author: xinzhi 
"""

import re
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from nltk.tokenize import word_tokenize
import nltk


class PTO(object):
    def __init__(self):
        self.id = ''
        self.name = ''
        self.snnonym = []
        self.is_a = ''
        self.part_of = ''

def read_pto(pto_file):
    opt_dic = defaultdict(list)
    with open(pto_file) as f:
        for line in f:
            l = line.strip()
            if l.startswith('id'):
                id = l.split()[1]
            if l.startswith('name'):
                name = ' '.join(l.split(' ')[1:])
                opt_dic[id].append(name)
            if l.startswith('synonym'):
                pattern = re.compile('"(.*)"')
                synonym = pattern.findall(l)
                synonym = ' '.join(synonym[0].split(' ')[:-1])
                opt_dic[id].append(synonym)
    return opt_dic

def abs_read(abstract_file: str):
    title_abstract_list = []
    with open(abstract_file) as f:
        line = f.readline()
        for line in f:
            l = line.strip().split('\t')
            try:
                title = l[0]
                abs = l[4]
                title_abstract_list.append((title, abs))
            except:
                print(l)
                continue
    return title_abstract_list


def save_result(ta_list: list, pto_dic: dict, out_file: str):
    wf = open(out_file, 'w')
    count = 0
    wf.write('PTO\tsentence\ttitle\tterm\n')
    for ta in ta_list:
        title = ta[0]
        abs = ta[1]
        sent_list = [i.lower() for i in sent_tokenize(abs)]
        # sent_list = [i for i in sent_tokenize(abs)]
        for term, name_list in pto_dic.items():
            for name in name_list:
                for sent in sent_list:
                    # word_tokens = word_tokenize(sent)
                    if name.lower() in sent:
                    # if name in sent:
                        wf.write('{0}\t{1}\t{2}\t{3}\n'.\
                                 format(term, sent, title, '|'.join(name_list)))
    wf.close()

def main(pto_file, abs_file, out_file):
    pto_dic = read_pto(pto_file)
    ta_list = abs_read(abs_file)
    save_result(ta_list, pto_dic, out_file)



if __name__ == '__main__':
    pto_file = 'to-basic.obo'
    abs_file = 'reference.table.txt'
    out_file = 'match_result.txt'
    main(pto_file, abs_file, out_file)
