# -*- coding:utf-8 -*-
#! usr/bin/env python3
"""
Created on 18/03/2020 下午1:05
@Author: xinzhi
"""

"""
Pubmed 摘要预处理:
    所有数字替换为"NBR"
    所有标点替换为 空格
PTO 数据库预处理:
    提取高频词 fre>4
整合语料库:
    短语组合:
        stanford parser 句法树解析
        包含高频词的最小子树下划线连接
    实体组合:
        PubTator注释
"""

import os
import re
import string
from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
import argparse


def str_norm(str_list: list, punc2=' ', num2='NBR', space2=' '):
    punctuation = string.punctuation.replace('-', '')
    rep_list = str_list.copy()
    for index, row in enumerate(rep_list):
        row = row.strip()
        row = re.sub("\d+.\d+", num2, row)
        row = re.sub('\d+', num2, row)
        for pun in punctuation:
            row = row.replace(pun, punc2)
        rep_list[index] = re.sub(' +', space2, row)
    return rep_list


def abstract_process(pubtator_file: str, abstract_file: str, ner_abstract_file: str):
    # todo: corpus 句子长度限制
    punctuation = string.punctuation.replace('-', '')
    wf = open(abstract_file, 'w')
    wf_ner = open(ner_abstract_file, 'w')
    ann_list= []
    count = 0
    with open(pubtator_file) as f:
        for line in tqdm(f):
            if count % 300000 == 0:
                print('{0} abstracts process done'.format(count))

            l = line.strip()
            if l != '':
                l = l.split('|')
                if len(l) > 2:
                    if l[1] == 't':
                        title_len = len(l[2])
                        count += 1
                    elif l[1] == 'a':
                        abstract = l[2].lower()
                        sent_list = sent_tokenize(abstract)
                        # sent_list = re.split(r'\?|\!|\.', abstract)
                        sent_list = str_norm(sent_list, punc2=' ', num2='NBR')
                        ner_sent = []
                else:
                    l = line.strip().split('\t')
                    ann_list.append(l[3].lower())
            else:
                norm_ann = str_norm(ann_list, punc2=' ', num2='NBR')
                norm_rep = str_norm(ann_list, punc2='_', num2='NBR', space2='_')
                for sent in sent_list:
                    flag = False
                    sent_copy = sent
                    for i in range(len(norm_ann)):
                        if len(norm_ann[i].split(' ')) < 2:
                            continue
                        if norm_ann[i] in sent_copy:
                            flag = True
                            sent_copy = sent_copy.replace(norm_ann[i], norm_rep[i])
                    if flag:
                        ner_sent.append(sent_copy)
                for sent in sent_list:
                    wf.write('{0}\n'.format(sent))
                for sent in ner_sent:
                    wf_ner.write('{0}\n'.format(sent))
                ann_list = []
    wf.close()
    wf_ner.close()
    return abstract, ann_list

def pto_process(pto_file: str, fre=4):
    pto_set = set()
    stopwords_list = stopwords.words('english')
    with open(pto_file) as f:
        for line in f:
            l = line.strip()
            if l.startswith('name'):
                name = ' '.join(l.split(' ')[ 1: ])
                if name.find('('):
                    name = name[ : name.find('(') - 1 ]
                pto_set.add(name)

            if l.startswith('synonym'):
                pattern = re.compile('"(.*)"')
                synonym = pattern.findall(l)[ 0 ]

                pattern_ba = re.compile(r'[(].*?[)]')
                backets = pattern_ba.findall(l)
                if synonym.startswith('('):
                    for ba in backets[ 1: ]:
                        synonym = synonym.replace(ba, '')
                else:
                    for ba in backets:
                        synonym = synonym.replace(ba, '')
                if ':' in synonym:
                    synonym = synonym.split(':')[ 0 ]
                pto_set.add(synonym.lower())
    print('共出现: {0} 个术语.'.format(len(pto_set)))
    # High frequency vocabulary
    voc_count = defaultdict(int)
    for pto in pto_set:
        for voc in pto.split(' '):
            voc_count[voc] += 1
    high_voc = set([k for k, v in voc_count.items() if v > fre and k not in stopwords_list and len(k) > 1])
    print('包含 {0} 个高频单词.'.format(len(high_voc)))
    return pto_set, high_voc

def pto_phrase(abstract_file: str, high_voc: set, abstract_pto_file: str, err_log: str):
    wf = open(abstract_pto_file, 'w')
    err_wf = open(err_log, 'w')
    en_model = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05', lang='en')
    count = 0
    with open(abstract_file) as f:
        for line in f:
            count += 1
            if count % 3000 == 0:
                # todo: 加上计时
                print('{0} sentences process done'.format(count))
            try:
                l = line.strip()
                sent_list = sent_tokenize(l)
                # sent_list = re.split(r'\?|\!|\.', l)
                for sent in sent_list:
                    flag = False
                    rep_list = []

                    parsing_result = en_model.parse(sent)
                    pattern_np = re.compile(r'[(](NP.*)[)]')
                    np_list = pattern_np.findall(parsing_result)
                    if len(np_list) == 0:
                        continue
                    pattern_voc = re.compile(r'[(](.*?)[)]', re.S)
                    for np in np_list:
                        voc_list = pattern_voc.findall(np)
                        if len(voc_list) < 2:
                            continue
                        for index, voc in enumerate(voc_list):
                            voc_list[index] = voc.split(' ')[1]
                        for voc in voc_list:
                            if voc in high_voc:
                                flag = True
                                row_str = [' '.join(voc_list)]
                                rep_str = str_norm(row_str, punc2='_', num2='NBR', space2='_')
                                for index in range(len(row_str)):
                                    sent = sent.replace(row_str[index], rep_str[index])
                    if flag:
                        wf.write('{0}\n'.format(sent))
            except:
                err_wf.write('{0}\n'.format(sent))
                continue
    en_model.close()
    wf.close()
    err_wf.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='get_corpus.')
    parser.add_argument('-sp', dest='source_path', type=str, default='../data/pubtator_split', help='default: data')
    parser.add_argument('-pf', dest='pubtator_file', type=str, required=True)
    parser.add_argument('-ac', dest='abs_corpus', default='data/abs_corpus', help='default: ../data/abs_corpus')
    parser.add_argument('-nc', dest='ner_corpus', default='data/ner_corpus', help='default: ../data/ner_corpus')
    parser.add_argument('-pc', dest='pto_corpus', default='data/pto_corpus', help='default: ../data/pto_corpus')
    args = parser.parse_args()

    if not os.path.exists(args.abs_corpus):
        os.mkdir(args.abs_corpus)

    if not os.path.exists(args.ner_corpus):
        os.mkdir(args.ner_corpus)

    if not os.path.exists(args.pto_corpus):
        os.mkdir(args.pto_corpus)

    pto_file = '../data/to-basic.obo'

    pubtator_file = os.path.join(args.source_path, args.pubtator_file)
    abstract_file = os.path.join(args.abs_corpus, args.pubtator_file)
    ner_abstract_file = os.path.join(args.ner_corpus, args.pubtator_file)
    pto_abstract_file = os.path.join(args.pto_corpus, args.pubtator_file)
    corpus_file = 'data/corpus.txt'

    err_log = 'data/err.txt'

    pro_set, high_voc = pto_process(pto_file, fre=4)
    print('-'*50)
    print('abstract processing.')
    abstract, span_list = abstract_process(pubtator_file, abstract_file, ner_abstract_file)
    print('-'*50)
    print('pto phrase combination.')
    pto_phrase(abstract_file, high_voc, pto_abstract_file, err_log)
