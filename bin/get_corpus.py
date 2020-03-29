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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
import argparse
import time

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

def abstract_process(pubtator_file: str, abstract_file: str, ner_abstract_file: str, temp_file: str, high_voc: set):
    # todo: corpus 句子长度限制
    punctuation = string.punctuation.replace('-', '')
    wf = open(abstract_file, 'w')
    wf_ner = open(ner_abstract_file, 'w')
    wf_tem = open(temp_file, 'w')
    ann_list= []
    count = 0
    start_time = time.time()
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
                        sent_list = str_norm(sent_list, punc2=' ', num2='NBR')
                        ner_sent = []
                else:
                    l = line.strip().split('\t')
                    ann_list.append(l[3].lower())
            else:
                # ner 连接
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
                # 存文件
                for sent in sent_list:
                    wf.write('{0}\n'.format(sent))
                for sent in ner_sent:
                    wf_ner.write('{0}\n'.format(sent))
                for sent in sent_list:
                    word_list = word_tokenize(sent)
                    for voc in high_voc:
                        if voc in word_list:
                            wf_tem.write('{0}\n'.format(sent))
                            break
                ann_list = []
    end_time = time.time()
    print('共处理了{0}个摘要, 花费 {1:.2f} s.'.format(count, end_time - start_time))
    wf.close()
    wf_ner.close()
    wf_tem.close()

def pto_process(pto_file: str, fre=4):
    pto_set = set()
    stopwords_list = stopwords.words('english')
    with open(pto_file) as f:
        for line in f:
            l = line.strip()
            if l.startswith('name'):
                name = ' '.join(l.split(' ')[ 1: ])
                if '(' in name:
                    name = name[:name.find('(') - 1 ]
                pto_set.add(name.lower())

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
    # 导入 Stanfordcorenlp 模型
    en_model = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05', lang='en')

    count = 0
    # 定义正则模板
    pattern_np = re.compile(r'[(](NP.*)[)]')
    pattern_voc = re.compile(r'[(](.*?)[)]', re.S)
    start_time = time.time()
    with open(abstract_file) as f:
        for line in f:
            count += 1
            if count % 3000 == 0:
                print('{0} sentences process done'.format(count))
            try:
                l = line.strip()
                sent = l
                flag = False
                # 句法树分析
                parsing_result = en_model.parse(sent)
                # np所有子树
                np_list = pattern_np.findall(parsing_result)

                if len(np_list) == 0:
                    continue
                for np in np_list:
                    # 取出子树中所有单词
                    voc_list = pattern_voc.findall(np)
                    if len(voc_list) < 2:
                        continue
                    for index, voc in enumerate(voc_list):
                        voc_list[index] = voc.split(' ')[1]
                    # 判断是否有单词在高频词中
                    for voc in voc_list:
                        if voc in high_voc:
                            flag = True
                            row_str = [' '.join(voc_list)]
                            rep_str = str_norm(row_str, punc2='_', num2='NBR', space2='_')
                            for index in range(len(row_str)):
                                sent = sent.replace(row_str[index], rep_str[index])

                if flag:
                    wf.write('{0}\n'.format(sent))
                # end_time = time.time()
                # print('解析一个句子, 花费 {0:.2f} s.'.format(end_time - start_time))
            except:
                err_wf.write('{0}\n'.format(sent))
                continue
    end_time = time.time()
    print('共处理了{0}个句子, 花费 {1:.2f} s.'.format(count, end_time - start_time))
    en_model.close()
    wf.close()
    err_wf.close()

if __name__ == '__main__':

    """
    pubtator file: 29 G
    ftp://ftp.ncbi.nlm.nih.gov/pub/lu/
    
    服务器配置
    https://www.jianshu.com/p/14af5b4e221b
    """

    # # 命令行参数
    # parser = argparse.ArgumentParser(description='get_corpus.')
    # parser.add_argument('-sp', dest='source_path', type=str, default='../data/pubtator_split', help='default: data')
    # parser.add_argument('-pf', dest='pubtator_file', type=str, required=True)
    # parser.add_argument('-ac', dest='abs_corpus', default='../data/abs_corpus', help='default: ../data/abs_corpus')
    # parser.add_argument('-nc', dest='ner_corpus', default='../data/ner_corpus', help='default: ../data/ner_corpus')
    # parser.add_argument('-pc', dest='pto_corpus', default='../data/pto_corpus', help='default: ../data/pto_corpus')
    # parser.add_argument('-tp', dest='temp_path', default='../data/temp_path', help='default: ../data/temp_path')
    # args = parser.parse_args()

    # 调试
    class config():
        def __init__(self):
            self.source_path = './data/pubtator_split'
            self.abs_corpus = './data/abs_corpus'
            self.ner_corpus = './data/ner_corpus'
            self.pto_corpus = './data/pto_corpus'
            self.pubtator_file = 'pub_1'
            self.temp_path = './data/temp_path'
    args = config()

    if not os.path.exists(args.abs_corpus):
        os.mkdir(args.abs_corpus)

    if not os.path.exists(args.ner_corpus):
        os.mkdir(args.ner_corpus)

    if not os.path.exists(args.pto_corpus):
        os.mkdir(args.pto_corpus)

    if not os.path.exists(args.temp_path):
        os.mkdir(args.temp_path)

    pto_file = './data/to-basic.obo'

    pubtator_file = os.path.join(args.source_path, args.pubtator_file)
    abstract_file = os.path.join(args.abs_corpus, args.pubtator_file)
    ner_abstract_file = os.path.join(args.ner_corpus, args.pubtator_file)
    pto_abstract_file = os.path.join(args.pto_corpus, args.pubtator_file)
    temp_file = os.path.join(args.temp_path, args.pubtator_file)

    err_log = './data/err.txt'

    pto_set, high_voc = pto_process(pto_file, fre=4)
    print('-'*50)
    print('abstract processing.')
    abstract_process(pubtator_file, abstract_file, ner_abstract_file, temp_file, high_voc)
    print('-'*50)
    print('pto phrase combination.')
    pto_phrase(temp_file, high_voc, pto_abstract_file, err_log)
