# -*- coding:utf-8 -*-
#! usr/bin/env python3

"""
Created on 21/03/2020 下午1:48 
@Author: xinzhi 
"""

import gensim
import argparse
import re
import os
from nltk.corpus import stopwords
import string
import numpy as np
from collections import defaultdict

def str_norm(str_list: list, punc2=' ', num2='NBR', space2=' '):
    punctuation = string.punctuation.replace('-', '')
    rep_list = str_list.copy()
    for index, row in enumerate(rep_list):
        row = row.strip()
        row = re.sub("\d+.\d+", num2, row)
        row = re.sub('\d+', num2, row)
        for pun in punctuation:
            row = row.replace(pun, punc2)
        # rep_list[index] = row.replace(' ', space2)
        rep_list[index] = re.sub(' +', space2, row)
    return rep_list

def load_pto(pto_file: str, norm=False):
    pto_set = set()
    stopwords_list = stopwords.words('english')
    with open(pto_file) as f:
        for line in f:
            l = line.strip()
            if l.startswith('name'):
                name = ' '.join(l.split(' ')[1:])
                if name.find('('):
                    name = name[: name.find('(')-1]
                pto_set.add(name)

            if l.startswith('synonym'):
                pattern = re.compile('"(.*)"')
                synonym = pattern.findall(l)[0]
                pattern_ba = re.compile(r'[(].*?[)]')
                backets = pattern_ba.findall(l)
                if synonym.startswith('('):
                    for ba in backets[1:]:
                        synonym = synonym.replace(ba, '')
                else:
                    for ba in backets:
                        synonym = synonym.replace(ba, '')
                if ':' in synonym:
                    synonym = synonym.split(':')[0]

                pto_set.add(synonym.lower())
    print('共出现: {0} 个术语.'.format(len(pto_set)))

    if norm:
        pto_set = set(str_norm(list(pto_set), punc2='_', space2='_'))

    return pto_set

def CosSim_top(vector_file: str, pto_set: set, out: str, topn=10):
    # todo: pto row name.
    emb = gensim.models.KeyedVectors.load_word2vec_format(vector_file, binary=False)
    vocab = emb.vocab
    wf = open(out, 'w')
    for pto in pto_set:
        if pto in vocab:
            res = emb.most_similar(emb[pto], topn=topn)
        wf.write('{0}\n'.format(pto))
        for voc in res:
            wf.write('{0}\t'.format(voc))
        wf.write('\n')
    wf.close()
    return emb

def CosSim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def ACosSim_top(vector_file: str, pto_set: set, out: str, topn=10):
    emb = gensim.models.KeyedVectors.load_word2vec_format(vector_file, binary=False)
    #vector = emb.vectors_norm l2 norm
    vocab = emb.vocab
    word2index = {word: index for index, word in enumerate(emb.index2word)}
    index2word = {index: word for index, word in enumerate(emb.index2word)}
    vector = emb.vectors
    mean = np.mean(vector, axis=0)
    vector_norm = vector / mean
    wf = open(out, 'w')
    sim_dic = defaultdict(float)
    count = 0
    for pto in pto_set:
        if pto in vocab:
            wf.write('{0}\n'.format(pto))
            count += 1
            for i in range(vector.shape[0]):
                if i != word2index[pto]:
                    cossim = CosSim(vector_norm[word2index[pto]], vector_norm[i])
                    sim_dic[index2word[i]] = cossim
            # print(sim_dic)
            sim_sort = sorted(sim_dic, key=lambda x: float(sim_dic[x]), reverse=True)
            print(sim_sort)
            for j in sim_sort[:topn]:
                wf.write('({0}, {1})\t'.format(j, sim_dic[j]))
            wf.write('\n')
    print('一共有 {0} 个pto得到候选词.'.format(count))
    wf.close()

if __name__ == '__main__':

    vector_file = '../data/embedding/pto_embedding.txt'
    pto_file = '../data/to-basic.obo'
    out = '../data/embedding/Sim.out'
    pto_set = load_pto(pto_file, True)
    print('-'*50)
    # pto_set=set(['of','and'])
    print('geting most simliar of pto.')
    # CosSim_top(vector_file, pto_set, out='data/embedding/Sim.out', topn=10)
    ACosSim_top(vector_file, pto_set, out=out, topn=10)
    print('-'*50)


