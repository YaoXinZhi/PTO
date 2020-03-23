# -*- coding:utf-8 -*-
#! usr/bin/env python3
"""
Created on 19/03/2020 下午7:39 
@Author: xinzhi 
"""

import os

if __name__ == '__main__':
    source_path = '../data/pubtator_split'

    file_list = os.listdir(source_path)
    for file in file_list:
        commend_line = "nohup python3 get_corpus.py -sp {0} -pf {1} &".format(source_path, file)
        print(commend_line)
        # os.system(commend_line)
