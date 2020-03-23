# -*- coding:utf-8 -*-
#! usr/bin/env python3
"""
Created on 19/03/2020 下午3:55 
@Author: xinzhi 
"""

import os

def spliter(pubtator_file: str, out_path: str, split_size=1000000, fix='pub'):
    fix = 'pub'
    idx = 1

    out = '{0}_{1}'.format(fix, idx)
    wf = open(os.path.join(out_path, out), 'w')
    count = 0
    with open(pubtator_file) as f:
        for line in f:
            l = line.strip().split('|')

            if len(l) > 1:
                if l[1] == 't':
                    count += 1
                    if count % split_size == 0:
                        wf.close()
                        print('{0} save done.'.format(out))
                        idx += 1
                        out = '{0}_{1}'.format(fix, idx)
                        wf = open(os.path.join(out_path, out), 'w')

            wf.write(line)
        wf.close()


if __name__ == '__main__':

    pubtator_source = '../data/bioconcepts2pubtator_offsets'
    out_path = '../data/pubtator_split'

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    spliter(pubtator_source, out_path, split_size=1000000, fix='pub')

