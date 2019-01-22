#!/bin/env python
#-*- encoding: utf-8 -*-


import sys
sys.path.append('../')

import os
import codecs
from tools import config
from tools.nlp_util import NLPUtil


gemii_nli_path = config.gemii_nli_path

for file_ in os.listdir(gemii_nli_path):
    if file_.startswith('labels'): 
        continue
    file_ = os.path.join(gemii_nli_path, file_)
    with codecs.open(file_, 'r', 'utf-8') as rfd, \
        codecs.open(file_ + '.tokenized', 'w', 'utf-8') as wfd:
        tokenize_func = NLPUtil.tokenize_via_jieba
        data = map(tokenize_func, rfd.readlines())
        for item in data:
            wfd.write(' '.join(item) + '\n')

