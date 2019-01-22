#!/bin/env python
#-*- encoding: utf-8 -*-


import os
import codecs


## Model file
embeddings_size = 300
word2vec_path = '../sentence_embeddings_model/dataset/word2vec'
word2vec_path = os.path.join(word2vec_path, 'v1.w2v_sgns_win1_d%d.kv' % embeddings_size)
temp_word2vec_path = '../sentence_embeddings_model/dataset/word2vec/80/w2v.model'
paras_fpath = './data/paras.pkl'
saved_model_path = '../sentence_embeddings_model/serving_models/1'
#saved_model_path = '../sentence_embeddings_model/best_serving_models/Gemii-1-6'
sentence_emb_path = '../sentence_embeddings_preprocess/data/sent_embeddings' 


## Nli data file
gemii_knowledge_fpath = '../sentence_embeddings_preprocess/data/gemii_knowledge.txt'
gemii_nli_path = '../sentence_embeddings_preprocess/data/gemii_nli'


## User define words
_common_units = ('cc,cm,cr,db,h,hz,k,kcal,kg,kj,l,m,mah,mg,'
    'min,ml,mm,mmhg,mmol,nm,ohp,ppm,ug,year'.split(','))
g_ud_words_cfg = ['float_t', 'phone_t', 'email_t', 'int_t']
g_ud_words_cfg.extend(_common_units)

ud_kw_fpath = '../tools/conf/gemii_ud.keywords'
if os.path.exists(ud_kw_fpath):
    with codecs.open(ud_kw_fpath, 'r', 'utf-8') as rfd:
        ud_words = rfd.read().splitlines()
        g_ud_words_cfg.extend(ud_words)
else:
    print('WARNING: ud file not found, path=%s' % (ud_kw_fpath))


## Stop words
g_stop_words_cfg = set(['\n'])
# add common stop words from file
# fetched from https://github.com/JNU-MINT/TextBayesClassifier
sw_fpath = '../tools/conf/stopword_chs.txt'
if os.path.exists(sw_fpath):
    with codecs.open(sw_fpath, 'r', 'utf-8') as rfd:
        words = rfd.read().splitlines()
        g_stop_words_cfg.update(words)
if 0 == len(g_stop_words_cfg):
    print('WARNING: stop_words_cfg is empty, make sure this is what you expect!')


## WMD sorting
wmd_cfg = {
    'wmd_flag': False,
    'wmd_number': 30,
}


## Search para
# If query length large than limit, it will be clipped
max_sentence_length = 70 
