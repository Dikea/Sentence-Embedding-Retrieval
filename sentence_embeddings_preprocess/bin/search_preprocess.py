#!/bin/env python
#-*- encoding: utf-8 -*-


import sys
sys.path.append('../')

import os
import time
import numpy as np
import cPickle as pickle
import codecs
from tools import config
from tools.log import g_log_inst as logger
from tools.nlp_util import NLPUtil
from tools.data_helper import Helper
from tools.infersent_encoder import Encoder


EXPAND_FLAG = False


def encode_sentences(encoder, sentence_corpus, save_path):
	s_embeddings = None
	batch_size = 64 
	s_len = len(sentence_corpus)
	for idx in xrange(0, s_len, batch_size):
		try:
			sents = sentence_corpus[idx: idx + batch_size]
			sents = [['<s>'] + [word for word in NLPUtil.tokenize_via_jieba(sent) 
				if word in Helper._word2vec] + ['</s>'] for sent in sents]
			s_embs = encoder.encode(sents)
			if s_embeddings is None:
				s_embeddings = s_embs
			else: 
				s_embeddings = np.vstack((s_embeddings, s_embs))
			if idx % (64 * 2) == 0:
				logger.get().debug('have encoded %s sentences', idx)
		except Exception as e:
			logger.get().warn('idx=%s, errmsg=%s', idx, e)
	
	# Save encode results
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	np.save(os.path.join(save_path, 'sentence.embeddings'), s_embeddings)


def main():
	logger.start('./log/preprocess.log', name = __name__, level = 'DEBUG')
	knowledge_fpath = config.gemii_knowledge_fpath
	sentence_embeddings_fpath = config.sentence_emb_path

	# Read knowledge library
	with codecs.open(knowledge_fpath, 'r', 'utf-8') as in_f:
		headers = in_f.readline()
		corpus = [line.rstrip().split('|') for line in in_f]

	# Expand corpus by using knowledge titles as questions
	last_kid = None
	k_info_set = set()
	for item in corpus:
		kq_id, k_question, k_id, k_title, k_answer, status, ktag_id = item
		if k_id is not last_kid:
			new_item = ['#' + k_id, k_title, k_id, k_title, 
						k_answer, status, ktag_id]
			k_info_set.add('|'.join(new_item))
			last_kid = k_id
	if EXPAND_FLAG: 
		for item in k_info_set:
			corpus.append(item.split('|'))
	with codecs.open(knowledge_fpath + '.expand', 'w', 'utf-8') as wfd:
		wfd.write(headers)
		for item in corpus:
			wfd.write('|'.join(item) + '\n')

	# Encode sentences
	Helper.init()
	encoder = Encoder()
	time_s = time.time()
	corpus = [item[1] for item in corpus]
	encode_sentences(encoder, corpus, sentence_embeddings_fpath)	
	print 'Encode finished, time elapsed: %.4f seconds' % (
		time.time() - time_s)


if __name__ == '__main__':
	main()
