#!/bin/env python
#-*- encoding: utf-8 -*-

import codecs
import tensorflow as tf
import config 
from nlp_util import NLPUtil
from data_helper import Helper
from log import g_log_inst as logger


class Encoder(object):
	
	def __init__(self):
		try:
			self.sess = tf.Session()
			meta_graph_def = tf.saved_model.loader.load(self.sess, 
				['infersent_model'], config.saved_model_path)
			signature = meta_graph_def.signature_def
			signature_def_key = 'encoder'
			s_embedded_name = signature[signature_def_key].inputs['s_embedded'].name
			s_lengths_name = signature[signature_def_key].inputs['s_lengths'].name
			s_embeddings_name = signature[signature_def_key].outputs['s_embeddings'].name
			self.s_embedded = self.sess.graph.get_tensor_by_name(s_embedded_name)
			self.s_lengths = self.sess.graph.get_tensor_by_name(s_lengths_name)
			self.s_embeddings = self.sess.graph.get_tensor_by_name(s_embeddings_name)
			logger.get().info('init sentence encoder success')
		except Exception as e:
			logger.get().warn('init sentence encoder failed, errmsg=%s', e)

	
	def encode(self, seq_list):
		try:
			_s_embedded, _s_lengths = Helper.get_batch(seq_list)		
			feed_dict = {
				self.s_embedded: _s_embedded,
				self.s_lengths: _s_lengths}
			s_embeddings = self.sess.run(self.s_embeddings, feed_dict = feed_dict)
			return s_embeddings
		except Exception as e:
			logger.get().debug('seq_length=%s, errmsg=%s', len(seq_list), e)
	

def test():
	Helper.init()
	with codecs.open('./data/test.txt', 'r', 'utf-8') as in_f:
		corpus = [line.strip('\n') for line in in_f.readlines()]
	corpus = [['<s>'] + [word for word in NLPUtil.tokenize_via_jieba(sent)
		if word in Helper._word2vec] + ['</s>'] for sent in corpus]
	s_encoder = Encoder()	
	s_embeddings = s_encoder.encode(corpus)
	print s_embeddings.shape
	print s_embeddings.dtype
	print s_embeddings[0]


if __name__ == '__main__':
	logger.start('./log/encode.log', name = __name__, level = 'DEBUG')
	test()
