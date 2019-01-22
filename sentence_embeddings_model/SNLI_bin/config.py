#!/bin/env python
#-*- encoding: utf-8 -*-


# Path
GLOVE_PATH = './dataset/GloVe/glove.840B.300d.txt'
snli_path = './dataset/SNLI/'
model_path = './models/m0/'
log_path = './log/model.log'


# Model version for serving
serving_model_version = 1
serving_model_path = './serving_models/'


# Model paras
class Infersent_para(object):
	"""Build parameters"""
	embedding_size = 300
	hidden_size = 2048
	hidden_layers = 1
	batch_size = 64
	keep_prob_dropout = 1.0
	learning_rate = 0.1 
	bidirectional = True
	decay = 0.99
	lrshrink = 5
	eval_step = 500
	uniform_init_scale = 0.1
	clip_gradient_norm = 5.0
	save_every = 1000000
	epochs = 10

	# Settings for debug
	debug_mode = True
	if debug_mode:
		train_steps = 10
		dev_steps = 1 
		eval_step = 10 
		epochs = 2


