#!/bin/env python
#-*- encoding: utf-8 -*-

import sys
import threading
import pickle as pkl
from grpc.beta import implementations
import numpy
import tensorflow as tf
from datetime import datetime 
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from data_helper import build_input

tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS



class Client(object):
	
	@classmethod
	def do_inference(seq_list):
		# Connect to server
		host = 'localhost'
		port = '9000'
		channel = implementations.insecure_channel(host, int(port))
		stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
		
		# Prepare request object
		request = predict_pb2.PredictRequest()
		request.model_spec.name = 'default'
		request.model_spec.signature_name = 'prediction'

		# Process input sentence list
		cls._build_input_tensor(seq_list, request)
		res = stub.Predict(request, 0.2)
		return res

	
	@classmethod
	def _build_input_tensor(cls,seq_list, request):
		try:
			s_embeded, s_lengths = build_vec(seq_list)
			request.inputs['s_embedded'].CopyFrom(
				tf.contrib.util.make_tensor_proto(s_embeded, shape=data.shape))
			request.inputs['s_lengths'].CopyFrom(
				tf.contrib.util.make_tensor_proto(s_lengths, shape=data.shape))
			return True
		except Exception as e:
			return False


def test():
	pass


if __name__ == "__main__":
	test()	
