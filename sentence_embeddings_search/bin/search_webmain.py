#!/bin/env python
#-*- encoding: utf-8 -*-

import sys
sys.path.append('../')

import os
import signal
import time
import json
from tornado import httpserver
from tornado import ioloop
from tornado import web
import search_core
from tools.log import g_log_inst as logger


MAX_WAIT_SECONDS_BEFORE_SHUTDOWN = 0.5


# Disable tensorflow GPU service
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class SimilaritySearchHandler(web.RequestHandler):
	def get(self):
		# Parse query params
		params = {}
		required_keys = ['query']
		optional_keys = {'size': 30,}
		for key in required_keys:
			value = self.get_query_argument(key)
			params[key] = value
		for k, v in optional_keys.items(): 
			value = self.get_query_argument(k, v)
			params[k] = value
		# Process request
		(status, rsp) = search_core.ApiHandler.search_similar_items(params)
		if 200 == status:
			self.set_header('content-type', 'application/json')
			self.finish(rsp)
		else:
			self.set_status(404)
			self.finish()


def signal_handler(sig, frame):
	logger.get().warn('Caught signal: %s', sig)
	ioloop.IOLoop.instance().add_callback_from_signal(shutdown)


def shutdown():
	logger.get().info('begin to stop http server ...')
	server.stop()

	logger.get().info('shutdown in %s seconds ...', MAX_WAIT_SECONDS_BEFORE_SHUTDOWN)
	io_loop = ioloop.IOLoop.instance()
	deadline = time.time() + MAX_WAIT_SECONDS_BEFORE_SHUTDOWN

	def stop_loop():
		now = time.time()
		if now < deadline and (io_loop._callbacks or io_loop._timeouts):
			io_loop.add_timeout(now + 1, stop_loop)
		else:
			io_loop.stop()
			logger.get().info('shutdown finished')
	stop_loop()


def main():
	try:
		log_path = './log/search.log'
		logger.start(log_path, name = __name__, level = 'DEBUG')

		if 2 != len(sys.argv):
			logger.get().warn('start search api failed, argv=%s' % (sys.argv))
			return 1
		port = int(sys.argv[1])

		if False == search_core.ApiHandler.init():
			logger.get().warn('init failed, quit now')
			return 1

		app_inst = web.Application([
			(r'/in/nlp/sentence/search', SimilaritySearchHandler),
		], compress_response = True)

		global server
		server = httpserver.HTTPServer(app_inst)
		server.listen(port)

		## install signal handler, for gracefully shutdown
		signal.signal(signal.SIGTERM, signal_handler)
		signal.signal(signal.SIGINT, signal_handler)

		logger.get().info('server start, port=%s' % (port))
		ioloop.IOLoop.instance().start()
	except KeyboardInterrupt, e:
		raise


if '__main__' == __name__:
	main()
