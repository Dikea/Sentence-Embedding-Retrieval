#!/bin/env python
#-*- encoding: utf-8 -*-


import codecs
from tools import config

class KnowledgeTree(object):

	"""
	Knowledge tree, like a multi-tree, the none-leaf node 
	represent tagid, k_id or kq_id and leaf node represent 
	information of question, like {kq_id, k_question}
	"""

	def __init__(self, knowledge_fpath):
		self.knowledge_tree = {}
		self.knowledge_fpath = knowledge_fpath


	def build(self):
		knowledge_corpus = self._read_data(self.knowledge_fpath)
		for info in knowledge_corpus:
			self._add_node(info)

	
	def get_node(self, id_path):
		id_list = id_path.split(',')	
		tree = self.knowledge_tree
		for idx in id_list:
			tree = tree[idx]
		return tree


	def _read_data(self, file_path):
		"""
		Read knowledge data, columns: `kq_id, k_question, 
		k_id, k_title, k_answer, status, ktag_id`
		"""
		with codecs.open(file_path, 'r', 'utf-8') as in_f:
			in_f.readline()
			corpus = [line.strip('\n').split('|') 
					  for line in in_f.readlines()] 
		return corpus


	def _add_node(self, info):
		"""
		Add node to knowledge tree
		"""
		try:
			kq_id, k_question, k_id, k_title, k_answer, status, ktag_id = info
			node = k_question
			id_list = ktag_id.split(',') + ['#' + k_id, kq_id]
			tree = self.knowledge_tree
			for pos, idx in enumerate(id_list):
				if idx not in tree:
					rid_list = id_list[pos+1:][::-1]
					for ridx in rid_list:
						node = {ridx: node}
					tree[idx] = node
					break
				else:
					tree = tree[idx]
		except Exception as e:
			print '[ERROR] info=%s, errmsg=%s' % (info, e)


def test():
	knowledge_fpath = config.gemii_knowledge_fpath
	knowledge_tree = KnowledgeTree(knowledge_fpath)
	knowledge_tree.build()
	print len(knowledge_tree.knowledge_tree)
	import json
	print json.dumps(knowledge_tree.get_node('401,624,1361,#23'))


if __name__ == '__main__':
	test()
