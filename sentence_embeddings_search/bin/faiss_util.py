#!/bin/env python
#-*- encoding: utf-8 -*-


import os
import sys
#sys.path.append('../')
import codecs
from collections import defaultdict
import numpy as np
import faiss
from tools.infersent_encoder import Encoder
from tools import config
from tools.log import g_log_inst as logger


class FaissUtil(object):
    
    
    _encoder = None
    _id2info_dict = {} # index id to info
    _qid2id_dict = {} # question id to index id
    _kid2ids_dict = defaultdict(set) # knowledge id to id-set of questions 
    _index_count = None 
    _sentence_fpath = config.gemii_knowledge_fpath
    _sentence_emb_fpath = os.path.join(config.sentence_emb_path, 
                                       'sentence.embeddings.npy')


    @classmethod
    def init(cls):
        myself = sys._getframe().f_code.co_name
        try:
            cls._encoder = Encoder()
            cls._build_faiss_index(cls._sentence_fpath, cls._sentence_emb_fpath)
        except Exception as e:
            logger.get().warn('%s failed, fpath=%s, errmsg=%s',
                myself, cls._sentence_emb_fpath, e)

    
    @classmethod
    def search_similar_items(cls, sentence, topk = 10):
        myself = sys._getframe().f_code.co_name
        try:
            s_embedding = cls._encoder.encode([sentence])
            l2_distance, sim_ids = cls._faiss_index.search(s_embedding, topk)
            sim_items = zip(sim_ids.tolist()[0], l2_distance.tolist()[0])
            id2sentence = cls._id2info_dict
            ret_items = []
            for item in sim_items:
                idx = item[0]
                info = id2sentence[idx]
                info['l2distance'] = '{0:.6f}'.format(item[1])
                ret_items.append(info)  
            logger.get().debug('%s success, sentence=%s', myself, '|'.join(sentence))
            return ret_items
        except Exception as e:
            logger.get().warn('%s failed, sentence=%s, topk=%s, errmsg=%s', 
                myself, '|'.join(sentence), topk, e)
            return None


    @classmethod
    def update_faiss_index(cls, paras):
        myself = sys._getframe().f_code.co_name
        try:
            update_func = {
                # Knowledge
                'addK': 'cls._add_knowledge', 
                'upK': 'cls._update_knowledge',
                'delK': 'cls._delete_knowledge',
                # Question
                'addQ': 'cls._add_question', 
                'upQ': 'cls._update_question',
                'delQ': 'cls._delete_question'}
            return True
        except Exception as e:
            logger.get().warn('%s failed, paras=%s, errmsg=%s', myself, paras, e)
            return False


    @classmethod
    def _add_knowledge(cls, paras):
        data = paras['addK']
        k_id, k_title = data['k_id'], data['k_title']
        s_emb = cls._encoder.encode([k_title])
        cls._index_count += 1
        cls._faiss_index.add_with_ids(s_emb, np.array([cls._index_count]))
        info = {
            'k_id': k_id, 
            'k_title': k_title, 
            'kq_id': '#' + k_id,
            'k_question': k_title}
        cls._id2info_dict[cls._index_count] = info 


    @classmethod
    def _update_knowledge(cls, paras):
        pass


    @classmethod
    def _delete_knowledge(cls, paras):
        pass


    @classmethod
    def _add_question(cls, paras):
        pass


    @classmethod
    def _update_question(cls, paras):
        pass


    @classmethod
    def _delete_question(cls, paras):
        pass


    @classmethod
    def _build_faiss_index(cls, sentence_fpath, sentence_emb_fpath):
        myself = sys._getframe().f_code.co_name
        try:
            s_embeddings = np.load(sentence_emb_fpath) 
            with codecs.open(sentence_fpath + '.expand', 'r', 'utf-8') as rfd:
                kq_id_list = []
                for idx, line in enumerate(rfd.readlines()[1:]):
                    kq_id, k_question, k_id, k_title, k_answer, status, ktag_id = (
                        line.rstrip().split('|'))
                    info = {
                        'kq_id': kq_id,
                        'k_question': k_question,
                        'k_id': k_id,
                        'k_title': k_title}
                    cls._id2info_dict[idx] = info
                    cls._kid2ids_dict[k_id].add(idx)
                    cls._qid2id_dict[kq_id] = idx
            cls._index_count = idx 
            dim_size = s_embeddings.shape[1] 
            index = faiss.index_factory(dim_size, 'IDMap,Flat')
            index.add_with_ids(s_embeddings, np.arange(idx + 1))
            cls._faiss_index = index
            logger.get().debug('%s success, index_size=%s',
                myself, index.ntotal)
            return True
        except Exception as e:
            logger.get().warn('%s failed, errmsg=%s', myself, e)
            raise
            return False


if __name__ == '__main__':
    logger.start('./log/faiss.log', name = __name__, level = 'DEBUG')
    FaissUtil.init()
