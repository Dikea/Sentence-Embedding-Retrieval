#!/bin/env python
#-*- encoding: utf-8 -*-


import time
import sys
import json
from collections import OrderedDict
from faiss_util import FaissUtil
from wmd_util import WmdUtil 
from tools.nlp_util import NLPUtil
from tools.data_helper import Helper
from tools.log import g_log_inst as logger
from tools import config


class ApiHandler(object):


    _encoder = None
    _wmd_cfg = config.wmd_cfg


    @classmethod
    def init(cls):
        myself = sys._getframe().f_code.co_name
        try:
            Helper.init()
            FaissUtil.init()    
            if cls._wmd_cfg['wmd_flag']:
                WmdUtil.init()
            return True
        except Exception as e:
            logger.get().warn('%s failed, errmsg=%s', myself, e)
            return False


    @classmethod
    def search_similar_items(cls, params):
        myself = sys._getframe().f_code.co_name
        log_kw = ', '.join(map(lambda (k, v): '%s=%s' % (k, v), params.items()))
        logger.get().debug('%s begin, %s' % (myself, log_kw))
        try:
            time_s = time.time()
            query = params['query'][:config.max_sentence_length]
            query_tokens =['<s>'] + [w for w in NLPUtil.tokenize_via_jieba(query)
                          if w in Helper._word2vec] + ['</s>']
            topk = int(params['size'])
            sim_items = (FaissUtil.search_similar_items(query_tokens, topk)
                        if len(query_tokens) > 2 else [])
            if sim_items is not None:
                if cls._wmd_cfg['wmd_flag']:
                    sim_items = cls._wmd_sort(query, sim_items)
                rsp = json.dumps(OrderedDict([
                    ('errno', 0), ('query', query), 
                    ('tokens', ' '.join(query_tokens[1:-1])), 
                    ('count', len(sim_items)), ('data', sim_items)]))
            else:
                rsp = {'errno': 10022, 'errmsg': 'get similar items failed', 
                      'query': query}
            logger.get().debug('Search time elapsed: %.4f sec', 
                time.time() - time_s)
            return (200, rsp)
        except Exception as e:
            logger.get().warn('%s failed, query=%s, errmsg=%s'
                % (myself, query, e))
            rsp = {'errno': 10022, 'errmsg': 'get similar items failed'}
            return (200, rsp)

    
    @classmethod
    def _wmd_sort(cls, query, sim_items):
        myself = sys._getframe().f_code.co_name
        try:
            wmd_number = cls._wmd_cfg['wmd_number'] 
            wmd_sim_items = sim_items[:wmd_number]
            rest_items = sim_items[wmd_number:]
            corpus = [item['k_question'] for item in wmd_sim_items]
            wmd_res = WmdUtil.get_wmd_similarity(query, corpus, wmd_number)
            wmd_res_dict = dict(wmd_res)
            for idx, item in enumerate(wmd_sim_items):
                item['wmd_score'] = wmd_res_dict[idx]
            wmd_sim_items = [wmd_sim_items[idx] for idx, _ in wmd_res]
            logger.get().debug('%s', wmd_res)
            return wmd_sim_items + rest_items
        except Exception as e:
            logger.get().warn('%s failed, errmsg=%s', myself, e)


def test():
    log_path = './log/search_core.py'
    logger.start(log_path, name = __name__, level = 'DEBUG')
    ApiHandler.init()
    params = {'query': 'girl', 'size': 10}
    print ApiHandler.search_similar_items(params)


if __name__ == '__main__':
    test()
