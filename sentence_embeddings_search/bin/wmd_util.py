#!/bin/env python
#-*- encoding: utf-8 -*-


import sys
sys.path.append('../')
from gensim.models.word2vec import Word2Vec
from gensim.similarities import WmdSimilarity
from tools.nlp_util import NLPUtil
from tools.log import g_log_inst as logger
from tools import config


class WmdUtil(object):


    _word2vec= None


    @classmethod 
    def init(cls):
        myself = sys._getframe().f_code.co_name
        try:
            w2v_path = config.temp_word2vec_path
            w2v_model = Word2Vec.load(w2v_path)
            cls._word2vec = w2v_model.wv
            cls._word2vec.init_sims(replace = True)
            logger.get().debug('load word2vec for wmd success')
            return True
        except Exception as e:
            logger.get().warn('%s failed, errmsg=%s', myself, e)
            return False


    @classmethod
    def get_wmd_similarity(cls, doc, corpus, limit_number = 30):
        myself = sys._getframe().f_code.co_name
        try:
            tokenize_func = NLPUtil.tokenize_via_jieba
            corpus = map(tokenize_func, corpus)
            corpus_size = len(corpus)
            wmd_inst = WmdSimilarity(corpus, 
                                     cls._word2vec,
                                     num_best = limit_number, 
                                     normalize_w2v_and_replace = False)
            doc_tokens = tokenize_func(doc)
            similar_items = wmd_inst[doc_tokens] if doc_tokens else []
            return similar_items
        except Exception as e:
            logger.get().warn('%s failed, doc=%s, limit_number=%d', 
                doc, limit_number)
            raise 


def test():
    logger.start('./log/wmd_util.log', name=__name__, level='DEBUG')
    WmdUtil.init()
    doc = u'宝宝什么时候可以用安抚奶嘴'
    corpus = [u'安抚奶嘴一般什么时候给宝宝用比较好',
              u'安抚奶嘴什么时候用',
              u'宝宝多久可以用安抚奶嘴']
    res = WmdUtil.get_wmd_similarity(doc, corpus)
    print res


if __name__ == '__main__':
    test()
