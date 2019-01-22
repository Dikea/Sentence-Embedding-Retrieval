#!/env/bin python
#-*- encoding: utf-8 -*-

import os
import sys
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim import models
import codecs
import nltk
import pickle as pkl
from nlp_util import NLPUtil
from log import g_log_inst as logger
import config


class Helper(object):


    _word2vec = None


    @classmethod
    def init(cls):
        """Prepare required data"""
        myself = sys._getframe().f_code.co_name
        try:
            # Load word2vec
            w2v_path = config.word2vec_path
            cls._word2vec = cls.get_word2vec(w2v_path)
            return True
        except Exception as e:
            logger.get().warn('%s failed, errmsg=%s', myself, e)
            return False

    
    @classmethod
    def get_batch(cls, batch):
        """Embeds the words of sentences using word2vec"""
        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        embed = np.zeros((len(batch), max_len, config.embeddings_size))
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[i, j, :] = cls._word2vec[batch[i][j]]
        return embed, lengths
 

    @classmethod
    def get_word2vec(cls, w2v_fpath):
        wv = models.KeyedVectors.load_word2vec_format(w2v_fpath, binary = False)
        logger.get().info('load word2vec success, vector length: %s', 
                (wv['<s>'].shape[0]))
        return wv

    
    @classmethod
    def get_nli(cls, data_path):
        s1 = {}
        s2 = {}
        target = {}
        
        dico_label = {'entailment': 0, 'contradiction': 1}
        
        for data_type in ['train', 'dev', 'test']:
            s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
            s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type + '.tokenized')
            s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type + '.tokenized')
            target[data_type]['path'] = os.path.join(data_path, 'labels.' + data_type)
            
            s1[data_type]['sent'] = [line.rstrip() for line in 
                codecs.open(s1[data_type]['path'], 'r', 'utf-8')]
            s2[data_type]['sent'] = [line.rstrip() for line in 
                codecs.open(s2[data_type]['path'], 'r', 'utf-8')]
            target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')]
                for line in open(target[data_type]['path'], 'r')])
            
            assert (len(s1[data_type]['sent']) == len(s2[data_type]['sent'])
                    == len(target[data_type]['data']))
            
            print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                data_type.upper(), len(s1[data_type]['sent']), data_type))
            
        train = {'s1':s1['train']['sent'], 's2':s2['train']['sent'], 
            'label':target['train']['data']}
        dev = {'s1':s1['dev']['sent'], 's2':s2['dev']['sent'], 
            'label':target['dev']['data']}
        test  = {'s1':s1['test']['sent'] , 's2':s2['test']['sent'] , 
            'label':target['test']['data'] }

        # Tokenize sentences
        for split in ['s1', 's2']:
            for data_type in ['train', 'dev', 'test']:
                eval(data_type)[split] = np.array([['<s>'] + 
                    [word for word in sent.split(' ')
                    if word in cls._word2vec] + ['</s>'] for sent in 
                    eval(data_type)[split]])     
        print 'Load train, dev, test data successfully'
        return train, dev, test


def test():
    Helper.init()
    train, dev, test = Helper.get_nli(config.gemii_nli_path)
    print '|'.join(train['s1'][0]), '|'.join(train['s2'][0]), train['label'][0]
    print '|'.join(train['s1'][1]), '|'.join(train['s2'][1]), train['label'][0]


if __name__ == '__main__':
    test()
