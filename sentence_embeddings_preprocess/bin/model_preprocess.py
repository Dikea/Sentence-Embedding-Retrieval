#!/bin/env python
#-*- encoding: utf-8 -*-


import sys
sys.path.append('../')

import os
import time
import random
import itertools
import shutil
import codecs
from knowledge_util import KnowledgeTree

from tools import config
from tools.nlp_util import NLPUtil


# Paras settings
FAST_FLAG = True
MAX_DIFF_NUM = 20 if FAST_FLAG else 4
SINGLE_QUESTION_NUM_LIMIT = 20
MAX_SENTENCE_LENGTH = 200


class NliDataBuilder(object):

    
    def __init__(self):
        self._knowledge_fpath = config.gemii_knowledge_fpath

        # nli entailment/ data
        self._raw_data = []
        self._nli_data = []
        self._unique_pair_set = set()
        self._id2question_dict, self._raw_data = self._get_id2question_dict()

        # Build knowledge tree
        self._knowledge_tree = KnowledgeTree(self._knowledge_fpath)
        self._knowledge_tree.build()

        # Get existing ktagid path set
        self._ktagid_path_set = self._get_ktagid_path_set()
    

    def build(self):
        for id_path in self._ktagid_path_set:
            tree_node = self._knowledge_tree.get_node(id_path)

            # Construct nli data
            for key in tree_node.keys():
                if key.startswith('#'):
                    self._contruct_nli_data(key, tree_node)
        
        # Save nli data
        self._save()


    def _contruct_nli_data(self, k_id, tree_node):
        knowledge = tree_node[k_id]
        q_id_list = knowledge.keys()
        pairs_list = []
        last_id1 = q_id_list[0]
        for id1, id2 in itertools.combinations(q_id_list, 2):
            question1, question2 = knowledge[id1], knowledge[id2]
            entailment_pair = (id1, id2, 1)
            if not self._is_valid_pair(entailment_pair):
                continue
            contradiction_pair = self._get_contradiction_pair(
                id1, question1, k_id, tree_node)
            if contradiction_pair:
                self._unique_pair_set.add(entailment_pair)
                self._unique_pair_set.add(contradiction_pair)   
                pairs_list.append((entailment_pair, contradiction_pair))
                if id1 is not last_id1:
                    limit_num = SINGLE_QUESTION_NUM_LIMIT
                    if len(pairs_list) > limit_num:
                        random.shuffle(pairs_list)
                        pairs_list = pairs_list[:limit_num]
                    self._nli_data.extend(pairs_list)
                    pairs_list = []
                    last_id1 = id1
        return True 

    
    def _get_contradiction_pair(self, question_id, question, k_id, 
                                tree_node, random_mode = True):
        # Random select question from different knowledge to build contradicton
        if random_mode:
            data_len = len(self._raw_data)
            while True:
                item = self._raw_data[random.randint(0, data_len - 1)]
                pair = (question_id, item[0], -1)
                if item[2] is not k_id and self._is_valid_pair(pair):
                    break
            return pair 

        # Select question from adjacent knowledge to build contradicton 
        pair = ()
        k_id_list = [key for key in tree_node.keys() if 
                     key.startswith('#') and key is not k_id]
        if not k_id_list:
            return pair
        random_k_id = k_id_list[random.randint(0, len(k_id_list) - 1)] 
        question_id_list = tree_node[random_k_id].keys()
        qid_len = len(question_id_list)
        for _ in xrange(qid_len):
            random_question_id = question_id_list[
                random.randint(0, qid_len - 1)]
            pair = (question_id, random_question_id, -1)
            rpair = (random_question_id, question_id, -1)
            random_question = self._id2question_dict.get(random_question_id)
            if self._is_valid_pair(pair):
                break
        return pair if self._is_valid_pair(pair) else ()

    
    def _is_valid_pair(self, pair):
        if not pair:
            return False
        rpair = (pair[1], pair[0], pair[2])
        if pair in self._unique_pair_set or rpair in self._unique_pair_set:
            return False
        question1 = self._id2question_dict.get(pair[0])
        question2 = self._id2question_dict.get(pair[1])
        # If sentence length exceed max_length, return false
        max_length = MAX_SENTENCE_LENGTH
        if len(question1) > max_length or len(question2) > max_length:
            return False
        if FAST_FLAG:
            # Judge by sentence length
            return abs(len(question1) - len(question2)) <= MAX_DIFF_NUM
        # Judge by token length
        tokens1 = NLPUtil.tokenize_via_jieba(question1)
        tokens2 = NLPUtil.tokenize_via_jieba(question2)
        if abs(len(tokens1) - len(tokens2)) > MAX_DIFF_NUM:
            return False
        return True

    
    def _get_ktagid_path_set(self):
        with codecs.open(self._knowledge_fpath, 'r', 'utf-8') as in_f:
            in_f.readline()
            ktagid_path_set = set([line.strip('\n').split('|')[-1] 
                                   for line in in_f.readlines()])
        return ktagid_path_set

    
    def _get_id2question_dict(self):
        id2question_dict = {}
        with codecs.open(self._knowledge_fpath, 'r', 'utf-8') as in_f:
            in_f.readline()
            data = map(lambda x: x.rstrip().split('|'), in_f.readlines())
            for item in data:
                id2question_dict[item[0]] = item[1]
        return id2question_dict, data
        

    def _save(self):
        nli_path = config.gemii_nli_path
        if os.path.exists(nli_path):
            shutil.rmtree(nli_path)
            print 'Remove existing nli-folder'
        os.mkdir(nli_path)

        data_len = len(self._nli_data)  
        data_types = ['train', 'dev', 'test']
        data_ratios = [0.964, 0.018, 0.018]

        for idx, data_type in enumerate(data_types):
            start_num =  int(data_len * sum(data_ratios[:idx]))
            end_num = int(data_len * sum(data_ratios[:idx+1]))
            s1_path = os.path.join(nli_path, 's1.' + data_type)
            s2_path = os.path.join(nli_path, 's2.' + data_type)
            target_path = os.path.join(nli_path, 'labels.' + data_type)
            with codecs.open(s1_path, 'w', 'utf-8') as s1_f, \
                 codecs.open(s2_path, 'w', 'utf-8') as s2_f, \
                 codecs.open(target_path, 'w', 'utf-8') as target_f:
                for item in self._nli_data[start_num: end_num]:
                    for pair in item:
                        s1_f.write(self._id2question_dict.get(pair[0]) + '\n')
                        s2_f.write(self._id2question_dict.get(pair[1]) + '\n')
                        value = 'entailment' if pair[2] is 1 else 'contradiction'
                        target_f.write(value + '\n')

        print 'Nli data saved'


def test():
    nli_builder = NliDataBuilder()
    print nli_builder._is_valid_pair(('4532', '4506', -1))

if __name__ == '__main__':
    #test()
    time_s = time.time()
    nli_builder = NliDataBuilder()
    nli_builder.build()
    print 'Time elapsed: %.4f seconds' % (time.time() - time_s)
