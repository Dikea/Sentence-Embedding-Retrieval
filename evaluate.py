#!/bin/env python
#-*- encoding: utf-8 -*-


import time
import codecs
import random
import requests
from collections import Counter 
from collections import defaultdict
import numpy as np


host = '211.159.179.239:6008'
api = 'http://%s/in/nlp/sentence/search' % (host)


"""
Print badcase for analysis
"""
PRINT_BADCASE_FLAG = True

if PRINT_BADCASE_FLAG:
    badcase_f = codecs.open('./badcase.txt', 'w', 'utf-8')
    badcase_f.write('Q: question, K: knowledge, D: l2distance\n\n')

def print_badcase(query, k_title, stat_array, result, check_number = 3):
    if not PRINT_BADCASE_FLAG or stat_array[check_number - 1]:
        return 
    badcase_f.write('>>> Q-%s\tK-%s\n' % (query, k_title))
    badcase_f.write('top 10: %s' % stat_array[:10] + '\n')
    questions = ['Q-%s\tK-%s\tD-%s' %
                 (info['k_question'], info['k_title'], info['l2distance'])
                 for info in result[:check_number]]
    for idx, ques in enumerate(questions):
        badcase_f.write('%d\t%s\n' % (idx, ques))
    badcase_f.write('\n')


def evaluate(test_data, qid2kid_dict, topk = 5, size = 100):
    """
    Evaluate search effect(accuracy) on topk 
    record of each query result
    """
    print 'Test number is: %d' % (len(test_data))
    test_len = len(test_data)
    stat_matrix = np.ones((test_len, size), dtype = bool)

    for idx, item in enumerate(test_data):
        query_id, query = item[:2]
        kid_set = qid2kid_dict[query_id]
        stat_array, result = get_query_result(query, query_id, kid_set, size)
        print_badcase(query, item[3], stat_array, result) 
        stat_matrix[idx, :] = stat_array
        if idx % 200 == 0:
            print 'Handle %d queries done' % (idx)
    
    for idx in xrange(topk):
        print 'Accuracy of top %d : %.4f' % (idx + 1, 
            sum(stat_matrix[:, idx]) * 1.0 / test_len)
    return True


def get_query_result(query, query_id, kid_set, size):
    """
    Get single query result
    """
    stat_array = np.array([True for _ in xrange(size)])
    payload = {
        'query': query,
        'size': size}
    r = requests.get(api, params = payload).json()
    if r['errno'] != 0:
        return stat_array 
    # Remove same question as query, for leave one out test
    result = [info for info in r['data'] if info['k_question'] != query]
    # Remove duplicated knowldege(k_id) in result
    uniq_kid_set = set()
    uniq_result = []
    for info in result:
        if info['k_id'] not in uniq_kid_set:
            uniq_kid_set.add(info['k_id'])
            uniq_result.append(info)
    result = uniq_result
    result_len = len(result)
    for idx in xrange(result_len):
        if result[idx]['k_id'] in kid_set:
            return stat_array, result
        else:
            stat_array[idx] = False
    if result_len != size:
        stat_array[result_len:] = False
    return stat_array, result
    

def get_test_data(data_path, num = 1000, k_q_num_lower_limit = 5, 
                  rand_seed = None):
    """
    Prepare test data
    """
    with codecs.open(data_path, 'r', 'utf-8') as rfd:
        data = [line.rstrip().split('|') for line in rfd.readlines()[1:]]

    # Dict of knowledge id to question number of the knowledge
    kid2num_dict = Counter()
    # Dict of question to question id
    question2qid_dict = defaultdict(set) 
    # Dict of question to knowledge id
    question2kid_dict = defaultdict(set) 
    # Dict of question id to knowledge id
    qid2kid_dict = defaultdict(set)

    for info in data:
        kq_id, k_question, k_id = info[:3]
        kid2num_dict[k_id] += 1
        question2qid_dict[k_question].add(kq_id)
        question2kid_dict[k_question].add(k_id)
    
    # Build dict of question id to knowldege id,
    # because the same question can exist in different 
    # knowledge with different question id
    for q, qid_set in question2qid_dict.items():
        kid_set = question2kid_dict[q]
        for qid in qid_set:
            qid2kid_dict[qid] = kid_set
    
    # If questions number of knowledge lower than lower_limit,
    # discard the knowledge (don't toke it to test accuracy)
    data = [info for info in data if kid2num_dict[info[2]] >= 
            k_q_num_lower_limit and not info[0].startswith('#')]
    if rand_seed is not None:
        random.seed(rand_seed)
    random.shuffle(data)

    return data[:num], qid2kid_dict


if __name__ == '__main__':
    time_s = time.time()
    data_path = './sentence_embeddings_preprocess/data/gemii_knowledge.txt'

    test_data, qid2kid_dict = get_test_data(data_path, num = 3000,
        k_q_num_lower_limit = 5, rand_seed = 1234)
    evaluate(test_data, qid2kid_dict, topk = 50, size = 100)

    print 'Time elapsed %.4f seconds' % (time.time() - time_s)
