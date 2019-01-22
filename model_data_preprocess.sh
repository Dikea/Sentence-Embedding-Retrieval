#!/bin/bash

echo 'Model prepare start...'

cd sentence_embeddings_preprocess/
python bin/model_preprocess.py
python bin/pre_tokenize.py

echo 'Model prepare done.'
