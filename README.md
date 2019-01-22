# Sentence Embeddings

## Introduction

Build sentence embeddings encoder for searching purpose.

NOTE: the project is dividied to three parts, data preprocess, model training and searching supply. Each part has a directory to keep its codes and data, and common used codes and data is placed in `tools` directory.


## Graphs
![](http://p09kjlqc4.bkt.clouddn.com/18-1-6/50100971.jpg)


## Running Steps


### Step 1: Prepare data for model training

```shell
./model_data_preprocess.sh
```

### Step 2: Train model

```shell
cd sentence_embeddings_model
python bin/model.py 
```

### Step 3: Prepare data for searching


```shell
./search_data_preprocess.sh
```


### Step 4: Supply searching

```shell
cd sentence_embeddings_search
supervisord -c conf/sup.qsearch.conf 
```

After process enabled, url (e.g. http://211.159.179.239:6008/in/nlp/sentence/search?query=产后失眠怎么办&size=100) can be visited.


## Analysis

For more infomation of model and searching, please refer to [**model_analysis**](model_analysis.md)

## Reference

- [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364)
