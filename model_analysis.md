# 模型分析


## Infersent模型分析

### 论文复现工作

此项工作是基于facebook发表的论文：[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364)，复现了论文的效果，在开发集上准确率能够达到84.9% (论文中是85.0%)。

### NLI数据构造

原论文中使用的数据集是SNLI，该数据集如下：

![](http://p09kjlqc4.bkt.clouddn.com/18-1-6/74218997.jpg)

在母婴知识库数据集上，尝试了两种构造方式：

- 相邻构造

    entailment: 同一知识点下，选择两个问题q1, q2，构成entailment关系；

    contradiction: 相邻知识点（属于同一级标签/tag下），选择两个问题q1, q2，构成contradiction关系。

    存在的问题：同一知识点下，问题描述可能存在差异（很多时候差异较大），不能构成严格的entailment关系，又因为相邻知识点的问题意思较近，不能构成严格的contradiction关系，所以这个二分类问题界限不清晰，导致模型分类结果不理想（73%准确率），根据这种方式训练的模型去搭建检索，top3准确率85%。

- 随机构造

    entailment: 同一知识点下，选择两个问题q1, q2，构成entailment关系；

    contradiction: 随机选择两个不属于同一知识点的问题q1, q2，构成contradiction关系。

    这样的构造方式由于两种关系差异较大，易于二分类（97%准确率），根据这种方式训练的模型去搭建检索，top3准确率91%。



