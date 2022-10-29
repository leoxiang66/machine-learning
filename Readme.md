# 机器学习算法及练习

- 理论知识: https://www.zhihu.com/column/c_1409273240435421184
- 问题讨论: https://kaihei.co/fMCUT6



---
---

# 机器学习基础知识

# 前言

这里是第四期训练营机器学习，本文将简单介绍机器学习的基础知识，帮助您快速了解机器学习，不会对阅读有相关的前提要求。

# 常见的机器学习分类

## 机器学习通常分为四类：

- 监督学习
- 无监督学习
- 半监督学习
- 强化学习

## 监督学习

在监督学习当中，每一个实例都是由一个输入对象和一个期望的输出值组成。监督学习算法是通过分析该训练数据，并产生一个推断。一个好的方案可以使得监督学习算法正确地做出决策。

### 在监督学习中，有两个典型的分类：

1. 分类问题：
   以邮件分类过滤为例，通过一组特性，将邮件分为正常邮件以及垃圾邮。
2. 回归问题：
   我们期望得到的回归结果是对目标数值进行预测，例如在住房问题上，通过一组特性（大小，房间数量等），预测房屋的售价。

### 以下是常见的监督学习的算法：

- Logistic Regression （逻辑回归）
- k-Nearest Neighbors （K近邻算法）
- Support Vector Machines (SVMs) （支持向量机）
- Naive Bayesian Model （朴素贝叶斯）
- Decision Trees （决策树）
- Random Forests （随机森林）
- Neural networks （神经网络）

## 无监督学习

无监督学习就是按照数据的性质将他们自动分为很多组，即所有的数据都只有特征向量而没有标签。

### 以下是常见的无监督学习的算法：

### 聚类

- k-Means
- spectral clustering （谱聚类）
- mean-shift

### 降维

- Principal Component Analysis (PCA)
- Feature Selection （特征选择）
- Nonnegative Matrix Factorization （非负矩阵分解）

### 半监督学习

半监督学习在训练过程中结合了大量的未标记数据以及少量的标签数据，能够带来更小的训练成本以及更高的准确度

### 强化学习

强化学习是智能体不断与环境进行交互，通过试错的方式来回的最佳策略，强化学习的典型例子就是阿尔法狗。

# 总结与后文

机器学习是一门多领域的交叉学科，而机器学习的算法已经广泛应用在了我们的日常生活当中。在后续的文章当中，我们会详细介绍学习机器学习所涉及的算法。

# 参考来源

1. 统计学习方法-李航