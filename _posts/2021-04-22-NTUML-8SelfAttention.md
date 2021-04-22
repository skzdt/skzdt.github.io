---
title: NTUML - Self-attention
---

<!-- more -->

<iframe src="https://www.youtube.com/embed/hYdO9CscNes" allowfullscreen></iframe>

<iframe src="https://www.youtube.com/embed/gmsMY5kc-zw" allowfullscreen></iframe>

复杂输入的挑战

输入是一个序列 长度不定

![image-20210422202900738](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422202900.png)

向量化词汇

![image-20210422203053417](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422203053.png)

[ML Lecture 14: Unsupervised Learning - Word Embedding - YouTube](https://www.youtube.com/watch?v=X7PH3NuYW0Q)

向量化声音信号

![image-20210422203233979](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422203234.png)

向量化 图, 分子模型等

**输出?**

输出N对N(seq labeling)

N对一

seq to seq 长度不同的输出

![image-20210422203809166](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422203809.png)

N2N问题 FC会有问题, 相同的输入会有相同输出

引入window等方法

![image-20210422203923871](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422203923.png)

seq有长有短, window长度难选

引入更好的方法, 处理整个seq

![image-20210422211853989](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422211854.png)

>  Attention is all you need.

![image-20210422212023299](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422212023.png)

输出的向量 都是考虑了所有的输入才得到的

![image-20210422212135584](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422212135.png)

去寻找相关性$\alpha$

![image-20210422212249908](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422212249.png)

一些计算方法

![image-20210422212336253](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422212336.png)

q - query

k - key

自己和自己也需要进行一次

![image-20210422212445453](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422212445.png)

![image-20210422212634533](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422212634.png)

$b^2$ 等不需要依次产生, 可以并行

![image-20210422212817036](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422212817.png)

$b^2$同理

化为矩阵乘法

![image-20210422212910527](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422212910.png)

拼矩阵, $a$得到$Q, K, V$

![image-20210422212952328](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422212952.png)

![image-20210422213300434](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422213300.png)

![image-20210422213332898](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422213333.png)

**multi-head**

![image-20210422213539283](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422213539.png)

相关这件事有很多不同的形式, 有很多种不同的定义

$q$负责不同种类的相关性

![image-20210422213813641](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422213813.png)

self-attention 层缺失位置信息

![image-20210422213928103](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422213928.png)

positional encoding还在研究中

![image-20210422214202725](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422214202.png)

平方的复杂度 注意力矩阵

还是要加类似window这种

![image-20210422214308667](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422214308.png)

![image-20210422214354344](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422214354.png)

network通过学习 自己去划定范围

![image-20210422214450238](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422214450.png)

![image-20210422214526367](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422214526.png)

![image-20210422214647771](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422214647.png)

RNN 的顺序问题

最右的vec 考虑最左的vec 记忆性不好

但是Self-attention不会有这种问题

RNN不能并行

![image-20210422214931828](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210422214931.png)

在图上来说, 注意力矩阵可以是图的邻接矩阵, 就不用去学习