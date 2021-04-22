---
title: NTUML - Introduction of ML/DL

---

<!-- more -->

<iframe src="https://www.youtube.com/embed/Ye018rCVvOo" allowfullscreen></iframe>

<iframe src="https://www.youtube.com/embed/bHcJCp2Fyxs" allowfullscreen></iframe>


ML$\approx$ Looking for a function

different types of functions:

1. **regression**: The function outputs a scalar
2. **Classification**: Given options (classes), the function outputs the correct one.

Classification 可以多选项

![](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210421222929.png)

Structured Learning: **create** something with structure(image, document)

以预测YouTube频道播放量为例

1. 基于domain knowledge建立猜测模型, 带有未知参数

![image-20210421223608551](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210421223608.png)

2. 设定Loss Function

   ![image-20210421224114755](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210421224114.png)

3. Optimization

   Gradient Descent

   ​	local minima 不是重要的问题

通过观察 发现7天有个明显周期, 更新模型

![image-20210421230115486](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210421230115.png)

通过增加考虑的天数, 有所改善, 但是也到达了某种极限

**Model Bias**

使用一个具有阶跃的函数 加上参数用来拟合 **引入Sigmoid**

![image-20210421230838360](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210421230838.png)

调参来改变sigmoid的形状

![image-20210421230910585](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210421230910.png)

更新一次参数 **update**

看完所有batch **epoch**

![image-20210421232426119](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210421232426.png)

引入 **activation function**

![image-20210421232851068](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210421232851.png)

引入**hidden layer**

![image-20210421233303273](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210421233303.png)

