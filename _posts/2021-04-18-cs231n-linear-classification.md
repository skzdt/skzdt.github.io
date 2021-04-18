---
title: cs231n - Linear Classification


---

<!-- more -->

## Parameterized mapping from images to label scores

training dataset $x_i \in R^D$ with label $y_i, i = 1\dots N, y_i \in 1\dots K$

$N$ examples(each with dimensionality $D$), $K$ distinct categories

score function $f: R^D \mapsto R^K$

eg: CIFAR-10 $N=50000$ images, $D=32\times 32\times 3=3072$ pixels, $K=10$

**linear classifier**: 

$$f(x_i, W, b) =  W x_i + b$$

$W$ , weights; $b$, bias vector 

## Interpreting a linear classifier

![image-20210418220737121](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210418220737.png)

**Analogy of images as high-dimensional points**

We can interpret image as a single point in a hight-dimensional space.

eg: CIFAR-10 3072-dimensional space

![](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210418221305.jpeg)

As we saw above, every row of $W$ is a classifier for one of the classes. The geometric interpretation of these numbers is that as we change one of the rows of $W$, the corresponding line in the pixel space will rotate in different directions. The biases $b$, on the other hand, allow our classifiers to translate the lines. In particular, note that without the bias terms, plugging in $ x_i = 0$ would always give score of zero regardless of the weights, so all lines would be forced to cross the origin.

**Interpretation of linear classifiers as template matching**

$W$ corresponding to a *template* for one of the classes.

![](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210418221603.jpeg)

 The linear classifier is too weak to properly account for different-colored things.

**Hard cases for a linear classifier**

![image-20210418222154342](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210418222154.png)

multi-modal data

**Bias trick**

$$f(x_i, W, b) =  W x_i + b \ \rightarrow\  f(x_i, W) = W x_i$$

![img](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210418221835.jpeg)

## Loss function

> The loss function quantifies our unhappiness with predictions on the training set

**Multiclass SVM loss**

$$L_i = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

![image-20210419013804645](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210419013804.png)

```python
def L_i_vectorized(x, y, W):
  scores = W.dot(x)
  margins = np.maximum(0, scores - scores[y] + 1)
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i
```

<div class="fig figcenter fighighlight">
  <img src="https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210419014220.jpeg">
  <div class="figcaption">
    The Multiclass Support Vector Machine "wants" the score of the correct class to be higher than all other scores by at least a margin of delta. If any class has a score inside the red region (or higher), then there will be accumulated loss. Otherwise the loss will be zero. Our objective will be to find the weights that will simultaneously satisfy this constraint for all examples in the training data and give a total loss that is as low as possible.<br>
  </div>
</div>
---

for $f(x_i; W) =  W x_i$

$$L_i = \sum_{j\neq y_i} \max(0, w_j^T x_i - w_{y_i}^T x_i + \Delta)$$

$$L = \dfrac{1}{N} \sum_i L_i$$

first time $L_i \approx K - 1$(?)

**Regularization**

Suppose that we have a dataset and a set of parameters $W$ that correctly classify every example. The issue is that this set of $W$ is not necessarily unique: there might be many similar $W$ that correctly classify the examples. 

Add **regularization penalty** $R(W)$

**L2**(most common) : $R(W) = \sum_k\sum_l W_{k,l}^2$

**L1** :$R(W) = \sum_k\sum_l \vert W_{k,l}\vert$

**Elastic net(L1+L2)** $R(W) = \sum_k\sum_l \beta W_{k,l}^2+\vert W_{k,l}\vert$

$$L =  \underbrace{ \frac{1}{N} \sum_i L_i }_\text{data loss} + \underbrace{ \lambda R(W) }_\text{regularization loss} \\\\$$

**data loss**: model prediction should match **training data**

**regularization loss**: model should be 'simple' (Occam’s Razor)

Improve the generalization performance of the classifiers on test images and lead to less *overfitting*.

it is common to only regularize the weights $W$ but not the biases $b$.

## Practical Considerations

see [https://cs231n.github.io/linear-classify/#practical-considerations](https://cs231n.github.io/linear-classify/#practical-considerations)

## Softmax classifier

**cross-entropy loss** 交叉熵损失

$$L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) \hspace{0.5in} \text{or equivalently} \hspace{0.5in} L_i = -f_{y_i} + \log\sum_j e^{f_j}$$

