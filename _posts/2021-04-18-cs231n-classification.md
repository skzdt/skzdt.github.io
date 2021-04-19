---
title: cs231n - Image Classification - Data-driven Approach
---

<!-- more -->

## Image Classification pipeline

**image classification** : A core task in CV

1. **semantic gap** 语义鸿沟

   ![](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/classify.png)

2. **Challenge**: 
   
   - **Viewpoint variation**. A single instance of an object can be oriented in many ways with respect to the camera.
   - **Scale variation**. Visual classes often exhibit variation in their size (size in the real world, not only in terms of their extent in the image).
   - **Deformation**. Many objects of interest are not rigid bodies and can be deformed in extreme ways.
   - **Occlusion**. The objects of interest can be occluded. Sometimes only a small portion of an object (as little as few pixels) could be visible.
   - **Illumination conditions**. The effects of illumination are drastic on the pixel level.
   - **Background clutter**. The objects of interest may *blend* into their environment, making them hard to identify.
   - **Intra-class variation**. The classes of interest can often be relatively broad, such as *chair*. There are many different types of these objects, each with their own appearance.

![](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/challenges.jpeg)

3. **classifier**

   ```python
   def classify_image(image):
     # some magic here?
     return class_label
   ```

4. **Data-Driven Approach**

   1. **Collect** a dataset of images and labels
   2. Use Machine Learning to **train a classifier**
   3. **Evaluate** the classifier on new images

   ```python
   def train(images, labels):
       # Machine learning!
       return model
   def predict(model, test_images):
       # use model to predict labels
       return test_labels
   ```

---

**The image classification pipeline**. We’ve seen that the task in Image Classification is to take an array of pixels that represents a single image and assign a label to it. Our complete pipeline can be formalized as follows:

   - **Input:** Our input consists of a set of *N* images, each labeled with one of *K* different classes. We refer to this data as the *training set*.
   - **Learning:** Our task is to use the training set to learn what every one of the classes looks like. We refer to this step as *training a classifier*, or *learning a model*.
   - **Evaluation:** In the end, we evaluate the quality of the classifier by asking it to predict labels for a new set of images that it has never seen before. We will then compare the true labels of these images to the ones predicted by the classifier. Intuitively, we’re hoping that a lot of the predictions match up with the true answers (which we call the *ground truth*).

## Nearest Neighbor Classifier

```python
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
```

Train $O(1)$  Predict: $O(n)$

we want classifiers that are **fast** at prediction; **slow** for training is ok

**Distance Metric to compare images**

**L1 distance (Manhattan distance)**

$$d_1 (I_1, I_2) = \sum_{p} \left| I^p_1 - I^p_2 \right|$$

![](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210418012026.jpeg)

**L2 distance (Euclidean distance)**

$$d_2 (I_1, I_2) = \sqrt{\sum_{p} \left( I^p_1 - I^p_2 \right)^2}$$

```python
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
```

![image-20210418014343773](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210418014343.png)

![image-20210418014426066](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210418014426.png)

## k - Nearest Neighbor Classifier

Instead of finding the single closest image in the training set, we will find the top **k** closest images, and have them vote on the label of the test image.

Intuitively, higher values of **k** have a smoothing effect that makes the classifier more resistant to outliers:

![](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210418013204.jpeg)

1. **Hyper-parameters**

   Choices about the algorithm that we set rather than learn, such as **K** (K-Nearest Neighbor), distance metric.

   Very problem-dependent.

2. **Setting Hyper-parameters**

   ![image-20210418014858291](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210418014858.png)![image-20210418015108330](https://lllthhhh-aliyun-oss.oss-cn-beijing.aliyuncs.com/img/20210418015108.png)

- Very slow at test time(on images, there are too many pixels.)
- Distance metrics on pixels are not informative.

**So, K-Nearest Neighbor on images is never used.**



