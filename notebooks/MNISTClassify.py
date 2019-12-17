# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 17:06
# @Author  : yhao

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#%% 导入mnist数据集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

#%% 打印样本数据
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#%% 对样本进行归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

#%% 打印25个样本
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#%% 设置神经网络结构
model = keras.Sequentail([
    keras.layers.Flatten(input_shape=(28, 28)),     # 将样本扁平化处理
    keras.layers.Dense(128, activation='relu'),     # 设置该层神经元个数和激活函数
    keras.layers.Dense(10, activation='softmax')    # 设置该层神经元个数和激活函数
])
