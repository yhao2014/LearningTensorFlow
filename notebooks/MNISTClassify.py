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
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # 将样本扁平化处理
    keras.layers.Dense(128, activation='relu'),     # 设置该层神经元个数和激活函数
    keras.layers.Dense(10, activation='softmax')    # 设置该层神经元个数和激活函数
])

model.compile(optimizer='adam',                         # 设置最优化算法为adam算法
              loss='sparse_categorical_crossentropy',   # 设置损失函数为交叉熵损失
              metrics=['accuracy'])                     # 设置评估标准

#%% 训练模型
model.fit(train_images, train_labels, epochs=10)        # 网络训练迭代次数epochs=10

#%% 模型评估
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\tTest accuracy: ', test_acc)
print('\tTest loss: ', test_loss)

#%% 预测
predictions = model.predict(test_images)
print(predictions[0])
print(test_labels[0], np.argmax(predictions[0]))        # 打印标签和预测类别


#%% 绘制预测结果
def plot_image(i, predictions_array, labels, imgs):
    predictions_array, label, img = predictions_array, labels[i], imgs[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% {}".format(class_names[predicted_label],
                                       100*np.max(predictions_array),
                                       class_names[label],
                                       color=color))


def plot_value_array(i, predictions_array, labels):
    predictions_array, label = predictions_array, labels[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[label].set_color('blue')


row_num = 5
col_num = 3
image_num = row_num * col_num
plt.figure(figsize=(2*2*col_num, 2*row_num))
for i in range(image_num):
    plt.subplot(row_num, 2*col_num, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(row_num, 2*col_num, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#%% 单样本预测
print(test_images[1].shape)
img = np.expand_dims(test_images[1], 0)     # 在axis=0添加维度
print(img.shape)

predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
