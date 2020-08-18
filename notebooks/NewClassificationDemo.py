# -*- coding: utf-8 -*-
# @Time    : 2020/8/14 15:24
# @Author  : yhao

import sys
import os
import time
import timeit
import warnings
import jieba
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
# %% 文件路径
src_path = "H:/8.数据/新闻分类数据/chinaNews"
tar_path = "G:/数据/中国新闻数据"
file_name = "mini_data_for_chinanews.txt"

# %% 生成小分类数据集

# files = os.listdir(src_path)
#
# class_labels = {"mil", "sh", "cj", "fortune", "gj", "gn", "it", "life", "ty", "cul", "yl"}
#
# with open(os.path.join(tar_path, file_name), mode="w", encoding="utf-8") as f_out:
#     m = 0
#     for file in files:
#         print("开始处理: %s..." % os.path.join(src_path, file))
#         with open(os.path.join(src_path, file), mode="r", encoding="utf-8") as f_in:
#             n = 0
#             for line in f_in.readlines():
#                 tokens = line.split("\u00ef")
#                 if len(tokens) > 8 and tokens[0] in class_labels and len(line) > 10:
#                     f_out.write(line.strip() + "\n")
#                     n += 1
#                     m += 1
#                 if n >= 10000:
#                     break
#             print("读取 %d 条有效数据，剩余文件数：%d" % (n, len(files) - files.index(file) - 1))
#     print("\n写入文件: ", os.path.join(tar_path, file_name))
#     print("写入数据 %d 条" % m)

# %% 为文件分词
# seg_file_name = "mini_data_for_chinanews_seg.txt"
# with open(os.path.join(tar_path, seg_file_name), mode="w", encoding="utf-8") as f_out:
#     with open(os.path.join(tar_path, file_name), mode="r", encoding="utf-8") as f_in:
#         n = 1
#         for line in f_in.readlines():
#             if n % 1000 == 0:
#                 print("已处理 %d 条数据..." % n)
#             n += 1
#             tokens = line.split("\u00ef")
#             if len(tokens) > 8:
#                 label = tokens[0]
#                 label_name = tokens[1].split("\\|")[-1]
#                 title = tokens[2]
#                 time = tokens[3] + " " + tokens[4]
#                 keywords = tokens[6].split(",")
#                 abstract = tokens[7]
#                 content = tokens[8]
#                 content = content[content.index(")")+1:content.rindex("(")] if ")" in content and "(" in content else content
#
#                 if len(content.strip()) > 0:
#                     seg_content = jieba.cut(content)
#                     f_out.write(label + "\u00ef" + " ".join(seg_content) + "\n")

#%% 加载分词后的数据，训练模型并预测
stopword_file = "G:/数据/停用词词典/stopwords.txt"
stopword_list = []
with open(stopword_file, encoding="utf-8") as stf:
    stopword_list = [line.strip() for line in stf.readlines()]

labels_str = []
text = []
seg_file_name = "mini_data_for_chinanews_seg.txt"
with open(os.path.join(tar_path, seg_file_name), encoding="utf-8") as f:
    for line in f.readlines():
        tokens = line.split("\u00ef")
        labels_str.append(tokens[0])
        text.append(tokens[1])

# 标签编码
labels = preprocessing.LabelEncoder().fit_transform(labels_str)

# 文本向量化
vectorizer = CountVectorizer(
    encoding="utf-8",
    token_pattern=r'\b\w+\b',
    max_features=10000,
    stop_words=set(stopword_list))

vectorizer.fit(text)
data = vectorizer.transform(text)

# 切分训练集和测试集
text_train, text_test, labels_train, labels_test = train_test_split(data,
                                                                    labels,
                                                                    random_state=42,
                                                                    stratify=labels,
                                                                    test_size=0.3)

train_samples, n_features = text_train.shape
n_classes = np.unique(labels).shape[0]
print("训练样本数：%d, 特征数：%d, 标签数：%d" % (train_samples, n_features, n_classes))

#%% 模型训练与测试评估
models = {
    "ovr": {"name": "One versus Rest", "iters": [1, 2, 4]},
    "multinomial": {"name": "Multinomial", "iters": [1, 3, 7]}
}

t0 = timeit.default_timer()
solver = 'saga'
for model in models:
    # Add initial chance-level values for plotting purpose
    accuracies = [1 / n_classes]
    times = [0]
    densities = [1]

    model_params = models[model]

    # Small number of epochs for fast runtime
    for this_max_iter in model_params['iters']:
        print('[model=%s, solver=%s] Number of epochs: %s' %
              (model_params['name'], solver, this_max_iter))
        lr = LogisticRegression(solver=solver,
                                multi_class=model,
                                penalty='l1',
                                max_iter=this_max_iter,
                                random_state=42,
                                )
        t1 = timeit.default_timer()
        lr.fit(text_train, labels_train)
        train_time = timeit.default_timer() - t1

        label_pred = lr.predict(text_test)
        accuracy = np.sum(label_pred == labels_test) / labels_test.shape[0]
        density = np.mean(lr.coef_ != 0, axis=1) * 100
        accuracies.append(accuracy)
        densities.append(density)
        times.append(train_time)
    models[model]['times'] = times
    models[model]['densities'] = densities
    models[model]['accuracies'] = accuracies
    print('Test accuracy for model %s: %.4f' % (model, accuracies[-1]))
    print('%% non-zero coefficients for model %s, '
          'per class:\n %s' % (model, densities[-1]))
    print('Run time (%i epochs) for model %s:'
          '%.2f \n' % (model_params['iters'][-1], model, times[-1]))

fig = plt.figure()
ax = fig.add_subplot(111)

for model in models:
    name = models[model]['name']
    times = models[model]['times']
    accuracies = models[model]['accuracies']
    ax.plot(times, accuracies, marker='o',
            label='Model: %s' % name)
    ax.set_xlabel('Train time (s)')
    ax.set_ylabel('Test accuracy')
ax.legend()
fig.suptitle('Multinomial vs One-vs-Rest Logistic L1\n'
             'Dataset %s' % '20newsgroups')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
run_time = timeit.default_timer() - t0
print('Example run in %.3f s' % run_time)
plt.show()
