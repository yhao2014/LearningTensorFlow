# -*- coding: utf-8 -*-
# @Time    : 2020/8/14 15:03
# @Author  : yhao

import sys
import os
import jieba

#%%
seg_list = jieba.cut("小明2019年毕业于清华大学")

# 默认分词
print("Default mode: ", ",".join(seg_list))

# 全切分
seg_list = jieba.cut("小明2019年毕业于清华大学", cut_all=True)
print("Full mode: ", ",".join(seg_list))

# 搜索模式分词
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算机研究生，后来到华为去工作")
print("Search mode: ", ",".join(seg_list))
