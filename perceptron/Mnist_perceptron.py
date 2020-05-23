#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/23 下午2:31
# @Author  : YuanYun
# @Site    : 
# @File    : Mnist_perceptron.py
# @Software: PyCharm

"""
文件说明：
使用感知机对Mnist数据集作分类
"""

import numpy as np
import time
from matplotlib import pyplot as plt


def loadData(fileName):
    """
    加载文件
    :param fileName:要加载的文件路径
    :return: 数据集和标签集
    """
    print('start read file')
    # 存放数据及标记
    data_arr = []
    label_arr = []
    # 读取文件
    fr = open(fileName)
    # 遍历文件中的每一行
    for line in fr.readlines():
        # strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
        cur_line = line.strip().split(',')
        # 将每行中除标记外的数据放入数据集中
        data_arr.append([int(num) for num in cur_line[1:]])
        # 将标记信息放入标记集中
        if int(cur_line[0]) > 5:
            label_arr.append(1)
        else:
            label_arr.append(-1)
    # 返回数据集和标记
    return data_arr, label_arr


def percetron(data_arr, label_arr, iter=30):
    """
        感知机算法
        :param data_arr:训练集数据
        :param label_arr: 训练集标记
        :param iter: 迭代次数
        :return: 训练好的 权重 和 bias
    """
    print('start to train')
    data_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).T
    m, n = np.shape(data_mat)

    w = np.zeros((1, np.shape(data_mat)[1]))
    print(np.shape(data_mat))
    b = 0
    rate = 0.01

    # 迭代计算

    for k in range(iter):
        # 有两种梯度下降方式
        # 1.把全部样本计算一遍之后，进行一次梯度下降
        # 2.每计算一个样本，针对该样本进行一次梯度下降
        for i in range(m):
            xi = data_mat[i]
            yi = label_mat[i]
            # 判断分类结果
            if yi * (w * xi.T + b) <= 0:
                # 对于误分类样本，进行梯度下降，更新w和b
                w = w + rate * yi * xi
                b = b + rate * yi
            # 打印训练进度
        print('train: (%d/%d)' % (k, iter))

    return w, b


def model_test(testDataArr, testLabelArr, w, b):
    """
        测试正确率
        :param trainDataArr:训练集数据集
        :param trainLabelArr: 训练集标记
        :param w :权重
        :param b :偏置
        :return: 正确率
    """
    print('start test')
    # 将所有列表转换为矩阵形式，方便运算
    testDataMat = np.mat(testDataArr)
    testLabelMat = np.mat(testLabelArr).T
    m, n = np.shape(testDataMat)
    errorCnt = 0
    # 遍历测试集，对每个测试集样本进行测试
    # for i in range(len(testDataMat)):
    for i in range(m):
        xi = testDataMat[i]
        yi = testLabelMat[i]
        print('test (%d/%d)' % (i, m))

        if yi * (w * xi.T + b) <= 0:
            errorCnt += 1

    # 返回正确率
    return 1 - (errorCnt / m)


if __name__ == "__main__":
    start = time.time()

    # 获取训练集
    trainData, trainLabel = loadData('../mnist/mnist_train.csv')
    # 获取测试集
    testData, testLabel = loadData('../mnist/mnist_test.csv')
    # 画出k值对应正确率的散点图
    w, b = percetron(trainData, trainLabel)
    accur = model_test(testData, testLabel,w,b)

    end = time.time()
    # 打印正确率
    print('accuracy is:%d' % (accur * 100), '%')
    # 显示花费时间
    print('cost time:', end - start)
