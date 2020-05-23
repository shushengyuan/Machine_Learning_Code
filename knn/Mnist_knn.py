#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/20 下午5:58
# @Author  : YuanYun
# @Site    : 
# @File    : Mnist_knn.py
# @Software: PyCharm

"""
文件说明：
    数据集为Mnist的 knn算法
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
        label_arr.append(int(cur_line[0]))
    # 返回数据集和标记

    return data_arr, label_arr


def calc_dist(x1, x2):
    """
    计算两个样本点向量之间的距离
    使用的是欧氏距离，即 样本点每个元素相减的平方和再开方
    :param x1:向量1
    :param x2:向量2
    :return:向量之间的欧式距离
    """
    return np.sqrt(np.sum(np.square(x1 - x2)))

    # 曼哈顿距离计算公式
    # print(x1.dtype)
    # print(x2)
    # print(np.sum(x1 - x2))
    # return np.sum(x1 - x2)


def getClosest(trainDataMat, trainLabelMat, x, topK):
    """
    预测样本x的标记。
    获取方式通过找到与样本x最近的topK个点，并查看它们最多的那类标签
    :param trainDataMat:训练集数据集
    :param trainLabelMat:训练集标签集
    :param x:要预测的样本x
    :param topK:选择最邻近样本数目
    :return:预测的标记
    """
    # 建立一个存放向量x与每个训练集中样本距离的列表
    # 列表的长度为训练集的长度，distList[i]表示x与训练集中第i个样本的距离

    distList = [0] * len(trainLabelMat)
    # distList = np.zeros_like(trainLabelMat)
    # 遍历训练集中所有的样本点，计算与x的距离

    for i in range(len(trainDataMat)):
        # 获取训练集中当前样本的向量
        x1 = trainDataMat[i]
        # 计算向量x与训练集样本x的距离
        curDist = calc_dist(x1, x)
        # 将距离放入对应的列表位置中
        distList[i] = curDist

    # 对距离列表进行排序
    topKList = np.argsort(distList)[:topK]  # 升序排序

    labelList = [0] * 10

    # 对topK个索引进行遍历
    for index in topKList:
        labelList[int(trainLabelMat[index])] += 1
    return labelList.index(max(labelList))


def model_test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK):
    """
    测试正确率
    :param trainDataArr:训练集数据集
    :param trainLabelArr: 训练集标记
    :param testDataArr: 测试集数据集
    :param testLabelArr: 测试集标记
    :param topK: 选择多少个邻近点参考
    :return: 正确率
    """
    print('start test')
    # 将所有列表转换为矩阵形式，方便运算
    trainDataMat = np.mat(trainDataArr)
    trainLabelMat = np.mat(trainLabelArr).T
    testDataMat = np.mat(testDataArr)
    testLabelMat = np.mat(testLabelArr).T

    errorCnt = 0
    # 遍历测试集，对每个测试集样本进行测试
    # for i in range(len(testDataMat)):
    for i in range(200):
        # print('test %d:%d'%(i, len(trainDataArr)))
        print('test %d:%d' % (i, 200))
        # 读取测试集当前测试样本的向量
        x = testDataMat[i]
        # 获取预测的标记
        y = getClosest(trainDataMat, trainLabelMat, x, topK)
        # 如果预测标记与实际标记不符，错误值计数加1
        if y != testLabelMat[i]:
            errorCnt += 1

    # 返回正确率
    return 1 - (errorCnt / 200)


def draw_fun1(trainData, trainLabel, testData, testLabel):
    k_num = 7
    plt.title("test topK")
    plt.xlabel("k ")
    plt.ylabel("accuracy")
    y = np.zeros(k_num)
    for x in range(k_num):
        accur = model_test(trainData, trainLabel, testData, testLabel, x)
        y[x] = accur * 100

    x = np.arange(1, k_num+1)
    plt.scatter(x[1:], y[1:])
    # 打印正确率
    # print('accuracy is:%d' % (accur * 100), '%')

    plt.show()


if __name__ == "__main__":
    start = time.time()

    # 获取训练集
    trainData, trainLabel = loadData('../mnist/mnist_train.csv')
    # 获取测试集
    testData, testLabel = loadData('../mnist/mnist_test.csv')
    # 画出k值对应正确率的散点图
    draw_fun1(trainData, trainLabel, testData, testLabel)

    end = time.time()
    # 显示花费时间
    print('cost time:', end - start)
