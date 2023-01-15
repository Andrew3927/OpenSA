"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

"""
    这段代码主要实现了一个函数 QuantitativeAnalysis，这个函数
    根据传入的 model 参数来进行定量分析。model 可以是 "Pls"，"ANN"，
    "SVR"，"ELM" 或 "CNN"。如果 model 是 "Pls"，就调用 Pls 函数进
    行 PLS 回归分析，如果是 "ANN"，就调用 Anngression 函数进行人工神
    经网络分析，如果是 "SVR"，就调用 Svregression 函数进行支持向量回
    归分析，如果是"ELM"，就调用 ELM 函数进行极限学习机回归分析，如果是 
    "CNN"，就调用 CNNTrain 函数进行卷积神经网络分析。每种分析方法都返回
    三个结果，分别是均方根误差（Rmse），决定系数（R2）和平均绝对误差（Mae）。
"""

from Regression.ClassicRgs import Pls, Anngression, Svregression, ELM
from Regression.CNN import CNNTrain


def QuantitativeAnalysis(model, X_train, X_test, y_train, y_test):
    """
    回归模型定量分析函数，输入模型类型和训练、测试数据集，返回RMSE, R2, MAE指标。
    :param model: str 模型类型，可选值为 "Pls"、"ANN"、"SVR"、"ELM"、"CNN"。
    :param X_train: numpy array，shape (n_samples, n_features) 训练数据集，n_samples是样本数量，n_features是样本特征数量
    :param X_test: numpy array，shape (n_samples，n_features) 测试数据集，n_samples 是样本数量，n_features 是样本特征数量。
    :param y_train: numpy array, shape (n_sample, ) 训练数据标签，n_sample 是样本数量。
    :param y_test: numpy array, shape (n_sample, ) 测试数据标签，n_sample 是样本数量。
    :return:
        Rmse : float
            RMSE指标。
        R2 : float
            R2指标。
        Mae : float
            MAE指标。
    """
    Rmse = 0
    R2 = 0
    Mae = 0
    if model == "Pls":
        Rmse, R2, Mae = Pls(X_train, X_test, y_train, y_test)
    elif model == "ANN":
        # Rmse, R2, Mae = Anngression(X_train, X_test, y_train, y_test)
        Rmse, R2, Mae = Anngression(X_train, X_test, y_train, y_test)
    elif model == "SVR":
        Rmse, R2, Mae = Svregression(X_train, X_test, y_train, y_test)
    elif model == "ELM":
        Rmse, R2, Mae = ELM(X_train, X_test, y_train, y_test)
    elif model == "CNN":
        Rmse, R2, Mae = CNNTrain("AlexNet", X_train, X_test, y_train, y_test, 5)
    else:
        print("no this model of QuantitativeAnalysis")

    return Rmse, R2, Mae
