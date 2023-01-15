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
    这个代码是一个光谱分析的程序，它主要包含了光谱预处理、光谱波长筛选、
    聚类分析、定量分析和定性分析等功能。

    首先，通过调用Preprocessing模块中的Preprocessing函数进行光谱预处理。
    
    然后，通过调用WaveSelect模块中的SpctrumFeatureSelcet函数进行光谱波
    长筛选。
    
    接着，通过调用Clustering模块中的Cluster函数进行聚类分析。
    
    再次，通过调用Regression模块中的QuantitativeAnalysis函数进行定量
    分析，通过调用Classification模块中的QualitativeAnalysis函数进行
    定性分析。
    
    最后，程序还提供了两个函数 SpectralClusterAnalysis 和 
    SpectralQuantitativeAnalysis, 分别对光谱聚类分析和光谱定量分析
    进行了封装。
"""

import numpy as np
from DataLoad.DataLoad import SetSplit, LoadNirtest
from Preprocessing.Preprocessing import Preprocessing
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet
from Plot.SpectrumPlot import plotspc
# from Plot.SpectrumPlot import ClusterPlot
from Simcalculation.SimCa import Simcalculation
from Clustering.Cluster import Cluster
from Regression.Rgs import QuantitativeAnalysis
from Classification.Cls import QualitativeAnalysis


# 光谱聚类分析
def SpectralClusterAnalysis(data, label, ProcessMethods, FslecetedMethods, ClusterMethods):
    """
     :param data: shape (n_samples, n_features), 光谱数据
     :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
     :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
     :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
     :param ClusterMethods : string, 聚类的方法，提供Kmeans聚类、FCM聚类
     :return: Clusterlabels: 返回的隶属矩阵

     """
    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, _ = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    Clusterlabels = Cluster(ClusterMethods, FeatrueData)
    # ClusterPlot(data, Clusterlabels)
    return Clustebrlabels


# 光谱定量分析
def SpectralQuantitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods, model):
    """
    :param data: shape (n_samples, n_features), 光谱数据
    :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
    :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
    :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
    :param SetSplitMethods : string, 划分数据集的方法, 提供随机划分、KS划分、SPXY划分
    :param model : string, 定量分析模型, 包括ANN、PLS、SVR、ELM、CNN、SAE等，后续会不断补充完整
    :return: Rmse: float, Rmse回归误差评估指标
             R2: float, 回归拟合,
             Mae: float, Mae回归误差评估指标
    """
    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, labels = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, FeatrueData, labels, test_size=0.2, randomseed=123)
    Rmse, R2, Mae = QuantitativeAnalysis(model, X_train, X_test, y_train, y_test)
    return Rmse, R2, Mae


# 光谱定性分析
def SpectralQualitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods, model):
    """
    :param data: shape (n_samples, n_features), 光谱数据
    :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
    :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
    :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
    :param SetSplitMethods : string, 划分数据集的方法, 提供随机划分、KS划分、SPXY划分
    :param model : string, 定性分析模型, 包括ANN、PLS_DA、SVM、RF、CNN、SAE等，后续会不断补充完整
    :return: acc： float, 分类准确率
    """

    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, labels = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, FeatrueData, labels, test_size=0.2, randomseed=123)
    acc = QualitativeAnalysis(model, X_train, X_test, y_train, y_test)

    return acc
