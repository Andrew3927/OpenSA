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
import torch
import torch.cuda
import torch.nn as nn
# import torch.nn.functional.py as F
# import torch.optim as optim
# from torch.utils.data import random_split
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler

from DataLoad.DataLoad import SetSplit, LoadNirtest
from Preprocessing.Preprocessing import Preprocessing
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet
from Regression.Rgs import QuantitativeAnalysis
import multiprocessing
import programParameter.stringConfig as STRING_CONFIG
import programParameter.threadConfig as THREAD_CONFIG
from Regression.CNN import ZspPocessnew
from Evaluate.RgsEvaluate import ModelRgsevaluatePro

from functools import partial
import numpy as np
import os

from AutoTune.AutoTuneConfig import autoTuneMain


# 光谱定量分析
def SpectralQuantitativeAnalysis(data, label, ProcessMethods,
                                 FslecetedMethods, SetSplitMethods,
                                 model, EPOCH, acti, cnn_depth, loss, optim):
    """

    :param is_autoTune:
    :param autoHyperConfig:
    :param data: shape (n_samples, n_features), 光谱数据
    :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
    :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
    :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
    :param SetSplitMethods: string, 划分数据集的方法, 提供随机划分、KS划分、SPXY划分
    :param model: string, 定量分析模型, 包括ANN、Pls、SVR、ELM、CNN_vgg, CNN_inception,
                        CNN_Resnet, CNN_DenseNet等，后续会不断补充完整
    :param EPOCH: int, 数据集训练次数
    :param acti: string, 激活函数，提供 relu、lrelu
    :param cnn_depth:
    :param loss: string, loss 函数，提供 MSE、L1、CrossEntropy、Poisson、KLDiv (not supported)、SmoothL1Loss、Mean Loss、
                        QuantileLoss
    :param optim: string, 优化器，提供 Adam、SGD、Adagrad、Adadelta、RMSprop、Adamax、LBFGS
    :return: Rmse: float, Rmse回归误差评估指标
             R2: float, 回归拟合,
             Mae: float, Mae回归误差评估指标
    """
    # nirplot_assign(data,600,1898,2)
    processedData = Preprocessing(ProcessMethods, data)
    featureData, labels = SpctrumFeatureSelcet(FslecetedMethods, processedData, label)
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, featureData, labels, test_size=0.2, randomseed=123)
    QuantitativeAnalysis(model, X_train, X_test, y_train, y_test, EPOCH,
                         acti, cnn_depth, loss, optim)

if __name__ == '__main__':
    # 载入原始数据并可视化
    data2, label2 = LoadNirtest('Rgs')

    # 是否使用自动超参功能
    IS_AUTOTUNE = True
    TRAINING_BATCH_SIZE = 16
    TESTING_BATCH_SIZE = 240

    if IS_AUTOTUNE:
        # 获得本设备CPU最大核心数量。
        #   注意：如果出现可用内存空间不足，请自行调整`max_cpu_cores`以减少CPU核心数的使用！
        max_cpu_cores = multiprocessing.cpu_count()
        print("CPU has " + str(max_cpu_cores) + " cores on this device.")

        # 设置环境变量，解决Ray Tune因为CPU检测不准确而给出的warning。
        os.environ['RAY_USE_MULTIPROCESSING_CPU_COUNT'] = '1'

        # 设置超参数搜索空间
        SEARCH_SPACE = {
            STRING_CONFIG.cnn_depth: tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8]),
            STRING_CONFIG.epochs: tune.grid_search([10, 20, 30]),  # 具体EPOCHS搜参空间请根据模型的收敛情况判断而定
            STRING_CONFIG.lr: tune.grid_search([.001, .01, .1]),
            STRING_CONFIG.momentum: tune.grid_search([.5, .9, .99]),
            STRING_CONFIG.optimizer: tune.grid_search(["Adam", "SGD", "Adadelta", "RMSprop", "Adamax"]),
            STRING_CONFIG.acti_func: tune.choice(["relu", "lrelu"])
        }

        autoTuneMain(num_samples=10, max_num_epochs=10, gpus_per_trial=1, network="CNN_vgg",
                     data=data2, label=label2, ProcessMethods="MMS", FslecetedMethods="None",
                     SetSplitMethods="random", model="CNN_vgg", cnn_depth=5,
                     loss=STRING_CONFIG.MSE, config=SEARCH_SPACE, max_cpu_cores=max_cpu_cores,
                     trainingBatchSize=TRAINING_BATCH_SIZE, testingBatchSize=TESTING_BATCH_SIZE)

    # 传统的模型训练 & CNN模型（不自动超参）
    else:
        # 光谱定量分析演示
        # 示意1: 预处理算法:MSC , 波长筛选算法: Uve, 数据集划分:KS, 定性分量模型: SVR
        # 这里我改了参数，10,'relu',5,'MSE','Adam'，10 是epoch，relu是激活函数，可以选。5是CNN有5层，MSE是损失函数，Adam是优化，这些可以到CNN.p文件看一下有什么选择。
        SpectralQuantitativeAnalysis(data=data2, label=label2,
                                     ProcessMethods="MMS", FslecetedMethods="None",
                                     SetSplitMethods="random",
                                     model="CNN_vgg", EPOCH=3, acti='relu',
                                     cnn_depth=5, loss='MSE', optim='SGD')
