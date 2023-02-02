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

from DataLoad.DataLoad import SetSplit, LoadNirtest
from Preprocessing.Preprocessing import Preprocessing
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet
from Regression.Rgs import QuantitativeAnalysis


# 光谱定量分析
def SpectralQuantitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods, model, EPOCH, acti,
                                 c_num, loss, optim):
    """
    :param data: shape (n_samples, n_features), 光谱数据
    :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
    :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
    :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
    :param SetSplitMethods: string, 划分数据集的方法, 提供随机划分、KS划分、SPXY划分
    :param model: string, 定量分析模型, 包括ANN、Pls、SVR、ELM、CNN等，后续会不断补充完整
    :return: Rmse: float, Rmse回归误差评估指标
             R2: float, 回归拟合,
             Mae: float, Mae回归误差评估指标
    """
    # nirplot_assign(data,600,1898,2)
    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, labels = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, FeatrueData, labels, test_size=0.2, randomseed=123)
    Rmse, R2, Mae = QuantitativeAnalysis(model, X_train, X_test, y_train, y_test, EPOCH, acti, c_num, loss, optim)

    return Rmse, R2, Mae


if __name__ == '__main__':
    # 载入原始数据并可视化
    data2, label2 = LoadNirtest('Rgs')
    # 光谱定量分析演示
    # 示意1: 预处理算法:MSC , 波长筛选算法: Uve, 数据集划分:KS, 定性分量模型: SVR
    # 这里我改了参数，10,'relu',5,'MSE','Adam'，10 是epoch，relu是激活函数，可以选。5是CNN有5层，MSE是损失函数，Adam是优化，这些可以到CNN.p文件看一下有什么选择。
    RMSE, R2, MAE = SpectralQuantitativeAnalysis(data=data2, label=label2, ProcessMethods="MMS", 
                                                 FslecetedMethods="None", SetSplitMethods="random", 
                                                 model="CNN_vgg", EPOCH=3, acti='relu', c_num=5,
                                                 loss='MSE', optim='Adam')
    print("The RMSE:{} R2:{}, MAE:{} of result!".format(RMSE, R2, MAE))
