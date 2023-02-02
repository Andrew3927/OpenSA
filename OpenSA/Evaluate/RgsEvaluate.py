"""
    这些代码主要是实现了几种不同的数据预处理方法（MinMaxScaler、
    Normalizer、StandardScaler）和一种神经网络模型（MLPRegressor）
    的使用。其中主要包括了对于预测结果进行评估的函数，在评估中主要用到了
    均方根误差、R平方和平均绝对误差这三种指标来评估预测结果的准确性。
"""
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neural_network import MLPRegressor
import numpy as np


def ModelRgsevaluate(y_pred, y_true):

    mse = mean_squared_error(y_true,y_pred)
    R2  = r2_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)

    return np.sqrt(mse), R2, mae

def ModelRgsevaluatePro(y_pred, y_true, yscale):

    yscaler = yscale
    y_true = yscaler.inverse_transform(y_true)
    y_pred = yscaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_true,y_pred)
    R2  = r2_score(y_true,y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return np.sqrt(mse), R2, mae