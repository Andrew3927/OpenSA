"""
    这些代码主要实现了使用多种回归模型，如PLS回归、SVR、MLP回归和ELM回归，
    对给定的训练数据和测试数据进行预测，并使用 ModelRgsevaluate 函数
    评估预测结果。其中 PLs，Svregression，Anngression和ELM分别对应使
    用PLS回归，SVR，MLP回归和ELM回归对数据进行预测。返回的 Rmse, R2, Mae
    分别表示均方根误差，R2分数，平均绝对误差。
"""

from Regression.ClassicRgs import Pls, Anngression, Svregression, ELM
from Regression.CNN import CNNTrain


def QuantitativeAnalysis(model, X_train, X_test, y_train, y_test, EPOCH, acti, c_num, loss, optim):
    if model == "Pls":
        Rmse, R2, Mae = Pls(X_train, X_test, y_train, y_test)
    elif model == "ANN":
        Rmse, R2, Mae = Anngression(X_train, X_test, y_train, y_test)
    elif model == "SVR":
        Rmse, R2, Mae = Svregression(X_train, X_test, y_train, y_test)
    elif model == "ELM":
        Rmse, R2, Mae = ELM(X_train, X_test, y_train, y_test)
    elif model[0:3] == "CNN":
        Rmse, R2, Mae = CNNTrain(model[4:], X_train, X_test, y_train, y_test, EPOCH, acti, c_num, loss, optim)
    else:
        print("no this model of QuantitativeAnalysis")

    return Rmse, R2, Mae
