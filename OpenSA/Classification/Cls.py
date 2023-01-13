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
    Zach Chan
    这段Python代码实现了一个名为QualitativeAnalysis的函数，该函数的目的是根据
    传入的模型名称，调用对应的训练和评估函数，并返回评估结果。

    代码中定义了一些训练和评估函数，如ANN,SVM,PLS_DA,RF,CNN,SAE。这些函数在被
    调用时会对训练数据进行训练，对测试数据进行预测并返回准确率。

    QualitativeAnalysis函数首先检查传入的模型名称，并根据模型名称调用对应的训练
    和评估函数，最后返回评估结果。
"""

from OpenSA.Classification.ClassicCls import ANN, SVM, PLS_DA, RF
from OpenSA.Classification.CNN import CNN
from OpenSA.Classification.SAE import SAE

def  QualitativeAnalysis(model, X_train, X_test, y_train, y_test):

    if model == "PLS_DA":
        acc = PLS_DA(X_train, X_test, y_train, y_test)
    elif model == "ANN":
        acc = ANN(X_train, X_test, y_train, y_test)
    elif model == "SVM":
        acc = SVM(X_train, X_test, y_train, y_test)
    elif model == "RF":
        acc = RF(X_train, X_test, y_train, y_test)
    elif model == "CNN":
        acc = CNN(X_train, X_test, y_train, y_test, 16, 160, 4)
    elif model == "SAE":
        acc = SAE(X_train, X_test, y_train, y_test)
    else:
        print("no this model of QuantitativeAnalysis")

    return acc