"""
    这段代码主要实现了一个Lar函数，它使用Lars回归算法来选择输入数据中最重
    要的特征变量。Lars算法是一种迭代线性回归算法，它可以选择出最重要的变量，
    并且不需要输入变量的数量。参数X是预测变量矩阵，参数y是标签，参数nums是
    选择的特征点的数目，默认为40。函数返回选择变量集的索引。
"""


from sklearn import linear_model
import numpy as np

def Lar(X, y, nums=40):
    '''
           X : 预测变量矩阵
           y ：标签
           nums : 选择的特征点的数目，默认为40
           return ：选择变量集的索引
    '''
    Lars = linear_model.Lars()
    Lars.fit(X, y)
    corflist = np.abs(Lars.coef_)

    corf = np.asarray(corflist)
    SpectrumList = corf.argsort()[-1:-(nums+1):-1]
    SpectrumList = np.sort(SpectrumList)

    return SpectrumList