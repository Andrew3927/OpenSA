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
    这个代码实现了两种划分数据集的方式：随机划分和基于SPXY算法的划分。
    首先实现了一个随机划分函数，利用sklearn中的train_test_split函数实
    现；其次实现了一个基于SPXY算法的划分函数。这个算法是基于数据和标签之
    间的相似度来划分数据集。
"""



from sklearn.model_selection import train_test_split
import numpy as np

#随机划分数据集
def random(data, label, test_ratio=0.2, random_state=123):
    """
    :param data: shape (n_samples, n_features)
    :param label: shape (n_sample, )
    :param test_size: the ratio of test_size, default: 0.2
    :param random_state: the randomseed, default: 123
    :return: X_train :(n_samples, n_features)
             X_test: (n_samples, n_features)
             y_train: (n_sample, )
             y_test: (n_sample, )
    """

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_ratio, random_state=random_state)

    return X_train, X_test, y_train, y_test

#利用SPXY算法划分数据集
def spxy(data, label, test_size=0.2):
    """
    :param data: shape (n_samples, n_features)
    :param label: shape (n_sample, )
    :param test_size: the ratio of test_size, default: 0.2
    :return: X_train :(n_samples, n_features)
             X_test: (n_samples, n_features)
             y_train: (n_sample, )
             y_test: (n_sample, )
    """
    x_backup = data
    y_backup = label
    M = data.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    label = (label - np.mean(label)) / np.std(label)
    D = np.zeros((M, M))
    Dy = np.zeros((M, M))

    for i in range(M - 1):
        xa = data[i, :]
        ya = label[i]
        for j in range((i + 1), M):
            xb = data[j, :]
            yb = label[j]
            D[i, j] = np.linalg.norm(xa - xb)
            Dy[i, j] = np.linalg.norm(ya - yb)

    Dmax = np.max(D)
    Dymax = np.max(Dy)
    D = D / Dmax + Dy / Dymax

    maxD = D.max(axis=0)
    index_row = D.argmax(axis=0)
    index_column = maxD.argmax()

    m = np.zeros(N)
    m[0] = index_row[index_column]
    m[1] = index_column
    m = m.astype(int)

    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros(M - i)
        for j in range(M - i):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(data.shape[0]), m)

    X_train = data[m, :]
    y_train = y_backup[m]
    X_test = data[m_complement, :]
    y_test = y_backup[m_complement]

    return X_train, X_test, y_train, y_test

#利用kennard-stone算法划分数据集
def ks(data, label, test_size=0.2):
    """
    :param data: shape (n_samples, n_features)
    :param label: shape (n_sample, )
    :param test_size: the ratio of test_size, default: 0.2
    :return: X_train: (n_samples, n_features)
             X_test: (n_samples, n_features)
             y_train: (n_sample, )
             y_test: (n_sample, )
    """
    M = data.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    D = np.zeros((M, M))

    for i in range((M - 1)):
        xa = data[i, :]
        for j in range((i + 1), M):
            xb = data[j, :]
            D[i, j] = np.linalg.norm(xa - xb)

    maxD = np.max(D, axis=0)
    index_row = np.argmax(D, axis=0)
    index_column = np.argmax(maxD)

    m = np.zeros(N)
    m[0] = np.array(index_row[index_column])
    m[1] = np.array(index_column)
    m = m.astype(int)
    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros((M - i))
        for j in range((M - i)):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(data.shape[0]), m)

    X_train = data[m, :]
    y_train = label[m]
    X_test = data[m_complement, :]
    y_test = label[m_complement]

    return X_train, X_test, y_train, y_test

# 分别使用一个回归、一个分类的公开数据集做为example
def LoadNirtest(type):

    if type == "Rgs":
        CDataPath1 = './/Data//Rgs//Cdata1.csv'
        VDataPath1 = './/Data//Rgs//Vdata1.csv'
        TDataPath1 = './/Data//Rgs//Tdata1.csv'

        Cdata1 = np.loadtxt(open(CDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        Vdata1 = np.loadtxt(open(VDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        Tdata1 = np.loadtxt(open(TDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)

        Nirdata1 = np.concatenate((Cdata1, Vdata1))
        Nirdata = np.concatenate((Nirdata1, Tdata1))
        data = Nirdata[:, :-4]
        label = Nirdata[:, -1]

    elif type == "Cls":
        path = './/Data//Cls//table.csv'
        Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data = Nirdata[:, :-1]
        label = Nirdata[:, -1]

    return data, label

def SetSplit(method, data, label, test_size=0.2, randomseed=123):
    """
    数据集划分函数，根据不同的划分方式进行划分。
    :param method: str
        划分方式，可选值为 "random"、"spxy"、"ks"。
    :param data: numpy array, shape (n_samples, n_features)
        输入的数据，n_samples 是样本数量，n_features 是样本特征数量。
    :param label: numpy array, shape (n_sample, )
        数据标签，n_sample 是样本数量。
    :param test_size: float, optional (default=0.2)
        测试数据集所占比例。
    :param randomseed: int, optional (default=123)
        随机种子。
    :return:
        X_train : numpy array, shape (n_samples, n_features)
        训练数据集。
        X_test : numpy array, shape (n_samples, n_features)
            测试数据集。
        y_train : numpy array, shape (n_sample, )
            训练数据标签。
        y_test : numpy array, shape (n_sample, )
            测试数据标签。
    """
    if method == "random":
        X_train, X_test, y_train, y_test = random(data, label, test_size, randomseed)
    elif method == "spxy":
        X_train, X_test, y_train, y_test = spxy(data, label, test_size)
    elif method == "ks":
        X_train, X_test, y_train, y_test = ks(data, label, test_size)
    else:
        print("no this  method of split dataset! ")

    return X_train, X_test, y_train, y_test
