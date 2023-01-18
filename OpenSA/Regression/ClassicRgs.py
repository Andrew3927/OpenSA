from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import hpelm
from sklearn.svm import SVR
from Evaluate.RgsEvaluate import ModelRgsevaluate
from sklearn.base import BaseEstimator

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
    这些代码主要实现了使用多种回归模型，如PLS回归、SVR、MLP回归和ELM回归，
    对给定的训练数据和测试数据进行预测，并使用 ModelRgsevaluate 函数
    评估预测结果。其中 PLs，Svregression，Anngression和ELM分别对应使
    用PLS回归，SVR，MLP回归和ELM回归对数据进行预测。返回的 Rmse, R2, Mae 
    分别表示均方根误差，R2分数，平均绝对误差。
"""


def Pls(X_train, X_test, y_train, y_test):
    """
    对于一维光谱检测而言，n_components取值的选择取决于具体的数据集和应
    用场景。一般来说，n_components应该取值在数据维度数之内。在实际使用
    中，我们可以通过交叉验证来选择最优的n_components取值。或者可以先从
    数据维度数的一半开始尝试，如果效果不够理想，再逐渐增加或减少。

    对于PLS回归，我们需要对n_components进行调参，这可以通过交叉验证来
    实现。

    :param X_train: 训练集的特征数据
    :param X_test: 测试集的特征数据
    :param y_train: 训练集的标签数据
    :param y_test: 测试集的标签数据
    :return: Rmse, R2, Mae三个评估指标的值
    """

    # 交叉验证参数
    param_grid = {'n_components': [2, 4, 6, 8, 10, 12, 14, 18]}

    # 构建PSL回归模型
    pls = PLSRegression()
    # model = PLSRegression(n_components=8)

    # 使用 GridSearchCV进行交叉验证
    pls_grid = GridSearchCV(pls, param_grid, cv=5)

    #  拟合模型
    pls_grid.fit(X_train, y_train)
    # model.fit(X_train, y_train)

    # 输出最优参数
    print("最优参数：\n", pls_grid.best_params_, "\n")

    # predict the values
    y_pred = pls_grid.predict(X_test)
    # y_pred = model.predict(X_test)

    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae


def Svregression(X_train, X_test, y_train, y_test):
    """
    对于SVR，我们需要对C，kernel和gamma三个参数进行调参。
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    # 交叉验证参数
    param_grid = {'C': [0.1, 1, 2, 10],
                  'kernel': ['linear', 'rbf'],
                  'gamma': [1e-07, 0.1, 1, 10]}

    # 构建SVR模型
    svr = SVR()
    # model = SVR(C=2, gamma=1e-07, kernel='linear')

    # 使用GridSearchCV进行交叉验证
    svr_grid = GridSearchCV(svr, param_grid, cv=5)

    # 拟合模型
    svr_grid.fit(X_train, y_train)
    # model.fit(X_train, y_train)

    # 输出最优参数
    print("最优参数：", svr_grid.best_params_, "\n")

    # predict the values
    y_pred = svr_grid.predict(X_test)
    # y_pred = model.predict(X_test)
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae


def Anngression(X_train, X_test, y_train, y_test):
    """
    对于MLP回归，我们需要对隐藏层神经元数量，激活函数和学习率三个参数进行调参。
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """

    # 交叉验证参数
    print("开始进行交叉验证参数……")
    param_grid = {'hidden_layer_sizes': [(100,), (200,), (300,)],
                  'activation': ['relu', 'logistic'],
                  'learning_rate_init': [0.001, 0.01, 0.1]}
    print("交叉验证参数完成\n")

    # 构建MLP回归模型
    print("构建MLP回归模型……")
    MAX_ITER = 600  # MAX_ITER = 400
    mlp = MLPRegressor(
        solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='constant', power_t=0.5, max_iter=MAX_ITER, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # model = MLPRegressor(
    #     hidden_layer_sizes=(20, 20), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    #     learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=MAX_ITER, shuffle=True,
    #     random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    #     early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    print("完成MLP回归模型构建\n")

    # 使用GridSearchCV进行交叉验证
    print("使用GridSearchCV进行交叉验证……")
    mlp_grid = GridSearchCV(mlp, param_grid, cv=5)
    print("GridSearchCV进行交叉验证完成\n")

    # 拟合模型
    print("开始拟合模型……")
    mlp_grid.fit(X_train, y_train)
    # model.fit(X_train, y_train)
    print("模型拟合完成\n")

    # 输出最优参数
    print("最优参数：", mlp_grid.best_params_, "\n")

    print("开始模型预测")
    # predict the values
    y_pred = mlp_grid.predict(X_test)
    # y_pred = model.predict(X_test)
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae


def ELM(X_train, X_test, y_train, y_test):
    """
    对于ELM回归，我们需要对隐藏层神经元数量和激活函数两个参数进行调参。

    为了实现自动搜索参数的功能，我们需要将参数搜索的过程和模型训练的过程
    分开来，并使用GridSearchCV进行交叉验证。

    在使用 GridSearchCV 方法时，需要先定义参数网格，然后使用 fit 方法进行
    搜索。最后，可以使用 best_params_ 属性获取最优参数。

    完整代码中在获取最优参数后，将其传入 add_neurons 方法中进行模型训练。
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    # 定义参数网格
    param_grid = {'hidden_neurons': [10, 20, 30], 'activation': ['sigm', 'rbf']}
    print("完成交叉验证参数设置\n")

    # 创建ELM模型
    # print("构建ELM回归模型……")
    # elm = hpelm.ELM(X_train.shape[1], 1)
    # print("完成ELM回归模型构建\n")

    print("初始化ELM模型……")
    elm_regressor = ELMRegressor()
    elm_regressor.setInputOutput(X_train, y_train)
    print("完成ELM模型初始化\n")

    # 创建GridSearchCV对象
    print("使用GridSearchCV进行自动搜索（交叉验证）……")
    grid_search = GridSearchCV(elm_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
    print("完成GridSearchCV进行自动搜索（交叉验证）\n")

    # 训练并返回最优参数
    print("训练并返回最优参数……")
    grid_search.fit(X_train, y_train)
    best_param = grid_search.best_params_
    print("完成训练并返回最优参数\n")

    # 输出最优参数
    print("最优参数：", best_param, "\n")

    # 用最优参数重新训练模型
    print("用最优参数重新训练模型……")

    elm2 = ELMRegressor()
    elm2.setInputOutput(X_train, y_train)
    elm2.add_neurons(best_param['hidden_neurons'], best_param['activation'])
    # elm.add_neurons(best_param['hidden_neurons'], best_param['activation'])
    elm2.train(X_train, y_train)
    # elm.train(X_train, y_train)
    y_pred = elm2.predict(X_test)
    print("模型训练完成\n")

    # 评估模型
    print("评估模型……")
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae


class ELMRegressor(BaseEstimator):
    """
    由于使用到了自定义的 hpelm 库中的 ELM 模型，而 GridSearchCV 默认只能识别 sklearn 中的模型。
    因此需要在使用 GridSearchCV 时将自定义的 ELM 模型封装到一个可以被 sklearn 识别的模型中。
    """
    elm_ = None
    def __init__(self, hidden_neurons=20, activation='sigm'):
        setattr(self, 'hidden_neurons', hidden_neurons)
        setattr(self, 'activation', activation)

    def fit(self, X, y):
        setattr(self, 'elm_', hpelm.ELM(X.shape[1], 1))
        self.elm_.add_neurons(self.hidden_neurons, self.activation)
        self.elm_.train(X, y)

    def setInputOutput(self, X, y):
        setattr(self, 'elm_', hpelm.ELM(X.shape[1], 1))


    def predict(self, X):
        return self.elm_.predict(X)


    def add_neurons(self, hidden_neurons, activation):
        # self.elm_.nnet.reset()
        self.elm_.add_neurons(hidden_neurons, activation)


    def train(self, X, y):
        self.elm_.train(X, y)
