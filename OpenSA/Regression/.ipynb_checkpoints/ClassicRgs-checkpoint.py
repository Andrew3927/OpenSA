
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from Evaluate.RgsEvaluate import ModelRgsevaluate
from sklearn.base import BaseEstimator
import numpy as np 
from Plot.plot import nirplot_eva
from sklearn.svm import SVR
from Evaluate.RgsEvaluate import ModelRgsevaluate
from parameter.parameter import Grid
from sklearn.model_selection import GridSearchCV
import hpelm

def Pls(X_train, X_test, y_train, y_test):
    
    model='pls'
    param_grid = {'n_components': np.arange(1, 40)}
    Rmse, R2, Mae = Grid(X_train, X_test, y_train, y_test,model, param_grid)
    return Rmse, R2, Mae

def Svregression(X_train, X_test, y_train, y_test):

    model='svr'
    param_grid = {'C': [0.1, 1, 2, 10],
                  'kernel': ['linear', 'rbf'],
                  'gamma': [1e-07, 0.1, 1, 10]}
    Rmse, R2, Mae = Grid(X_train, X_test, y_train, y_test,model, param_grid)
    return Rmse, R2, Mae

def Anngression(X_train, X_test, y_train, y_test):

    model='ann'
    param_grid = {'hidden_layer_sizes': [(100,), (200,), (300,)],
                  'activation': ['relu', 'logistic'],
                  'learning_rate_init': [0.001, 0.01, 0.1]}

    Rmse, R2, Mae = Grid(X_train, X_test, y_train, y_test,model, param_grid)

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

