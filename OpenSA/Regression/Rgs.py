"""
    这些代码主要实现了使用多种回归模型，如PLS回归、SVR、MLP回归和ELM回归，
    对给定的训练数据和测试数据进行预测，并使用 ModelRgsevaluate 函数
    评估预测结果。其中 PLs，Svregression，Anngression和ELM分别对应使
    用PLS回归，SVR，MLP回归和ELM回归对数据进行预测。返回的 Rmse, R2, Mae
    分别表示均方根误差，R2分数，平均绝对误差。
"""
import torch.nn
from Regression.ClassicRgs import Pls, Anngression, Svregression, ELM
from Regression.CNN import *
from Regression.CnnModel import *
import sys
import ray
import ray.tune as tune
import multiprocessing
from ColorCodePrint.color_print import *

tradic_net_dict = {
    'Pls': Pls,
    'ANN': Anngression,
    'SVR': Svregression,
    'ELM': ELM
}

# 使用字典映射 network结构
NET_DICT = {
    'vgg': AlexNet,
    'inception': DeepSpectra,
    'Resnet': Resnet,
    'DenseNet': DenseNet,
}

# 使用字典映射调用 loss函数
loss_dict = {
    'MSE': nn.MSELoss(),
    'L1': nn.L1Loss(),
    'CrossEntropy': nn.CrossEntropyLoss(ignore_index=-100),
    'Poisson': nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-08),
    # 'KLDiv': nn.KLDivLoss(reduction='batchmean'),
}


def QuantitativeAnalysis(model, X_train, X_test, y_train, y_test, EPOCH, acti, cnn_depth,
                         loss, optim, is_autoTune, autoHyperConfig):
    # 使用传统的推理模型
    if model in tradic_net_dict:
        CL_green_print("Using " + model + " for training ...")
        print("Start training")
        # Rmse, R2, Mae = tradic_net_dict[model](X_train, X_test, y_train, y_test)

    elif model[0:3] == "CNN" and model[4:] in NET_DICT:
        network = NET_DICT[model[4:]]

        # 初始化网络
        network = network(acti, cnn_depth)

        # 使用字典映射调用优化器
        OPTIM_DICT = {
            'Adam': torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=0.001),
            'SGD': torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9),
            # 接下来的激活函数 torch并不支持。
            # 'Adagrad': torch.optim.Adagrad(network.parameters(), lr=0.01),
            'Adadelta': torch.optim.Adadelta(network.parameters()),
            'RMSprop': torch.optim.RMSprop(network.parameters(), lr=0.01, alpha=0.99),
            'Adamax': torch.optim.Adamax(network.parameters(), lr=0.002, betas=(0.9, 0.999)),
            # 'LBFGS': torch.optim.LBFGS(network.parameters(), lr=0.01)
        }
        # 设置优化器函数
        optim_func = OPTIM_DICT[optim]
        # 设置 loss 函数
        loss_func = loss_dict[loss]
        # 打印配置参数
        __printConfiguration(EPOCH=EPOCH, acti_func=acti, cnn_depth=cnn_depth, loss=loss,
                             optim=optim, is_autoTune=is_autoTune)
        # 开始训练
        train(network, X_train, X_test, y_train, y_test, EPOCH, loss_func,
              optim_func)
    else:
        CL_red_print("model=" + model + " hasn't been implemented in this project yet.")
        # Given unsupported parameters, break the program
        sys.exit()
