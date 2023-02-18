"""
    这些代码主要实现了使用多种回归模型，如PLS回归、SVR、MLP回归和ELM回归，
    对给定的训练数据和测试数据进行预测，并使用 ModelRgsevaluate 函数
    评估预测结果。其中 PLs，Svregression，Anngression和ELM分别对应使
    用PLS回归，SVR，MLP回归和ELM回归对数据进行预测。返回的 Rmse, R2, Mae
    分别表示均方根误差，R2分数，平均绝对误差。
"""
import torch.nn
from torch import optim
from Regression.ClassicRgs import Pls, Anngression, Svregression, ELM
from Regression.CNN import *
from Regression.CnnModel import *
import sys
import ray
import ray.tune as tune
import multiprocessing
from ColorCodePrint.color_print import *
import CustomizedLossFunc.functional as custLossFunc
from functools import partial

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
LOSS_DICT = {
    'MSE': nn.MSELoss(),
    'L1': nn.L1Loss(),
    'CrossEntropy': nn.CrossEntropyLoss(ignore_index=-100),
    'Poisson': nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-08),
    # 'KLDiv': nn.KLDivLoss(reduction='batchmean'),
    # todo: make sure these functiosn can be add to GPU device
    "SmoothL1Loss": nn.SmoothL1Loss(),
    "MeanLoss": custLossFunc.MeanLoss(),
    "QuantileLoss": custLossFunc.QuantileLoss(quantile=.5)
}

# 使用字典映射调用优化器
OPTIM_DICT = {
    # new added optimizer
    'Adadelta': partial(optim.Adadelta, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, foreach=None,
                        maximize=False),
    'Adagrad': partial(optim.Adagrad, lr=0.01),
    'Adam': partial(optim.Adam, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False,
                    foreach=None, maximize=False, capturable=False, differentiable=False, fused=False),
    'AdamW': partial(optim.AdamW, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                     amsgrad=False, maximize=False, foreach=None, capturable=False),

#     'SparseAdam': partial(optim.SparseAdam, lr=0.001, betas=(0.9, 0.999), eps=1e-08, maximize=False),

    'Adamax': partial(optim.Adamax, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, foreach=None,
                      maximize=False),

    'ASGD': partial(optim.ASGD, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0, foreach=None,
                    maximize=False),

    'LBFGS': partial(optim.LBFGS, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07,
                     tolerance_change=1e-09, history_size=100, line_search_fn=None),

    'NAdam': partial(optim.NAdam, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004,
                     foreach=None),

    'RAdam': partial(optim.RAdam, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, foreach=None),

    'RMSprop': partial(optim.RMSprop, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                       centered=False, foreach=None, maximize=False, differentiable=False),

    'SGD': partial(optim.SGD, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False,
                   foreach=None, differentiable=False),

}


def QuantitativeAnalysis(model, X_train, X_test, y_train, y_test, EPOCH, acti, cnn_depth,
                         loss, optim):
    # 使用传统的推理模型
    if model in tradic_net_dict:
        CL_green_print("Using " + model + " for training ...")
        print("Start training")
        # Rmse, R2, Mae = tradic_net_dict[model](X_train, X_test, y_train, y_test)

    elif model[0:3] == "CNN" and model[4:] in NET_DICT:
        network = NET_DICT[model[4:]]

        # 初始化网络
        network = network(acti, cnn_depth)

        # 设置优化器函数
        optim_func = OPTIM_DICT[optim](params=network.parameters())
        # 设置 loss 函数
        loss_func = LOSS_DICT[loss]
        # 打印配置参数
        printConfiguration(EPOCH=EPOCH, acti_func=acti, cnn_depth=cnn_depth, loss=loss,
                           optim=optim)
        # 开始训练
        train(network, X_train, X_test, y_train, y_test, EPOCH, loss_func,
              optim_func)
    else:
        CL_red_print("model=" + model + " hasn't been implemented in this project yet.")
        # Given unsupported parameters, break the program
        sys.exit()
