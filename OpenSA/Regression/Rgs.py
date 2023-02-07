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
    # Rmse, R2, Mae

    # if model == "Pls" or model == "ANN" or model == "SVR" or model == "ELM":
    if model in tradic_net_dict:
        print("\033[32m" + "Using " + model + " for training ..." + "\033[0m")
        print("Start training")
        Rmse, R2, Mae = tradic_net_dict[model](X_train, X_test, y_train, y_test)
    elif model[0:3] == "CNN" and model[4:] in NET_DICT:
        network = NET_DICT[model[4:]]
        if is_autoTune:
            print("\033[32m" + "Using " + model + " for training ..." + "\033[0m")

            # todo: print出正在使用自动超参，所用时间可能比较长。
            print("\033[1;32m正在使用自动超参，所用时间可能比较长。\033[0m")

            # 获得本设备CPU最大核心数量。
            #   注意：如果出现可用内存空间不足，请自行调整`max_cpu_cores`以减少CPU核心数的使用！
            max_cpu_cores = multiprocessing.cpu_count()

            # 获得本设备拥有的最大GPU数量
            #   注意：如果出现可用内存空间不足，请自行调整`max_gpu_num`以减少GPU数量的使用！
            max_gpu_num = torch.cuda.device_count();

            print(
                "Available CPUs for cpu training: " + max_cpu_cores + ",\nAvailable GPUs for gpu training: " + max_gpu_num)

            # 开始超参数搜索
            # todo: 目前正在将CNNTrain转换成一个类 ……

            # 因为使用了AutoTune, EOPCH, loss，optim的值都设置在自动超参的config中，因此在此默认使用None。
            cnnTrain = CNNTrain(network, X_train, X_test, y_train, y_test, None, None, None)
            tune.run(cnnTrain.train, config=autoHyperConfig,
                     resources_per_trial={"cpu": max_cpu_cores, "gpu": max_gpu_num})

            # 关闭Ray
            ray.shutdown()
        else:
            # 初始化网络
            network = network(acti, cnn_depth)

            # 使用字典映射调用优化器
            optim_dict = {
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
            optim_func = optim_dict[optim]
            # 设置 loss 函数
            loss_func = loss_dict[loss]
            # 打印配置参数
            __printConfiguration(EPOCH=EPOCH, acti_func=acti, cnn_depth=cnn_depth, loss=loss,
                                 optim=optim, is_autoTune=is_autoTune)

            # 开始训练
            train(network, X_train, X_test, y_train, y_test, EPOCH, loss_func,
                           optim_func)
            # Rmse, R2, Mae = CNNTrain(network, X_train, X_test,
            #                          y_train, y_test, EPOCH, loss_func, optim_func)
    else:
        print("model=" + "\033[1;31;40m" + model + "\033[0m" + " hasn't been implemented in this project yet.")
        # given unsupported parameters, break the program
        sys.exit()

    # return Rmse, R2, Mae


def __printConfiguration(EPOCH, acti_func, cnn_depth, loss, optim, is_autoTune):
    print("Training configuration:  " + "EPOCH=" + str(EPOCH) + ", activation function=" +
          acti_func + ", cnn_depth=" + str(cnn_depth) + ", loss function=" + loss + ", optimizer=" +
          optim + ", is_autoTune=" + str(is_autoTune) + "\n")
