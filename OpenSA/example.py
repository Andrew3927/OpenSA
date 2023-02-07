"""
    这个代码是一个光谱分析的程序，它主要包含了光谱预处理、光谱波长筛选、
    聚类分析、定量分析和定性分析等功能。

    首先，通过调用Preprocessing模块中的Preprocessing函数进行光谱预处理。

    然后，通过调用WaveSelect模块中的SpctrumFeatureSelcet函数进行光谱波
    长筛选。

    接着，通过调用Clustering模块中的Cluster函数进行聚类分析。

    再次，通过调用Regression模块中的QuantitativeAnalysis函数进行定量
    分析，通过调用Classification模块中的QualitativeAnalysis函数进行
    定性分析。

    最后，程序还提供了两个函数 SpectralClusterAnalysis 和
    SpectralQuantitativeAnalysis, 分别对光谱聚类分析和光谱定量分析
    进行了封装。
"""
import torch
import torch.cuda
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import random_split
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler

from DataLoad.DataLoad import SetSplit, LoadNirtest
from Preprocessing.Preprocessing import Preprocessing
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet
from Regression.Rgs import QuantitativeAnalysis
import multiprocessing
import programParameter.stringConfig as STRING_CONFIG
import programParameter.threadConfig as THREAD_CONFIG
from Regression.CNN import ZspPocessnew
from Evaluate.RgsEvaluate import ModelRgsevaluatePro

from functools import partial
import numpy as np
import os

from AutoTune.AutoTuneConfig import autoTuneMain


# 光谱定量分析
def SpectralQuantitativeAnalysis(data, label, ProcessMethods,
                                 FslecetedMethods, SetSplitMethods,
                                 model, EPOCH, acti, cnn_depth, loss, optim, is_autoTune, autoHyperConfig):
    """

    :param is_autoTune:
    :param autoHyperConfig:
    :param data: shape (n_samples, n_features), 光谱数据
    :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
    :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
    :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
    :param SetSplitMethods: string, 划分数据集的方法, 提供随机划分、KS划分、SPXY划分
    :param model: string, 定量分析模型, 包括ANN、Pls、SVR、ELM、CNN_vgg, CNN_inception,
                        CNN_Resnet, CNN_DenseNet等，后续会不断补充完整
    :param EPOCH: int, 数据集训练次数
    :param acti: string, 激活函数，提供 relu、lrelu
    :param cnn_depth:
    :param loss: string, loss 函数，提供 MSE、L1、CrossEntropy、Poisson、KLDiv (not supported)
    :param optim: string, 优化器，提供 Adam、SGD、Adagrad、Adadelta、RMSprop、Adamax、LBFGS
    :return: Rmse: float, Rmse回归误差评估指标
             R2: float, 回归拟合,
             Mae: float, Mae回归误差评估指标
    """
    # nirplot_assign(data,600,1898,2)
    processedData = Preprocessing(ProcessMethods, data)
    featureData, labels = SpctrumFeatureSelcet(FslecetedMethods, processedData, label)
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, featureData, labels, test_size=0.2, randomseed=123)
    QuantitativeAnalysis(model, X_train, X_test, y_train, y_test, EPOCH,
                         acti, cnn_depth, loss, optim, is_autoTune=is_autoTune,
                         autoHyperConfig=autoHyperConfig)
    # Rmse, R2, Mae = QuantitativeAnalysis(model, X_train, X_test, y_train, y_test, EPOCH,
    #                                      acti, cnn_depth, loss, optim, is_autoTune=is_autoTune,
    #                                      autoHyperConfig=autoHyperConfig)
    #
    # return Rmse, R2, Mae


# todo: moved
# def load_data(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods):
#     """
#     这个函数主要服务于Ray Tune流水线
#     :param data:
#     :param label:
#     :param ProcessMethods:
#     :param FslecetedMethods:
#     :param SetSplitMethods:
#     :return:
#     """
#     processedData = Preprocessing(ProcessMethods, data)
#     featureData, labels = SpctrumFeatureSelcet(FslecetedMethods, processedData, label)
#     X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, featureData, labels, test_size=0.2, randomseed=123)
#     return X_train, X_test, y_train, y_test


# todo: moved
# def autoTuneMain(num_samples=10, max_num_epochs=10, gpus_per_trial=1, data=None, label=None,
#                  ProcessMethods=None, FslecetedMethods=None, SetSplitMethods=None, model=None,
#                  cnn_depth=None, loss=None, config=None, max_cpu_cores=1):
#     # load data
#     data_wrapUp = load_data(data=data, label=label,
#                             ProcessMethods=ProcessMethods,
#                             FslecetedMethods=FslecetedMethods,
#                             SetSplitMethods=SetSplitMethods)
#
#     scheduler = ASHAScheduler(
#         metric="loss",
#         mode="min",
#         max_t=max_num_epochs,
#         grace_period=1,
#         reduction_factor=2)
#
#     reporter = CLIReporter(
#         parameter_columns=[STRING_CONFIG.cnn_depth, STRING_CONFIG.epochs, STRING_CONFIG.lr, STRING_CONFIG.momentum,
#                            STRING_CONFIG.optimizer, STRING_CONFIG.acti_func],
#         metric_columns=["loss", "accuracy", "training_iteration"])
#
#     # 使用字典映射调用 loss函数
#     LOSS_DICT = {
#         STRING_CONFIG.MSE: nn.MSELoss(),
#         STRING_CONFIG.L1: nn.L1Loss(),
#         STRING_CONFIG.CrossEntropy: nn.CrossEntropyLoss(ignore_index=-100),
#         STRING_CONFIG.Poisson: nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-08),
#         # 'KLDiv': nn.KLDivLoss(reduction='batchmean'),
#     }
#
#     result = tune.run(
#         partial(train, data_wrapUp=data_wrapUp),
#         resources_per_trial={"cpu": max_cpu_cores, "gpu": gpus_per_trial},
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter)
#
#     best_trial = result.get_best_trial("loss", "min", "last")
#     print("Best trial config: {}".format(best_trial.config))
#     print("Best trial final validation loss: {}".format(
#         best_trial.last_result["loss"]))
#     print("Best trial final validation accuracy: {}".format(
#         best_trial.last_result["accuracy"]))
#
#     # ============================= 自动超参完成后 ====================================
#     best_trained_model = Net(best_trial.config[STRING_CONFIG.acti_func], best_trial.config[STRING_CONFIG.cnn_depth])
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if gpus_per_trial > 1:
#             best_trained_model = nn.DataParallel(best_trained_model)
#     best_trained_model.to(device)
#
#     best_checkpoint_dir = best_trial.checkpoint.value
#     model_state, optimizer_state = torch.load(os.path.join(
#         best_checkpoint_dir, "checkpoint"))
#     best_trained_model.load_state_dict(model_state)
#
#     test_acc = test_accuracy(best_trained_model, device, data_wrapUp=data_wrapUp)
#     print("Best trial test set accuracy: {}".format(test_acc))
#     # ============================= 自动超参完成后 ====================================


# class Net(nn.Module):
#     def __init__(self, acti_func, cnn_depth):
#         super(Net, self).__init__()
#         self.layers = nn.ModuleList([])
#         input_channel = 1
#         output_channel = 16
#         for i in range(1, cnn_depth):
#             self.layers.append(nn.Conv1d(input_channel, output_channel, 3, padding=1))
#             self.layers.append(nn.BatchNorm1d(num_features=output_channel))
#             self.layers.append(ac_dict[acti_func](inplace=True))
#             self.layers.append(nn.MaxPool1d(2, 2))
#             input_channel = output_channel
#             output_channel = output_channel * 2
#         # linear[c_num-1]
#         self.reg = nn.Sequential(
#             nn.Linear(649, 1000),  # 根据自己数据集修改
#             nn.ReLU(inplace=True),
#             nn.Linear(1000, 500),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(500, 1), )
#
#     def forward(self, x):
#         out = x
#         for layer in self.layers:
#             out = layer(out)
#         out = out.flatten(start_dim=1)
#         # out = out.view(-1,self.output_channel)
#         out = self.reg(out)
#         return out

# todo: moved
# def train(config, checkpoint_dir=None, data_dir=None, loss_func=None, data_wrapUp=None, Net=None):
#     BATCH_SIZE = 16
#     TBATCH_SIZE = 240
#
#     # 实例化模型 （自动超参：激活函数、网络深度）
#     network = Net(config[STRING_CONFIG.acti_func], config[STRING_CONFIG.cnn_depth])
#
#     # 设置运行设备
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             network = nn.DataParallel(network)
#     network.to(device)
#
#     # 设置 损失函数
#     criterion = loss_func
#
#     # 使用字典映射调用优化器
#     OPTIM_DICT = {
#         STRING_CONFIG.Adam: torch.optim.Adam(network.parameters(), lr=config[STRING_CONFIG.lr], weight_decay=0.001),
#         STRING_CONFIG.SGD: torch.optim.SGD(network.parameters(), lr=config[STRING_CONFIG.lr],
#                                            momentum=config[STRING_CONFIG.momentum]),
#         # 接下来的激活函数 torch并不支持。
#         # 'Adagrad': torch.optim.Adagrad(network.parameters(), lr=0.01),
#         STRING_CONFIG.Adadelta: torch.optim.Adadelta(network.parameters()),
#         STRING_CONFIG.RMSprop: torch.optim.RMSprop(network.parameters(), lr=config[STRING_CONFIG.lr], alpha=0.99),
#         STRING_CONFIG.Adamax: torch.optim.Adamax(network.parameters(), lr=config[STRING_CONFIG.lr], betas=(0.9, 0.999)),
#         # 'LBFGS': torch.optim.LBFGS(network.parameters(), lr=0.01)
#     }
#     # 设置 模型优化器 (自动超参：模型优化器)
#     optimizer = OPTIM_DICT[config[STRING_CONFIG.optimizer]]
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
#                                                            mode="min",
#                                                            factor=0.5, verbose=1,
#                                                            eps=1e-06,
#                                                            patience=20)
#
#     # 多次训练之间共享数据
#     if checkpoint_dir:
#         model_state, optimizer_state = torch.load(
#             os.path.join(checkpoint_dir, "checkpoint"))
#         net.load_state_dict(model_state)
#         optimizer.load_state_dict(optimizer_state)
#
#     # 获得（划分处理完成后的）训练和测试数据集合
#     X_train, X_test, y_train, y_test = data_wrapUp
#
#     # 数据集预处理以及数据集载入
#     trainSet, testSet, yscaler = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)
#
#     test_abs = int(len(trainSet) * 0.8)
#     # 数据集划分
#     train_subset, val_subset = random_split(trainSet, [test_abs, len(trainSet) - test_abs])
#     train_loader = torch.utils.data.DataLoader(dataset=train_subset,
#                                                batch_size=BATCH_SIZE,
#                                                shuffle=True,
#                                                num_workers=THREAD_CONFIG.num_workers)
#     val_loader = torch.utils.data.DataLoader(dataset=val_subset,
#                                              batch_size=TBATCH_SIZE,
#                                              shuffle=True,
#                                              num_workers=THREAD_CONFIG.num_workers)
#
#     print("Start Training!\n")
#     # 以 per epoch 为单位记录模型训练损失 & 以 per iteration 为单位记录模型训练损失
#     trainIter_losses = []
#     epoch_loss = []
#     for epoch in range(config[STRING_CONFIG.epochs]):
#         train_rmse = []
#         train_r2 = []
#         train_mae = []
#         avg_loss = []
#
#         ################### 记录以epoch来记录 loss ###################
#         temp_trainLosses = 0
#         ################### 记录以epoch来记录 loss ###################
#
#         running_loss = 0.0
#         epoch_steps = 0
#         for i, data in enumerate(train_loader, 0):
#             # 获得输入，data is a list of [inputs, labels]
#             inputs, labels = data  # 输入和标签都等于data
#             inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
#             labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#             outputs = network(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             pred = outputs.detach().cpu().numpy()
#             y_true = labels.detach().cpu().numpy()
#             trainIter_losses.append(loss.item())
#             temp_trainLosses = loss.item()
#             rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
#             avg_train_loss = np.mean(trainIter_losses)
#             train_rmse.append(rmse)
#             train_r2.append(R2)
#             train_mae.append(mae)
#
#             # print statistics
#             running_loss += loss.item()
#             epoch_steps += 1
#             if i % 2000 == 1999:  # print every 2000 mini-batches
#                 print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
#                                                 running_loss / epoch_steps))
#                 running_loss = 0.0
#
#         epoch_loss.append(temp_trainLosses)
#         avgrmse = np.mean(train_rmse)
#         avgr2 = np.mean(train_r2)
#         avgmae = np.mean(train_mae)
#         # print('Epoch:{}, TRAIN:rmse:{}, R2:{}, mae:{}'.format((epoch + 1), (avgrmse), (avgr2), (avgmae)))
#         # print('lr:{}, avg_train_loss:{}'.format((optimizer.param_groups[0]['lr']), avg_train_loss))
#
#         avg_loss.append(np.array(avg_train_loss))
#
#         # Validation loss
#         val_loss = 0.0
#         val_steps = 0
#         total = 0
#         correct = 0
#
#         test_rmse = []
#         test_r2 = []
#         test_mae = []
#         for i, data in enumerate(val_loader, 0):
#             with torch.no_grad():
#                 inputs, labels = data  # 输入和标签都等于data
#                 inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
#                 labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
#
#                 outputs = network(inputs)
#                 pred = outputs.detach().cpu().numpy()
#                 # y_pred.append(pred.astype(int))
#                 y_true = labels.detach().cpu().numpy()
#                 # y.append(y_true.astype(int))
#                 rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
#                 test_rmse.append(rmse)
#                 test_r2.append(R2)
#                 test_mae.append(mae)
#
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.cpu().numpy()
#                 val_steps += 1
#
#             avgrmse = np.mean(test_rmse)
#             avgr2 = np.mean(test_r2)
#             avgmae = np.mean(test_mae)
#             # print('EPOCH：{}, TEST: rmse:{}, R2:{}, mae:{}\n'.format((epoch + 1), (avgrmse), (avgr2), (avgmae)))
#             # 将每次测试结果实时写入acc.txt文件中
#             scheduler.step(rmse)
#
#         with tune.checkpoint_dir(epoch) as checkpoint_dir:
#             path = os.path.join(checkpoint_dir, "checkpoint")
#             torch.save((network.state_dict(), optimizer.state_dict()), path)
#
#         tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
#     print("Finished Training")


# todo: moved
# def test_accuracy(net, device="cpu", data_wrapUp=None):
#     X_train, X_test, y_train, y_test = data_wrapUp
#
#     # 数据集预处理以及数据集载入
#     trainSet, testSet, yscaler = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)
#
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=4, shuffle=False, num_workers=2)
#
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     return correct / total


if __name__ == '__main__':
    # 载入原始数据并可视化
    data2, label2 = LoadNirtest('Rgs')

    # 是否使用自动超参功能
    is_autoTune = True
    TRAINING_BATCH_SIZE = 16
    TESTING_BATCH_SIZE = 240

    if is_autoTune:
        # 获得本设备CPU最大核心数量。
        #   注意：如果出现可用内存空间不足，请自行调整`max_cpu_cores`以减少CPU核心数的使用！
        max_cpu_cores = multiprocessing.cpu_count()
        print("CPU has " + str(max_cpu_cores) + " cores on this device.")

        # 设置环境变量，解决Ray Tune因为CPU检测不准确而给出的warning。
        os.environ['RAY_USE_MULTIPROCESSING_CPU_COUNT'] = '1'
        # 设置超参数搜索空间
        search_space = {
            STRING_CONFIG.cnn_depth: tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8]),
            STRING_CONFIG.epochs: tune.grid_search([10, 20, 30]),  # 具体EPOCHS搜参空间请根据模型的收敛情况判断而定
            STRING_CONFIG.lr: tune.grid_search([.001, .01, .1]),
            STRING_CONFIG.momentum: tune.grid_search([.5, .9, .99]),
            STRING_CONFIG.optimizer: tune.grid_search(["Adam", "SGD", "Adadelta", "RMSprop", "Adamax"]),
            STRING_CONFIG.acti_func: tune.choice(["relu", "lrelu"])
        }
        autoTuneMain(num_samples=10, max_num_epochs=10, gpus_per_trial=1, network="CNN_vgg",
                     data=data2, label=label2, ProcessMethods="MMS", FslecetedMethods="None",
                     SetSplitMethods="random", model="CNN_vgg", cnn_depth=5,
                     loss=STRING_CONFIG.MSE, config=search_space, max_cpu_cores=max_cpu_cores,
                     trainingBatchSize=TRAINING_BATCH_SIZE, testingBatchSize=TESTING_BATCH_SIZE)
    else:
        # 光谱定量分析演示
        # 示意1: 预处理算法:MSC , 波长筛选算法: Uve, 数据集划分:KS, 定性分量模型: SVR
        # 这里我改了参数，10,'relu',5,'MSE','Adam'，10 是epoch，relu是激活函数，可以选。5是CNN有5层，MSE是损失函数，Adam是优化，这些可以到CNN.p文件看一下有什么选择。
        SpectralQuantitativeAnalysis(data=data2, label=label2,
                                     ProcessMethods="MMS", FslecetedMethods="None",
                                     SetSplitMethods="random",
                                     model=CNN_vgg, EPOCH=3, acti='relu',
                                     cnn_depth=5, loss='MSE', optim='SGD',
                                     is_autoTune=False,
                                     autoHyperConfig=search_space)
