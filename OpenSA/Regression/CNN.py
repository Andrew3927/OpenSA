"""
    这段代码主要实现了神经网络处理模型进行NIRS预测的训练过程。
    代码包括了自定义数据加载，标准化处理，模型训练，训练结果评估等过程。
    其中定义了一个函数CNNTrain，该函数通过输入模型类型，训练数据，测试数
    据，训练标签，测试标签和训练轮数来进行模型训练。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
from sklearn.preprocessing import scale, MinMaxScaler, Normalizer, StandardScaler
import torch.optim as optim
from Regression.CnnModel import DeepSpectra, AlexNet, Resnet, DenseNet
import os
from datetime import datetime
from Evaluate.RgsEvaluate import ModelRgsevaluate, ModelRgsevaluatePro
import matplotlib.pyplot as plt
from Plot.plot import nirplot_eva_epoch, nirplot_eva_iterations

LR = 0.001
BATCH_SIZE = 16
TBATCH_SIZE = 240


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec, target = self.specs[index], self.labels[index]
        return spec, target

    def __len__(self):
        return len(self.specs)


# 定义是否需要标准化
def ZspPocessnew(XTrain, XTest, yTrain, yTest, need=True):  # True:需要标准化，False：不需要标准化

    global standscale
    global yscaler

    if need:
        standscale = StandardScaler()
        X_train_Nom = standscale.fit_transform(XTrain)
        X_test_Nom = standscale.transform(XTest)

        # yscaler = StandardScaler()
        yscaler = MinMaxScaler()
        yTrain = yscaler.fit_transform(yTrain.reshape(-1, 1))
        yTest = yscaler.transform(yTest.reshape(-1, 1))

        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]

        # 使用loader加载测试数据
        data_train = MyDataset(X_train_Nom, yTrain)
        data_test = MyDataset(X_test_Nom, yTest)
        return data_train, data_test, yscaler
    elif not need:
        yscaler = StandardScaler()
        # yscaler = MinMaxScaler()

        X_train_new = XTrain[:, np.newaxis, :]  #
        X_test_new = XTest[:, np.newaxis, :]

        yTrain = yscaler.fit_transform(yTrain)
        yTest = yscaler.transform(yTest)

        data_train = MyDataset(X_train_new, yTrain)
        # 使用loader加载测试数据
        data_test = MyDataset(X_test_new, yTest)

        return data_train, data_test, yscaler


# def CNNTrain(model, X_train, X_test, y_train, y_test, EPOCH, acti, c_num, loss_function, optim_func):
class CNNTrain:

    # def __init__(self, network, X_train, X_test, y_train, y_test):
    #     self.Net = network
    #     self.X_train = X_train
    #     self.X_test = X_test
    #     self.y_train = y_train
    #     self.y_test = y_test
    #     self.device = "cpu"
    #     self.is_autoTuning = True;

    def __init__(self, network, X_train, X_test, y_train, y_test, EPOCH, loss_func, optim_func):
        self.Net = network
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.EPOCH = EPOCH
        self.loss_func = loss_func
        self.optim_func = optim_func
        self.device = "cpu"
        self.is_autoTuning = False;

    def train(self, config):

        # preparing for training data
        data_train, data_test, _ = ZspPocessnew(self.X_train, self.X_test, self.y_train, self.y_test, need=True)
        train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH_SIZE,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=TBATCH_SIZE,
                                                  shuffle=True)
        # todo: 这里我还没有完全优化好，只有纵向结构可以传数，其他的需要把acti,c_num删掉
        # 目前只支持：AlexNet

        # 实例化模型
        model = self.Net(acti_func=config(['activation_function']),  # 激活函数自动超参
                         cnn_depth=config(['cnn_depth']))  # 模型深度自动超参

        # select device for using cpu, gpu, or multi-gpus
        if self.gpuDeviceEnable():
            if self.dataParallel():
                print("\033[32m" + "DataParallel is enabled for training!" + "\033[0m")
        self.Net.to(self.device)

        # 定义损失函数
        criterion = nn.MSELoss().to(self.device)

        # 定义优化器
        if config(["optimizer"]) == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        elif config(["optimizer"]) == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
        # elif config(["optimizer"]) == "Adagrad":
        #     optimizer = optim.Adagrad(model.parameters(), lr=config["lr"])
        elif config(["optimizer"]) == "Adadelta":
            optimizer = torch.optim.Adadelta(network.parameters())
        elif config(["optimizer"]) == "RMSprop":
            optimizer = torch.optim.RMSprop(network.parameters(), lr=config["lr"], alpha=0.99)
        elif config(["optimizer"]) == "Adamax":
            optimizer = torch.optim.Adamax(network.parameters(), lr=config["lr"], betas=(0.9, 0.999))
        # elif config(["optimizer"]) == "LBFGS":

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               factor=0.5, verbose=1,
                                                               eps=1e-06,
                                                               patience=20)
        # 开始训练
        print("Start Training!\n")
        # Taking down the loss as training
        train_losses = []
        epoch_loss = []
        for epoch in range(config(["EPOCHS"])):  # "EPOCHS" 自动超参
            # train_losses = []
            self.Net.train()  # 不训练
            train_rmse = []
            train_r2 = []
            train_mae = []
            avg_loss = []
            ################### 记录以epoch来记录 loss ###################
            temp_trainLosses = 0
            ################### 记录以epoch来记录 loss ###################
            for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
                inputs, labels = data  # 输入和标签都等于data
                inputs = Variable(inputs).type(torch.FloatTensor).to(self.device)  # batch x
                labels = Variable(labels).type(torch.FloatTensor).to(self.device)  # batch y
                output = self.Net(inputs)  # cnn output
                loss_function = criterion(output, labels)  # MSE
                optimizer.zero_grad()  # clear gradients for this training step
                loss_function.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                pred = output.detach().cpu().numpy()
                y_true = labels.detach().cpu().numpy()
                train_losses.append(loss_function.item())  # 以iteration来记录 loss
                temp_trainLosses = loss_function.item()  # 以epoch来记录 loss
                rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
                avg_train_loss = np.mean(train_losses)
                train_rmse.append(rmse)
                train_r2.append(R2)
                train_mae.append(mae)

            epoch_loss.append(temp_trainLosses)
            avgrmse = np.mean(train_rmse)
            avgr2 = np.mean(train_r2)
            avgmae = np.mean(train_mae)

            avg_loss.append(np.array(avg_train_loss))

            with torch.no_grad():  # 无梯度
                self.Net.eval()  # 不训练
                test_rmse = []
                test_r2 = []
                test_mae = []
                for i, data in enumerate(test_loader):
                    inputs, labels = data  # 输入和标签都等于data
                    inputs = Variable(inputs).type(torch.FloatTensor).to(self.device)  # batch x
                    labels = Variable(labels).type(torch.FloatTensor).to(self.device)  # batch y
                    outputs = self.Net(inputs)  # 输出等于进入网络后的输入
                    pred = outputs.detach().cpu().numpy()
                    # y_pred.append(pred.astype(int))
                    y_true = labels.detach().cpu().numpy()
                    # y.append(y_true.astype(int))
                    rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
                    test_rmse.append(rmse)
                    test_r2.append(R2)
                    test_mae.append(mae)
                avgrmse = np.mean(test_rmse)
                avgr2 = np.mean(test_r2)
                avgmae = np.mean(test_mae)
                print('EPOCH：{}, TEST: rmse:{}, R2:{}, mae:{}'.format((epoch + 1), (avgrmse), (avgr2), (avgmae)))
                # 将每次测试结果实时写入acc.txt文件中
                scheduler.step(rmse)

        # 调用画图函数绘制epoch-损失函数图或者iterations-损失函数图
        # nirplot_eva_iterations(train_losses)
        # nirplot_eva_epoch(epoch_loss)

        print("The RMSE:{} R2:{}, MAE:{} of result!\n".format(avgmae, avgr2, avgmae))
        # return avgrmse, avgr2, avgmae


def gpuDeviceEnable():
    """
    设置训练使用的设备类型。
    :return:
    """
    # 判断GPU是否可用
    if torch.cuda.is_available():
        # 设置使用的GPU数量
        device = "cuda:0"
        print("\033[32m" + "Using single GPU for training!" + "\033[0m")
        return True;
    else:
        print("Use CPU")
        return False


def dataParallel():
    """
    使用此函数之前需要先实例化模型(self.Net)。
    :return:
    """
    if torch.cuda.device_count() > 1:
        self.Net = nn.DataParallel(self.Net)
        return True
    return False

def train(Net, X_train, X_test, y_train, y_test, EPOCH, loss_function, optim_func):
    # preparing for training data
    data_train, data_test, _ = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)
    train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH_SIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=TBATCH_SIZE,
                                              shuffle=True)
    # todo: 这里我还没有完全优化好，只有纵向结构可以传数，其他的需要把acti,c_num删掉

    global device
    device = "cpu"
    # select device for using cpu, gpu, or multi-gpus
    if gpuDeviceEnable():
        if dataParallel():
            print("\033[32m" + "DataParallel is enabled for training!" + "\033[0m")
    Net.to(device)

    criterion = loss_function.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_func, 'min',
                                                           factor=0.5, verbose=1,
                                                           eps=1e-06,
                                                           patience=20)
    print("Start Training!\n")  # 定义遍历数据集的次数
    # to track the training loss as the model trains
    trainIter_losses = []
    epoch_loss = []
    for epoch in range(EPOCH):
        # trainIter_losses = []
        Net.train()  # 不训练
        train_rmse = []
        train_r2 = []
        train_mae = []
        avg_loss = []
        ################### 记录以epoch来记录 loss ###################
        temp_trainLosses = 0
        ################### 记录以epoch来记录 loss ###################
        for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            inputs, labels = data  # 输入和标签都等于data
            inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
            labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y

            output = Net(inputs)  # cnn output
            loss_function = criterion(output, labels)  # MSE

            optim_func.zero_grad()  # clear gradients for this training step
            loss_function.backward()  # backpropagation, compute gradients
            optim_func.step()  # apply gradients

            pred = output.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            trainIter_losses.append(loss_function.item())
            temp_trainLosses = loss_function.item()
            rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
            avg_train_loss = np.mean(trainIter_losses)
            train_rmse.append(rmse)
            train_r2.append(R2)
            train_mae.append(mae)



        epoch_loss.append(temp_trainLosses)
        avgrmse = np.mean(train_rmse)
        avgr2 = np.mean(train_r2)
        avgmae = np.mean(train_mae)
        print('Epoch:{}, TRAIN:rmse:{}, R2:{}, mae:{}'.format((epoch + 1), (avgrmse), (avgr2), (avgmae)))
        print('lr:{}, avg_train_loss:{}'.format((optim_func.param_groups[0]['lr']), avg_train_loss))

        avg_loss.append(np.array(avg_train_loss))

        with torch.no_grad():  # 无梯度
            Net.eval()  # 不训练
            test_rmse = []
            test_r2 = []
            test_mae = []
            for i, data in enumerate(test_loader):
                inputs, labels = data  # 输入和标签都等于data
                inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y

                outputs = Net(inputs)  # 输出等于进入网络后的输入
                pred = outputs.detach().cpu().numpy()
                # y_pred.append(pred.astype(int))
                y_true = labels.detach().cpu().numpy()
                # y.append(y_true.astype(int))
                rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
                test_rmse.append(rmse)
                test_r2.append(R2)
                test_mae.append(mae)

            avgrmse = np.mean(test_rmse)
            avgr2 = np.mean(test_r2)
            avgmae = np.mean(test_mae)
            print('EPOCH：{}, TEST: rmse:{}, R2:{}, mae:{}\n'.format((epoch + 1), (avgrmse), (avgr2), (avgmae)))
            # 将每次测试结果实时写入acc.txt文件中
            scheduler.step(rmse)

    # 调用画图函数绘制epoch-损失函数图或者iterations-损失函数图
    # nirplot_eva_iterations(trainIter_losses)
    # nirplot_eva_epoch(epoch_loss)

    print("The RMSE:{} R2:{}, MAE:{} of result!\n".format(avgmae, avgr2, avgmae))
    # return avgrmse, avgr2, avgmae
