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
def ZspPocessnew(X_train, X_test, y_train, y_test, need=True):  # True:需要标准化，False：不需要标准化

    global standscale
    global yscaler

    if need:
        standscale = StandardScaler()
        X_train_Nom = standscale.fit_transform(X_train)
        X_test_Nom = standscale.transform(X_test)

        # yscaler = StandardScaler()
        yscaler = MinMaxScaler()
        y_train = yscaler.fit_transform(y_train.reshape(-1, 1))
        y_test = yscaler.transform(y_test.reshape(-1, 1))

        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]

        # 使用loader加载测试数据
        data_train = MyDataset(X_train_Nom, y_train)
        data_test = MyDataset(X_test_Nom, y_test)
        return data_train, data_test
    elif not need:
        yscaler = StandardScaler()
        # yscaler = MinMaxScaler()

        X_train_new = X_train[:, np.newaxis, :]  #
        X_test_new = X_test[:, np.newaxis, :]

        y_train = yscaler.fit_transform(y_train)
        y_test = yscaler.transform(y_test)

        data_train = MyDataset(X_train_new, y_train)
        # 使用loader加载测试数据
        data_test = MyDataset(X_test_new, y_test)

        return data_train, data_test


# 使用字典映射调用函数
net_dict = {
    'vgg': AlexNet,
    'inception': DeepSpectra,
    'Resnet': Resnet,
    'DenseNet': DenseNet,
}

# 使用字典映射调用损失函数
loss_dict = {
    'MSE': nn.MSELoss(),
    'L1': nn.L1Loss(),
    'CrossEntropy': nn.CrossEntropyLoss(ignore_index=-100),
    'Poisson': nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-08),
    'KLDiv': nn.KLDivLoss(reduction='batchmean'),

}


def CNNTrain(NetType, X_train, X_test, y_train, y_test, EPOCH, acti, c_num, loss, optim):
    data_train, data_test = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)
    # data_train, data_test = ZPocess(X_train, X_test, y_train, y_test)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=True)
    # model = net_dict[NetType](acti,c_num).to(device)
    # 这里我还没有完全优化好，只有纵向结构可以传数，其他的需要把acti,c_num删掉
    model = net_dict[NetType](acti, c_num)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = loss_dict[loss].to(device)
    # 使用字典映射调用优化器
    optim_dict = {
        'Adam': torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001),
        'SGD': torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        'Adagrad': torch.optim.Adagrad(model.parameters(), lr=0.01),
        'Adadelta': torch.optim.Adadelta(model.parameters()),
        'RMSprop': torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99),
        'Adamax': torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999)),
        'LBFGS': torch.optim.LBFGS(model.parameters(), lr=0.01),
    }
    # # initialize the early_stopping object
    optimizer = optim_dict[optim]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=1, eps=1e-06,
                                                           patience=20)
    print("Start Training!")  # 定义遍历数据集的次数
    # to track the training loss as the model trains
    train_losses = []
    epoch_loss = []
    for epoch in range(EPOCH):
        # train_losses = []
        model.train()  # 不训练
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
            output = model(inputs)  # cnn output
            loss = criterion(output, labels)  # MSE
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            pred = output.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            train_losses.append(loss.item())
            temp_trainLosses = loss.item()
            rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
            avg_train_loss = np.mean(train_losses)
            train_rmse.append(rmse)
            train_r2.append(R2)
            train_mae.append(mae)

        epoch_loss.append(temp_trainLosses)
        avgrmse = np.mean(train_rmse)
        avgr2 = np.mean(train_r2)
        avgmae = np.mean(train_mae)
        print('Epoch:{}, TRAIN:rmse:{}, R2:{}, mae:{}'.format((epoch + 1), (avgrmse), (avgr2), (avgmae)))
        print('lr:{}, avg_train_loss:{}'.format((optimizer.param_groups[0]['lr']), avg_train_loss))

        avg_loss.append(np.array(avg_train_loss))

        with torch.no_grad():  # 无梯度
            model.eval()  # 不训练
            test_rmse = []
            test_r2 = []
            test_mae = []
            for i, data in enumerate(test_loader):
                inputs, labels = data  # 输入和标签都等于data
                inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
                outputs = model(inputs)  # 输出等于进入网络后的输入
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

    return avgrmse, avgr2, avgmae
