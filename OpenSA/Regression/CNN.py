"""
    Create on 2021-1-21
    Author：Pengyou Fu
    Describe：this for train NIRS with use 1-D Resnet model to transfer
"""

"""
    这段代码主要实现了利用1-D Resnet模型进行NIRS预测的训练过程。
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
from Regression.CnnModel import ConvNet, DeepSpectra, AlexNet, SpectraCNN
import os
from datetime import datetime
from Evaluate.RgsEvaluate import ModelRgsevaluate, ModelRgsevaluatePro
import matplotlib.pyplot as plt
from tqdm import tqdm
from Regression.CnnModel import ConvNet, AlexNet, DeepSpectra, SpectraCNN

LR = 0.001
BATCH_SIZE = 16
TBATCH_SIZE = 240

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义加载数据集
class MyDataset(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec, target = self.specs[index], self.labels[index]
        return spec, target

    def __len__(self):
        return len(self.specs)


###定义是否需要标准化
def ZspPocessnew(X_train, X_test, y_train, y_test, need=True):  # True:需要标准化，Flase：不需要标准化

    global standscale
    global yscaler

    if (need == True):
        standscale = StandardScaler()
        X_train_Nom = standscale.fit_transform(X_train)
        X_test_Nom = standscale.transform(X_test)

        # yscaler = StandardScaler()
        yscaler = MinMaxScaler()
        y_train = yscaler.fit_transform(y_train.reshape(-1, 1))
        y_test = yscaler.transform(y_test.reshape(-1, 1))

        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]

        ##使用loader加载测试数据
        data_train = MyDataset(X_train_Nom, y_train)
        data_test = MyDataset(X_test_Nom, y_test)
        return data_train, data_test
    elif ((need == False)):
        yscaler = StandardScaler()
        # yscaler = MinMaxScaler()

        X_train_new = X_train[:, np.newaxis, :]  #
        X_test_new = X_test[:, np.newaxis, :]

        y_train = yscaler.fit_transform(y_train)
        y_test = yscaler.transform(y_test)

        data_train = MyDataset(X_train_new, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test_new, y_test)

        return data_train, data_test


# 使用字典映射调用函数
net_dict = {
    'ConvNet': ConvNet,
    'AlexNet': AlexNet,
    'DeepSpectra': DeepSpectra,
    'SpectraCNN': SpectraCNN
}


def CNNTrain(NetType, X_train, X_test, y_train, y_test, EPOCH):
    """
    CNN模型训练函数
    :param NetType: 模型类型，包括'ConvNet'，'AlexNet'，'DeepSpectra', 'SpectraCNN'
    :param X_train: 训练数据
    :param X_test: 测试数据
    :param y_train: 训练标签
    :param y_test: 测试标签
    :param EPOCH: 迭代次数
    :return: None
    """

    data_train, data_test = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)
    # data_train, data_test = ZPocess(X_train, X_test, y_train, y_test)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=True)

    # 通过使用字典映射的方法使得函数调用避免使用过多地if-else判断
    model = net_dict[NetType]().to(device)

    """
    以下是可供选择的loss function，用于调参。
    这里推荐使用nn.MSELoss或者nn.L1Loss，因为在一维光谱问题中，
    我们需要预测出光谱的强度值，这是一个连续值，用MSE或者L1 loss是最常见的做法。
    
    - nn.MSELoss: 均方误差损失函数，用于回归问题
    - nn.L1Loss: 平均绝对误差损失函数，常用于回归问题
    - nn.CrossEntropyLoss: 交叉熵损失函数，常用于分类问题
    - nn.NLLLoss: 对数似然损失函数，常用于自然语言处理问题
    - nn.PoissonNLLLoss: Poisson对数似然损失函数，常用于回归问题
    - nn.KLDivLoss: Kullback-Leibler散度损失函数，常用于自然语言处理问题
    
        使用 CrossEntropyLoss 注意点：
            在使用nn.PoissonNLLLoss的时候，log_input=True表示输入
            的是对数值，full=False表示只计算真实值不为0的部分，eps=1e-08是为了防
            止除0错误。
        
        使用 CrossEntropyLoss, NULLLoss, PoissonNULLoss 注意点：
            CrossEntropyLoss, NLLLoss, PoissonNLLLoss都是二分类问题的
            Loss，如果你的问题是多分类问题，可以使用
            nn.CrossEntropyLoss(ignore_index=-100)来忽略某一个类别。
            
        使用 KLDivLoss：
            当reduction='batchmean'时，在一个batch中，会计算所有样本的KL divergence然后求平均。
            
            这样做的目的是为了减少每个样本对最终loss的贡献，使得loss更加稳定。当
            然如果你不需要这样做，可以将reduction设置为None。
        
    """
    ################################# Loss Functions ###################################
    criterion = nn.MSELoss().to(device)  # 损失函数为焦损函数，多用于类别不平衡的多分类问题
    # criterion = nn.L1Loss.to(device)
    # criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)
    # criterion = nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-08).to(device)
    # criterion = nn.KLDivLoss(reduction='batchmean').to(device)
    ################################# Loss Functions ###################################


    """
    以下提供了一些可替换的优化器，用于调参。
    
    需要注意的是每个优化器都有自己的默认参数值，在实际使用中可能需要根据
    实际情况进行微调。
    """
    ################################### Optimizers #####################################
    # optimizer = optim.Adam(model.parameters(), lr=LR)#,  weight_decay=0.001)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adadelta(model.parameters())
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
    # optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999))
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)
    ################################### Optimizers #####################################

    # # initialize the early_stopping object
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           factor=0.5,
                                                           verbose=1,
                                                           eps=1e-06,
                                                           patience=20)

    print("Start Training!\n")  # 定义遍历数据集的次数
    # to track the training loss as the model trains

    train_losses = []
    for epoch in range(EPOCH):
        model.train()  # 将模型置于训练状态，Pytorch会在向前传播时自动计算梯度。
        train_rmse = []
        train_r2 = []
        train_mae = []
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
            rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
            # plotpred(pred, y_true, yscaler))
            train_rmse.append(rmse)
            train_r2.append(R2)
            train_mae.append(mae)
        avg_train_loss = np.mean(train_losses)
        avgrmse = np.mean(train_rmse)
        avgr2 = np.mean(train_r2)
        avgmae = np.mean(train_mae)
        print('Epoch:{}, TRAIN:rmse:{}, R2:{}, mae:{}'.format((epoch + 1), (avgrmse), (avgr2), (avgmae)))
        print('lr:{}, avg_train_loss:{}'.format((optimizer.param_groups[0]['lr']), avg_train_loss))

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
                y_true = labels.detach().cpu().numpy()
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

    ##################### 将 训练时的loss 打印出来 #######################
    print("\n\nThe loss data of %d iterations has been recorded." % (np.array(train_losses).shape[0]))
    plt.rcParams['agg.path.chunksize'] = 100000  # 设置 matplotlib 画出来曲线的平滑度
    plt.plot(train_losses)
    plt.xlabel("Iterations")
    plt.ylabel("Training loss")
    plt.title("CNN Training Loss")
    plt.savefig("cnn_training_loss.png", dpi=300)  # matplotlib 将画出来的图片保存在本地，并且清晰度未300dpi
    plt.show()
    ############################################################

    return avgrmse, avgr2, avgmae
