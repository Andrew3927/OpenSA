import torch
import torch.cuda
import torch.nn as nn
from torch.autograd import Variable

from Regression.CNN import ZspPocessnew

from torch.utils.data import random_split

import programParameter.threadConfig as THREAD_CONFIG

from Evaluate.RgsEvaluate import ModelRgsevaluatePro

import programParameter.stringConfig as STRING_CONFIG

import numpy as np

from ray import tune

import os



def train(config, checkpoint_dir=None, data_dir=None, loss_func=None, data_wrapUp=None, Net=None,
          TRAINING_BATCH_SIZE=16, TESTING_BATCH_SIZE=240):

    # 实例化模型 （自动超参：激活函数、网络深度）
    network = Net(config[STRING_CONFIG.acti_func], config[STRING_CONFIG.cnn_depth])

    # 设置运行设备
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            network = nn.DataParallel(network)
    network.to(device)

    # 设置 损失函数
    criterion = loss_func

    # 使用字典映射调用优化器
    OPTIM_DICT = {
        STRING_CONFIG.Adam: torch.optim.Adam(network.parameters(), lr=config[STRING_CONFIG.lr], weight_decay=0.001),
        STRING_CONFIG.SGD: torch.optim.SGD(network.parameters(), lr=config[STRING_CONFIG.lr],
                                           momentum=config[STRING_CONFIG.momentum]),
        # 接下来的激活函数 torch并不支持。
        # 'Adagrad': torch.optim.Adagrad(network.parameters(), lr=0.01),
        STRING_CONFIG.Adadelta: torch.optim.Adadelta(network.parameters()),
        STRING_CONFIG.RMSprop: torch.optim.RMSprop(network.parameters(), lr=config[STRING_CONFIG.lr], alpha=0.99),
        STRING_CONFIG.Adamax: torch.optim.Adamax(network.parameters(), lr=config[STRING_CONFIG.lr], betas=(0.9, 0.999)),
        # 'LBFGS': torch.optim.LBFGS(network.parameters(), lr=0.01)
    }

    # 设置 模型优化器 (自动超参：模型优化器)
    optimizer = OPTIM_DICT[config[STRING_CONFIG.optimizer]]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode="min",
                                                           factor=0.5, verbose=1,
                                                           eps=1e-06,
                                                           patience=20)

    # 多次训练之间共享数据
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        Net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # 获得（划分处理完成后的）训练和测试数据集合
    X_train, X_test, y_train, y_test = data_wrapUp

    # 数据集预处理以及数据集载入
    trainSet, testSet, yscaler = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)

    test_abs = int(len(trainSet) * 0.8)
    # 数据集划分
    train_subset, val_subset = random_split(trainSet, [test_abs, len(trainSet) - test_abs])
    train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                               batch_size=TRAINING_BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=THREAD_CONFIG.num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                             batch_size=TESTING_BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=THREAD_CONFIG.num_workers)

    print("Start Training!\n")
    # 以 per epoch 为单位记录模型训练损失 & 以 per iteration 为单位记录模型训练损失
    trainIter_losses = []
    epoch_loss = []
    for epoch in range(config[STRING_CONFIG.epochs]):
        train_rmse = []
        train_r2 = []
        train_mae = []
        avg_loss = []

        ################### 记录以epoch来记录 loss ###################
        temp_trainLosses = 0
        ################### 记录以epoch来记录 loss ###################

        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            # 获得输入，data is a list of [inputs, labels]
            inputs, labels = data  # 输入和标签都等于data
            inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
            labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = outputs.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            trainIter_losses.append(loss.item())
            temp_trainLosses = loss.item()
            rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
            avg_train_loss = np.mean(trainIter_losses)
            train_rmse.append(rmse)
            train_r2.append(R2)
            train_mae.append(mae)

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        epoch_loss.append(temp_trainLosses)
        avgrmse = np.mean(train_rmse)
        avgr2 = np.mean(train_r2)
        avgmae = np.mean(train_mae)
        # print('Epoch:{}, TRAIN:rmse:{}, R2:{}, mae:{}'.format((epoch + 1), (avgrmse), (avgr2), (avgmae)))
        # print('lr:{}, avg_train_loss:{}'.format((optimizer.param_groups[0]['lr']), avg_train_loss))

        avg_loss.append(np.array(avg_train_loss))

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0

        test_rmse = []
        test_r2 = []
        test_mae = []
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data  # 输入和标签都等于data
                inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y

                outputs = network(inputs)
                pred = outputs.detach().cpu().numpy()
                # y_pred.append(pred.astype(int))
                y_true = labels.detach().cpu().numpy()
                # y.append(y_true.astype(int))
                rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
                test_rmse.append(rmse)
                test_r2.append(R2)
                test_mae.append(mae)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

            avgrmse = np.mean(test_rmse)
            avgr2 = np.mean(test_r2)
            avgmae = np.mean(test_mae)
            # print('EPOCH：{}, TEST: rmse:{}, R2:{}, mae:{}\n'.format((epoch + 1), (avgrmse), (avgr2), (avgmae)))
            # 将每次测试结果实时写入acc.txt文件中
            scheduler.step(rmse)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((network.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")

def test_accuracy(net, device="cpu", data_wrapUp=None):
    X_train, X_test, y_train, y_test = data_wrapUp

    # 数据集预处理以及数据集载入
    trainSet, testSet, yscaler = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

