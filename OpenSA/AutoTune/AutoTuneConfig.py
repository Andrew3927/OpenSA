from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from Preprocessing.Preprocessing import Preprocessing
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet
from DataLoad.DataLoad import SetSplit

from Regression.CnnModel import *

import sys

from AutoTune.Training import train, test_accuracy

import programParameter.stringConfig as STRING_CONFIG

from functools import partial

def load_data(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods):
    """
    这个函数主要服务于Ray Tune流水线
    :param data:
    :param label:
    :param ProcessMethods:
    :param FslecetedMethods:
    :param SetSplitMethods:
    :return:
    """
    processedData = Preprocessing(ProcessMethods, data)
    featureData, labels = SpctrumFeatureSelcet(FslecetedMethods, processedData, label)
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, featureData, labels, test_size=0.2, randomseed=123)
    return X_train, X_test, y_train, y_test


NET_DICT = {
    'vgg': AlexNet,
    'inception': DeepSpectra,
    'Resnet': Resnet,
    'DenseNet': DenseNet,
}


def autoTuneMain(num_samples=10, max_num_epochs=10, gpus_per_trial=1, data=None, label=None,
                 ProcessMethods=None, FslecetedMethods=None, SetSplitMethods=None, model=None,
                 cnn_depth=None, loss=None, config=None, max_cpu_cores=1, network=None,
                 trainingBatchSize=16, testingBatchSize=240):
    print("String with Auto Tune mode")

    if (network[0:3] != "CNN"):
        print("\033[1;31;40m" + "`is_autoTune` is True, using auto tune mode for training.\n"
                                "But " + network + " might be a traditional network, and only CNN type models are supported.\n"
                                               "Please reset `Net` in example.py with one of the models given following:\n"
                                               "CNN_vgg, CNN_inception, CNN_Resnet, CNN_DenseNet" + "\033[0m")
        sys.exit()

    if (network[4:] not in NET_DICT):
        print("model=" + "\033[1;31;40m" + model + "\033[0m" + " hasn't been implemented in this project yet.")
        sys.exit()

    Net = NET_DICT[network[4:]]

    print("\033[32m" + "Using " + network + " for training ..." + "\033[0m")

    # load data
    data_wrapUp = load_data(data=data, label=label,
                            ProcessMethods=ProcessMethods,
                            FslecetedMethods=FslecetedMethods,
                            SetSplitMethods=SetSplitMethods)

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=[STRING_CONFIG.cnn_depth, STRING_CONFIG.epochs, STRING_CONFIG.lr, STRING_CONFIG.momentum,
                           STRING_CONFIG.optimizer, STRING_CONFIG.acti_func],
        metric_columns=["loss", "accuracy", "training_iteration"])

    # 使用字典映射调用 loss函数
    LOSS_DICT = {
        STRING_CONFIG.MSE: nn.MSELoss(),
        STRING_CONFIG.L1: nn.L1Loss(),
        STRING_CONFIG.CrossEntropy: nn.CrossEntropyLoss(ignore_index=-100),
        STRING_CONFIG.Poisson: nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-08),
        # 'KLDiv': nn.KLDivLoss(reduction='batchmean'),
    }

    # Ray Tune 开始训练模型
    result = tune.run(
        partial(train, data_wrapUp=data_wrapUp, loss_func=nn.MSELoss(), Net=Net,
                TRAINING_BATCH_SIZE=trainingBatchSize, TESTING_BATCH_SIZE=testingBatchSize),
        resources_per_trial={"cpu": max_cpu_cores, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # ============================= 自动超参完成后 ====================================
    # todo: Net需要移植
    best_trained_model = network(best_trial.config[STRING_CONFIG.acti_func], best_trial.config[STRING_CONFIG.cnn_depth])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device, data_wrapUp=data_wrapUp)
    print("Best trial test set accuracy: {}".format(test_acc))
    # ============================= 自动超参完成后 ====================================
