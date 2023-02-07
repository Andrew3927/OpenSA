"""
模块级别定义变量来设置全局静态常量。变量在模块级别是全局变量，可以在任何地方使用。
"""
# 自动超参键名字典
cnn_depth = "cnn_depth"
epochs = "EPOCHS"
lr = "lr"
momentum = "momentum"
optimizer = "optimizer"
acti_func = "activation_function"

# 损失函数键名字典
MSE = "MSE"
L1 = "L1"
CrossEntropy = "CrossEntropy"
Poisson = "Poisson"

# 模型优化器键名字典
Adam = "Adam"
SGD = "SGD"
Adadelta = "Adadelta"
RMSprop = "RMSprop"
Adamax = "Adamax"