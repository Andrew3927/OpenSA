import os 
import matplotlib.pyplot as plt
import numpy as np 

#这个模块是应用在能够提供模版所示的数据集的排列的画图用的
def nirplot_default(data,path):
    l= np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    cur_path = os.path.abspath('   这里')
    print("你现在的数据绝对路径在：",cur_path)
    if data.shape[1]==l.shape[0]:
        print("波长文件和原始的数据匹配")
        a=plt.plot(l,data.T)
    else:
        print("波长文件与原始的额数据不匹配 \n请进行修改")
#这个模块可以指定光波的波长的范围和步长,不用指定波长的csv文件Initial wavelength Termination wavelength 和step
def nirplot_assign(data,iw,tw,s):
    l=np.arange(iw,tw,s)
    if data.shape[1]==l.shape[0]:
        print("波长文件和原始的数据匹配")
        a=plt.plot(l,data.T)
        plt.savefig("data_draw.jpg", dpi=300)
    else:
        print("波长文件与原始的额数据不匹配 \n请进行修改")

#这个提供了可以补充图片的横轴和纵轴和标题的功能
def nirplot_assign1(data,iw,tw,s,xlabel,ylabel,title):
    l=np.arange(iw,tw,s)
    if data.shape[1]==l.shape[0]:
        print("波长文件和原始的数据匹配")
        a=plt.plot(l,data.T)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.legend(loc='best')
    else:
        print("波长文件与原始的额数据不匹配 \n请进行修改")
    
def nirplot_assign_high(data,iw,tw,s,xlabel,ylabel,title):
    l=np.arange(iw,tw,s)
    if data.shape[1]==l.shape[0]:
        print("波长文件和原始的数据匹配")
        a=plt.plot(l,data.T)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        #这里补充了大量的科研绘图的详细功能，比如：规定横轴、纵轴、图例等
        #设置字体 : int or float or {‘xx-small’, ‘x-small’, ‘small’, ‘medium’, ‘large’, ‘x-large’, ‘xx-large’}
        #位置选择best\upper right\upper left\lower left\lower right\right\center left\center right\lower center\upper center\center
        #plt.legend(loc='best')#设置位置
        #plt.legend(loc='best',frameon=False) #去掉图例边框
        #plt.legend(loc='best',edgecolor='blue') #设置图例边框颜色
        #plt.legend(loc='best',facecolor='blue') #设置图例背景颜色,若无边框,参数无效
        #plt.legend(["BJ", "SH"],loc='upper left')#设置两个图例
        #坐标轴设置
        plt.rc('font',family='Times New Roman')#字体
        plt.xticks(fontsize=14) #x坐标轴刻度字号
        plt.yticks(fontsize=14) #y坐标轴刻度字号
        plt.xlim(600,1800)
        plt.ylim(0,8)
        #ax=plt.gca();#获得坐标轴的句柄
        #ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
        #ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
        #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))#设置坐标轴科学计数法表示

    else:
        print("波长文件与原始的额数据不匹配 \n请进行修改")
#画评估图
def nirplot_eva_epoch(loss_iterm):
    plt.rcParams['agg.path.chunksize'] = 100000  # 设置 matplotlib 画出来曲线的平滑度
    plt.plot(loss_iterm)
    plt.xlabel("epochs")
    plt.ylabel("Training loss")
    plt.title("CNN Training Loss")
    plt.savefig("cnn_training_epoch_loss.png", dpi=300)  # matplotlib 将画出来的图片保存在本地，并且清晰度未300dpi

    
def nirplot_eva_iterations(loss_iterm):
    plt.rcParams['agg.path.chunksize'] = 100000  # 设置 matplotlib 画出来曲线的平滑度
    plt.plot(loss_iterm)
    plt.xlabel("Iterations")
    plt.ylabel("Training loss")
    plt.title("CNN Training Loss")
    plt.savefig("cnn_training_iterations_loss.png", dpi=300)  # matplotlib 将画出来的图片保存在本地，并且清晰度未300dpi

def nirplot_eva(Y,Y_pred):
    # z = np.polyfit(np.ravel(Y), np.ravel(Y_pred), 1)        
    ax1 = plt.subplot(1,1,1, aspect=1)
    ax1.scatter(Y,Y_pred,c='k',s=2)
    # ax1.plot(Y, z[1]+z[0]*Y, c='blue', linewidth=2,label='linear fit')
    ax1.plot(Y, Y, color='orange', linewidth=1.5, label='y=x')
    plt.ylabel('Predicted')
    plt.xlabel('Measured')
    plt.title('Prediction from PLS')
    plt.savefig("test.png", dpi=300)