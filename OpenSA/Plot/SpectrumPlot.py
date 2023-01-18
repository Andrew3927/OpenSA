import matplotlib.pyplot as plt
import numpy as np


# def plotspc(data, title):
def plotspc(title):
    CDataPath1 = './Data/Rgs/Cdata1.csv'
    VDataPath1 = './Data/Rgs/Vdata1.csv'
    TDataPath1 = './Data/Rgs/Tdata1.csv'
    Cdata1 = np.loadtxt(open(CDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    Vdata1 = np.loadtxt(open(VDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    Tdata1 = np.loadtxt(open(TDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    Nirdata1 = np.concatenate((Cdata1, Vdata1))
    Nirdata = np.concatenate((Nirdata1, Tdata1))
    l = np.loadtxt(open('./Data/Rgs/l.csv', 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    data = Nirdata[:, :-3]
    label = Nirdata[:, -3:-1]
    l = l[1:, ]
    plt.plot(l, data[0:50].T)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel('Intensity')
    plt.title(title)
    plt.show()