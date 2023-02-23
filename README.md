# OpenSA
Aiming at the common training datsets split, spectrum preprocessing, wavelength select and calibration models algorithm involved in the spectral analysis process, a complete algorithm library is established, which is named opensa (openspectrum analysis).
# Series Article Directory
<font size=4 color=Red>"The essence of light is clear, the spectrum reveals differences." Spectroscopy, as the fingerprint of matter, is widely used in component analysis. With the development and popularization of miniaturized spectrometers/spectral imaging devices, analysis techniques based on spectroscopy will not only be confined to industry and laboratories, but will also enter daily life, enabling perception of all things and revealing the subtle. This series of articles is dedicated to popularizing and applying spectroscopic analysis technology.
<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">




@[TOC](Table of Content)

</font>

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# Introduction
Spectra, as the fingerprint of substances, are widely used in component analysis due to their ability to reveal the essence of light. With the development and popularization of miniature spectrometers/spectral imaging instruments, spectral analysis techniques are not limited to industrial and laboratory settings but are being introduced into daily life to achieve perception of all things and understand the subtleties. This series of articles aims to popularize and apply spectral analysis techniques.

The typical spectral analysis model (using near-infrared spectroscopy as an example, the analysis process for visible light, mid-infrared, fluorescence, Raman, hyperspectral, and other spectra are similar) is established as follows: during the building process, algorithms are used to select training samples, preprocess spectra, or extract spectral features. A calibration model is then constructed to achieve quantitative analysis, followed by model transfer or transmission for different measuring instruments or environments. Therefore, the selection of training samples, spectral preprocessing, wavelength selection, calibration model, model transfer, and parameters of the algorithms all affect the application effect of the model.

![图 1近红外光谱建模及应用流程](https://img-blog.csdnimg.cn/e4038170fff643468cacfed4fb34ab04.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBARWNob19Db2Rl,size_20,color_FFFFFF,t_70,g_se,x_16)
A complete algorithm library called OpenSA (Open Spectrum Analysis) has been developed to establish algorithms for the common sample division, spectral preprocessing, wavelength selection, and calibration model algorithms involved in the spectral analysis process. The architecture of the entire algorithm library is shown below.
![在这里插入图片描述](https://img-blog.csdnimg.cn/cf63e5d8980542bf824cb889d01f2e00.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBARWNob19Db2Rl,size_20,color_FFFFFF,t_70,g_se,x_16)
The sample division module provides three types of data set division methods: random division, SPXY division, and KS division. The spectral preprocessing module provides common spectral preprocessing methods. The wavelength selection module provides feature dimensionality reduction methods such as Spa, Cars, Lars, Uve, and Pca. The analysis module includes spectral similarity calculation, clustering, classification (qualitative analysis), and regression (quantitative analysis). The spectral similarity sub-module calculation provides similarity calculation methods such as SAM, SID, MSSIM, and MPSNR. The clustering sub-module provides clustering methods such as KMeans and FCM. The classification sub-module provides classic chemometrics methods such as ANN, SVM, PLS_DA, and RF, as well as cutting-edge deep learning methods such as CNN, AE, and Transformer. The regression sub-module provides classic chemometrics quantitative analysis methods such as ANN, SVR, and PLS, as well as cutting-edge deep learning quantitative analysis methods such as CNN, AE, and Transformer. The model evaluation module provides common evaluation indicators for model evaluation. The automatic parameter optimization module is used to automatically find the best model setting parameters, and provides three methods for finding the optimal parameters: grid search, genetic algorithm, and Bayesian probability. The visualization module provides full-process analysis visualization, which provides visual information for scientific research and model selection. The complete spectral analysis and application can be quickly implemented with just a few lines of code.

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">


<font  size=5 color=bule >This article focuses on the code open-source and usage demonstration of the OpenSA spectral preprocessing module.
# Update log 20220521
Improved OpenSA by:

Adding genetic algorithm (GA) to the wavelength selection algorithm
Adding ELM, regular convolutional neural network to the quantitative analysis algorithm
Also, reproduced the network DeepSpectra in Zone 1 paper and the network 1-D ALENET in Zone 2 paper.                                                            

# 1. Spectral data input
Two open-source datasets are provided as examples, one for public quantitative analysis and one for public qualitative analysis. This chapter only uses the public quantitative analysis dataset for demonstration.


##  1.1 Spectral data input
                                                                         

```python
# Using one regression and one classification public dataset as examples
def LoadNirtest(type):

    if type == "Rgs":
        CDataPath1 = './/Data//Rgs//Cdata1.csv'
        VDataPath1 = './/Data//Rgs//Vdata1.csv'
        TDataPath1 = './/Data//Rgs//Tdata1.csv'

        Cdata1 = np.loadtxt(open(CDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        Vdata1 = np.loadtxt(open(VDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        Tdata1 = np.loadtxt(open(TDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)

        Nirdata1 = np.concatenate((Cdata1, Vdata1))
        Nirdata = np.concatenate((Nirdata1, Tdata1))
        data = Nirdata[:, :-4]
        label = Nirdata[:, -1]

    elif type == "Cls":
        path = './/Data//Cls//table.csv'
        Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        data = Nirdata[:, :-1]
        label = Nirdata[:, -1]

    return data, label

```
##  1.2 Spectral Visualization
```python
    #Load the original data and visualize it
    data, label = LoadNirtest('Rgs')
    plotspc(data, "raw specturm")
```
The open source spectra used are shown in the figure below:
![原始光谱](https://img-blog.csdnimg.cn/04a9549619fd48198c9072c2d1acfd99.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBARWNob19Db2Rl,size_20,color_FFFFFF,t_70,g_se,x_16)

# 2. Spectral Preprocessing
##  2.1 Spectral Preprocessing Module
Common spectra have been encapsulated, and users only need to change the name to select the corresponding spectral analysis. The following is the core code of the spectral preprocessing module.
```python
"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github :
    @WeChat : Fu_siry
    @License：

"""
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from copy import deepcopy
import pandas as pd
import pywt


# Max-min normalization
def MMS(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MinMaxScaler :(n_samples, n_features)
       """
    return MinMaxScaler().fit_transform(data)


# Standardization
def SS(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after StandScaler :(n_samples, n_features)
       """
    return StandardScaler().fit_transform(data)


# Mean centering
def CT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MeanScaler :(n_samples, n_features)
       """
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        data[i] = data[i] - MEAN
    return data


# Standard normal transformation
def SNV(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after SNV :(n_samples, n_features)
    """
    m = data.shape[0]
    n = data.shape[1]
    print(m, n)  #
    # Calculation of standard deviation
    data_std = np.std(data, axis=1)  # Standard deviation of each spectrum
    # Calculation of mean
    data_average = np.mean(data, axis=1)  # Mean of each spectrum
    # SNV calculation
    data_snv = [[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    return  data_snv



# Moving average smoothing
def MA(data, WSZ=11):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param WSZ: int
       :return: data after MA :(n_samples, n_features)
    """

    for i in range(data.shape[0]):
        out0 = np.convolve(data[i], np.ones(WSZ, dtype=int), 'valid') / WSZ # WSZ is the window width and it is odd
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(data[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(data[i, :-WSZ:-1])[::2] / r)[::-1]
        data[i] = np.concatenate((start, out0, stop))
    return data


# Savitzky-Golay smoothing filter
def SG(data, w=11, p=2):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param w: int
       :param p: int
       :return: data after SG :(n_samples, n_features)
    """
    return signal.savgol_filter(data, w, p)


# First-order derivative
def D1(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after First derivative :(n_samples, n_features)
    """
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di


# Second-order derivative
def D2(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after second derivative :(n_samples, n_features)
    """
    data = deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    temp2 = (pd.DataFrame(data)).diff(axis=1)
    temp3 = np.delete(temp2.values, 0, axis=1)
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)
    return spec_D2


# Trend correction (DT)
def DT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after DT :(n_samples, n_features)
    """
    lenth = data.shape[1]
    x = np.asarray(range(lenth), dtype=np.float32)
    out = np.array(data)
    l = LinearRegression()
    for i in range(out.shape[0]):
        l.fit(x.reshape(-1, 1), out[i].reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] - (j * k + b)

    return out


# Multivariate scatter correction
def MSC(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MSC :(n_samples, n_features)
    """
    n, p = data.shape
    msc = np.ones((n, p))

    for j in range(n):
        mean = np.mean(data, axis=0)

    # Linear fitting
    for i in range(n):
        y = data[i, :]
        l = LinearRegression()
        l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        msc[i, :] = (y - b) / k
    return msc

# Wavelet transformation
def wave(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after wave :(n_samples, n_features)
    """
    data = deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    def wave_(data):
        w = pywt.Wavelet('db8')  # Daubechies8 wavelet is selected
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)
        threshold = 0.04
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, 'db8')
        return datarec

    tmp = None
    for i in range(data.shape[0]):
        if (i == 0):
            tmp = wave_(data[i])
        else:
            tmp = np.vstack((tmp, wave_(data[i])))

    return tmp

def Preprocessing(method, data):

    if method == "None":
        data = data
    elif method == 'MMS':
        data = MMS(data)
    elif method == 'SS':
        data = SS(data)
    elif method == 'CT':
        data = CT(data)
    elif method == 'SNV':
        data = SNV(data)
    elif method == 'MA':
        data = MA(data)
    elif method == 'SG':
        data = SG(data)
    elif method == 'MSC':
        data = MSC(data)
    elif method == 'D1':
        data = D1(data)
    elif method == 'D2':
        data = D2(data)
    elif method == 'DT':
        data = DT(data)
    elif method == 'WVAE':
        data = wave(data)
    else:
        print("no this method of preprocessing!")

    return data


```
## 2.2 Use of Spectral Preprocessing
In the example.py file, the usage of the spectral preprocessing module is provided, as shown below. Only two lines of code are required to implement all common spectral preprocessing.
Example 1: Implement MSC multivariate scatter correction using OpenSA
```python
 #Load the original data and visualize it
    data, label = LoadNirtest('Rgs')
    plotspc(data, "raw specturm")
    #Spectral preprocessing and visualization
    method = "MSC"
    Preprocessingdata = Preprocessing(method, data)
    plotspc(Preprocessingdata, method)
```
The preprocessed spectral data is shown in the figure below:
![在这里插入图片描述](https://img-blog.csdnimg.cn/3b38f01e6ebe4a22821274bca50aa5a2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBARWNob19Db2Rl,size_20,color_FFFFFF,t_70,g_se,x_16)


Example 2: Implement SNV preprocessing using OpenSA

```python
    #Load the original data and visualize it
    data, label = LoadNirtest('Rgs')
    plotspc(data, "raw specturm")
    #Spectral preprocessing and visualization
    method = "SNV"
    Preprocessingdata = Preprocessing(method, data)
    plotspc(Preprocessingdata, method)
```
The preprocessed spectral data is shown in the figure below:
![SNV](https://img-blog.csdnimg.cn/558d1c710da04519b72cab08da67e9cc.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBARWNob19Db2Rl,size_20,color_FFFFFF,t_70,g_se,x_16)
# Conclusion
<font color=#999AAA >OpenSA can be used to easily implement spectral preprocessing. The complete code can be obtained from the GitHub repository. If it is useful to you, please give it a like! The code is currently only for academic use. If it is helpful for your academic research, please cite my paper. Unauthorized use for commercial applications is prohibited. You are welcome to continue supplementing the algorithms involved in OpenSA.
