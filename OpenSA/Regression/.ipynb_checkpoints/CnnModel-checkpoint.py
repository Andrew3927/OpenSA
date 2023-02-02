import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable


ac_dict = {
    'relu':nn.ReLU,
    'lrelu':nn.LeakyReLU

}


linear=[649,5184,5184,5184,5120,5120,5120,5120,4096,4096]
class AlexNet(nn.Module):
    def __init__(self,acti,c_num):
        super(AlexNet, self).__init__()
        self.layers=nn.ModuleList([])
        input_channel=1
        output_channel=16
        for i in range(1,c_num):
            self.layers.append(nn.Conv1d(input_channel,output_channel,3,padding=1))
            self.layers.append(nn.BatchNorm1d(num_features=output_channel))
            self.layers.append(ac_dict[acti](inplace=True))
            self.layers.append(nn.MaxPool1d(2,2))
            input_channel=output_channel
            output_channel=output_channel*2
        #linear[c_num-1]
        self.reg = nn.Sequential(
             nn.Linear(5120, 1000),  #根据自己数据集修改
             nn.ReLU(inplace=True),
             nn.Linear(1000, 500),
             nn.ReLU(inplace=True),
             nn.Dropout(0.5),
             nn.Linear(500, 1),)

    def forward(self,x):
        out = x
        for layer in self.layers:
            out = layer(out)         
        out = out.flatten(start_dim=1)
        # out = out.view(-1,self.output_channel)
        out = self.reg(out)
        return out

class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,out_C):
        super(Inception,self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(in_c, c1,kernel_size=1,padding=0),
            nn.Conv1d(c1, c1, kernel_size=3, padding=1)
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(in_c, c2,kernel_size=1,padding=0),
            nn.Conv1d(c2, c2, kernel_size=5, padding=2)

        )
        self.p3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3,stride=1,padding=1),
            nn.Conv1d(in_c, c3,kernel_size=3,padding=1),
        )
        self.conv_linear = nn.Conv1d((c1+c2+c3), out_C, 1, 1, 0, bias=True)
        self.short_cut = nn.Sequential()
        if in_c != out_C:
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_c, out_C, 1, 1, 0, bias=False),

            )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        out =  torch.cat((p1,p2,p3),dim=1)
        out += self.short_cut(x)
        return out




class DeepSpectra(nn.Module):
    def __init__(self,acti,c_num):
        super(DeepSpectra, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=3, padding=0)
        )
        self.Inception = Inception(16, 32, 32, 32, 96)
        self.fc = nn.Sequential(
            nn.Linear(20640, 5000),
            nn.Dropout(0.5),
            nn.Linear(5000, 1)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Inception(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample,acti):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            ac_dict[acti](inplace=True),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            ac_dict[acti](inplace=True),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            ac_dict[acti](inplace=True),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual

    
# class Resnet(torch.nn.Module):
#     def __init__(self,acti,c_num):
#         super(Resnet, self).__init__()
#         self.features = torch.nn.Sequential(
#             torch.nn.Conv1d(1,16,kernel_size=7,stride=2,padding=3),
#             torch.nn.MaxPool1d(3,2,1),
#             Bottlrneck(16,16,64,False,acti),
#             Bottlrneck(64,16,64,False,acti),
#             Bottlrneck(64,16,64,False,acti),
#         )
#         self.layers=nn.ModuleList([])
#         input_channel=64
#         output_channel=128
#         med_channel=32
#         for i in range(1,c_num):
#             self.layers.append(Bottlrneck(input_channel,med_channel,output_channel,(1==i)|(4==i)|(7==i),acti))
#             if (1==i)|(4==i)|(7==i):
#                 input_channel=input_channel*2
#             elif (3==i)|(6==i):
#                 med_channel=med_channel*2
#                 output_channel=output_channel*2
#         self.layers.append(torch.nn.AdaptiveAvgPool1d(1))
#         self.classifer = torch.nn.Sequential(
#             nn.Linear(2048, 1000),
#             nn.Dropout(0.5),
#             nn.Linear(1000, 1)
#         )


class Resnet(torch.nn.Module):
    def __init__(self,acti,c_num):
        super(Resnet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(1,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),
            Bottlrneck(64,64,256,False,acti),
            Bottlrneck(256,64,256,False,acti),
            Bottlrneck(256,64,256,False,acti),
        )
        self.layers=nn.ModuleList([])
        input_channel=256
        output_channel=512
        med_channel=128
        for i in range(1,c_num+1):
            self.layers.append(Bottlrneck(input_channel,med_channel,output_channel,(1==i)|(4==i)|(7==i),acti))
            if (1==(i%3)):
                input_channel=input_channel*2
            elif (0==(i%3)):
                med_channel=med_channel*2
                output_channel=output_channel*2
        self.layers.append(torch.nn.AdaptiveAvgPool1d(1))
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048,1)
        )

    def forward(self,x):
        x = self.features(x)
        out = x
        for layer in self.layers:
            out = layer(out)         
        out = out.view(-1,2048)
        out = self.classifer(out)
        return out


class DenseLayer(torch.nn.Module):
    def __init__(self,in_channels,acti,middle_channels=128,out_channels=32):
        super(DenseLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels),
            ac_dict[acti](inplace=True),
            torch.nn.Conv1d(in_channels,middle_channels,1),
            torch.nn.BatchNorm1d(middle_channels),
            ac_dict[acti](inplace=True),
            torch.nn.Conv1d(middle_channels,out_channels,3,padding=1)
        )
    def forward(self,x):
        return torch.cat([x,self.layer(x)],dim=1)


class DenseBlock(torch.nn.Sequential):

    def __init__(self,layer_num,growth_rate,in_channels,acti,middele_channels=128):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channels+i*growth_rate,acti,middele_channels,growth_rate)
            self.add_module('denselayer%d'%(i),layer)

class Transition(torch.nn.Sequential):
    def __init__(self,channels,acti):
        super(Transition, self).__init__()
        self.add_module('norm',torch.nn.BatchNorm1d(channels))
        self.add_module('relu',ac_dict[acti](inplace=True))
        self.add_module('conv',torch.nn.Conv1d(channels,channels//2,3,padding=1))
        self.add_module('Avgpool',torch.nn.AvgPool1d(2))


class DenseNet(torch.nn.Module):
    def __init__(self,acti,c_num):
        super(DenseNet, self).__init__()
        layer_num=(6,12,24,16)
        growth_rate=32
        init_features=64
        middele_channels=128
        self.feature_channel_num=init_features
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(1,self.feature_channel_num,7,2,3),
            torch.nn.BatchNorm1d(self.feature_channel_num),
            ac_dict[acti](inplace=True),
            torch.nn.MaxPool1d(3,2,1),
        )
        self.DenseBlock1=DenseBlock(layer_num[0],growth_rate,self.feature_channel_num,acti,middele_channels)
        self.feature_channel_num=self.feature_channel_num+layer_num[0]*growth_rate
        self.Transition1=Transition(self.feature_channel_num,acti)
        
        self.layers=nn.ModuleList([])
        for i in range(1,c_num+1):
            self.layers.append(DenseBlock(layer_num[i],growth_rate,self.feature_channel_num//2,acti,middele_channels))
            self.feature_channel_num=self.feature_channel_num//2+layer_num[i]*growth_rate
            if (i!=c_num):
                self.layers.append(Transition(self.feature_channel_num,acti))
        self.layers.append(torch.nn.AdaptiveAvgPool1d(1))
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(self.feature_channel_num, self.feature_channel_num//2),
            ac_dict[acti](inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.feature_channel_num//2, 1),

        )


    def forward(self,x):
        x = self.features(x)

        x = self.DenseBlock1(x)
        x = self.Transition1(x)
        out = x
        for layer in self.layers:
            out = layer(out)    
        out = out.view(-1,self.feature_channel_num)
        out = self.classifer(out)

        return out