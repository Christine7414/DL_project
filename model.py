import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
import musdb
import matplotlib.pyplot as plt
class ARC(nn.Module) :
    def __init__(self):
        super(ARC, self).__init__()
        self.Conv1 = nn.Conv1d(1025, 1024, 3, 1)
        self.Bn1 = nn.BatchNorm1d(1024)
        self.Conv2 = nn.Conv1d(1024, 512, 3, 2)
        self.Bn2 = nn.BatchNorm1d(512)
        self.Conv3 = nn.Conv1d(512, 256, 3, 2)
        self.Bn3 = nn.BatchNorm1d(256)
        self.TConv4 = nn.ConvTranspose1d(256, 512, 3, 2, output_padding = 1)
        self.Bn4 = nn.BatchNorm1d(512)
        self.TConv5 = nn.ConvTranspose1d(512, 1024, 3, 2, output_padding = 1)
        self.Bn5 = nn.BatchNorm1d(1024)
        self.Conv6 = nn.Conv1d(1024, 4*1025, 3, 1, padding = 2)
        self.gru1 = nn.GRU(1024, 1024, batch_first = True)
        self.gru2 = nn.GRU(512, 512, batch_first = True)
    def forward(self, x) :
        bs = x.shape[0]
        x = self.Conv1(x)
        x = F.leaky_relu(self.Bn1(x), 0.01)
        out1 = x.permute(0, 2, 1)
        out1, _ = self.gru1(out1)
        out1 = out1.permute(0, 2, 1)
        x = self.Conv2(x)
        x = F.leaky_relu(self.Bn2(x), 0.01)
        out2 = x.permute(0, 2, 1)
        out2, _ = self.gru2(out2)
        out2 = out2.permute(0, 2, 1)
        x = self.Conv3(x)
        x = F.leaky_relu(self.Bn3(x), 0.01)
        x = self.TConv4(x)
        x = F.leaky_relu(self.Bn4(x), 0.01) + out2
        x = self.TConv5(x)
        x = F.leaky_relu(self.Bn5(x), 0.01) + out1
        x = self.Conv6(x)
        x = F.leaky_relu(x, 0.01)
        return x 

class Enhancement(nn.Module) :
    def __init__(self):
        super(Enhancement, self).__init__()
        self.Conv1 = nn.Conv1d(1025, 1024, 3, 1)
        self.Bn1 = nn.BatchNorm1d(1024)
        self.Conv2 = nn.Conv1d(1024, 512, 3, 2)
        self.Bn2 = nn.BatchNorm1d(512)
        self.Conv3 = nn.Conv1d(512, 256, 3, 2)
        self.Bn3 = nn.BatchNorm1d(256)
        self.TConv4 = nn.ConvTranspose1d(256, 512, 3, 2, output_padding = 1)
        self.Bn4 = nn.BatchNorm1d(512)
        self.TConv5 = nn.ConvTranspose1d(512, 1024, 3, 2, output_padding = 1)
        self.Bn5 = nn.BatchNorm1d(1024)
        self.Conv6 = nn.Conv1d(1024, 4*1025, 3, 1, padding = 2)
        self.CConv1 = nn.Conv1d(1024, 1024, 3, 1, padding = 1)
        self.CBn1 = nn.BatchNorm1d(1024)
        self.CConv2 = nn.Conv1d(512, 512, 3, 1, padding = 1)
        self.CBn2 = nn.BatchNorm1d(512)
    def forward(self, x) :
        bs = x.shape[0]
        x = self.Conv1(x)
        x = F.leaky_relu(self.Bn1(x), 0.01)
        out1 = self.CConv1(x)
        out1 = F.leaky_relu(self.CBn1(out1), 0.01)
        x = self.Conv2(x)
        x = F.leaky_relu(self.Bn2(x), 0.01)
        out2 = self.CConv2(x)
        out2 = F.leaky_relu(self.CBn2(out2), 0.01)
        x = self.Conv3(x)
        x = F.leaky_relu(self.Bn3(x), 0.01)
        x = self.TConv4(x)
        x = F.leaky_relu(self.Bn4(x), 0.01) + out2
        x = self.TConv5(x)
        x = F.leaky_relu(self.Bn5(x), 0.01) + out1
        x = self.Conv6(x)
        x = F.leaky_relu(x, 0.01)
        return x       

