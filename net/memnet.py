
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .blocks import *
dtype = torch.cuda.FloatTensor 




class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.LeakyReLU(inplace=inplace))  
        self.add_module('conv', nn.Conv2d(in_channels, channels, 3, 1, 1))  


class BNReLUMamba(nn.Sequential):
    def __init__(self, in_channels, channels, drop_path, H, W, inplace=True):
        super(BNReLUMamba, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.LeakyReLU(inplace=inplace))  #
        self.add_module('mamba', SF_Block(in_channels, channels, drop_path, H, W))  

class GateUnit(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(GateUnit, self).__init__()
        self.add_module('bn',nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.LeakyReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels,1,1,0))


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels, drop_path, H, W):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUMamba(channels, channels, drop_path, H, W, True)
        self.relu_conv2 = BNReLUMamba(channels, channels, drop_path, H, W, True)
        
    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class MemoryBlock(nn.Module):
    def __init__(self, channels, num_resblock, num_memblock, drop_path, H, W):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels, drop_path, H, W) for i in range(num_resblock)]
        )
        self.gate_unit = GateUnit((num_resblock+num_memblock) * channels, channels, True) 

    def forward(self, x, ys):
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
        gate_out = self.gate_unit(torch.cat(xs+ys, 1))  
        ys.append(gate_out)
        return gate_out



class MemNet3(nn.Module):
    def __init__(self, in_channels=3, channels=16, num_memblock=6, num_resblock=6, drop_path=0.1, H=100, W=100, m1=2, m2=4):
        super(MemNet3, self).__init__()

        self.extra_conv1 = BNReLUConv(in_channels, channels*2, True)
        self.extra_conv2 = BNReLUConv(channels*2, channels * 2, True)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.extra_conv3 = BNReLUConv(channels * 2, channels * 4, True)  

        self.recons_conv1 = BNReLUConv(channels * 4, channels * 2, True)  
        self.recons_conv2 = BNReLUConv(channels * 2, channels*2, True)  
        self.recons_conv3 = BNReLUConv(channels*2, in_channels, True)  

        self.fusion = BNReLUConv(channels * 4, channels * 4, True)

        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels * 4, num_resblock, i + 1, drop_path, H // 4, W // 4) for i in range(num_memblock)]
        )


        self.weights = nn.Parameter((torch.ones(1, num_memblock) / num_memblock), requires_grad=True)
        


    def forward(self, x):
        residual0 = x
        out = self.extra_conv1(x)
        residual1 = out
        out = self.extra_conv2(out)
        out = self.pool(out)
        residual2 = out
        out = self.extra_conv3(out)
        out = self.pool(out)
        residual3 = out


        w_sum = self.weights.sum(1)
        mid_feat = []  
        ys = [out]  
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)  
            mid_feat.append(out)

        pred = (self.fusion(mid_feat[0]) + residual3) * self.weights.data[0][0] / w_sum
        for i in range(1, len(mid_feat)):
            pred = pred + (self.fusion(mid_feat[i]) + residual3) * self.weights.data[0][i] / w_sum



        pred = F.interpolate(pred, scale_factor=2)
        pred = self.recons_conv1(pred)
        pred = residual2 + pred
        pred = F.interpolate(pred, scale_factor=2)
        pred = self.recons_conv2(pred)
        pred = residual1 + pred
        pred = self.recons_conv3(pred)
        pred = residual0 + pred

        return pred





