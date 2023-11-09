import math
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class SpatialGCN(nn.Module):
    def __init__(self, plane,inter_plane=None,out_plane=None):
        super(SpatialGCN, self).__init__()
        if inter_plane==None:
            inter_plane = plane #// 2
        if out_plane==None:
            out_plane = plane
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.conv_wgl = nn.Linear(inter_plane,out_plane)
        self.bn1 = nn.BatchNorm1d(out_plane)
        self.conv_wgl2 = nn.Linear(out_plane, out_plane)
        self.bn2 = nn.BatchNorm1d(out_plane)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        node_k = self.node_k(
            x)  # x#copy.deepcopy(x)#F.normalize(x,p=1,dim=-1)   #####nosym better, softmax better,only one gcn better
        node_q = self.node_q(x)  # x#copy.deepcopy(x)#F.normalize(x,p=1,dim=-1)#
        # print("input:",x.shape,node_k.shape)
        node_v = self.node_v(x)  # x#
        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)  ##b N C
        node_q = node_q.view(b, c, -1)  ###b c N
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)  ##b N C
        Adj = torch.bmm(node_k, node_q)  ###Q*K^T

        # test using cosine=(a*b)/||a||*||b|| to construct adjacency
        # Adj = torch.bmm(node_k,node_q)#ab_ij=node_i*node_j
        # batch_row_norm = torch.norm(node_k,dim=-1).unsqueeze(-1)
        # Adj = torch.div(Adj,torch.bmm(batch_row_norm,batch_row_norm.permute(0,2,1)))

        Adj = self.softmax(Adj)  ###adjacency matrix of size b N N

        # max = torch.max(Adj, dim=2)
        # min = torch.min(Adj, dim=2)
        # Adj = (Adj - min.values[:, :, None]) / max.values[:, :, None]  # normalized adjacency matrix
        # Adj[Adj<0.5]=0

        AV = torch.bmm(Adj,node_v)###AX
        AVW = F.relu(self.bn1(self.conv_wgl(AV).transpose(1,2)).transpose(1,2))###AXW b n C
        AVW = F.dropout(AVW)
        # add one more layer
        AV = torch.bmm(Adj,AVW)
        AVW = F.relu(self.bn2(self.conv_wgl2(AV).transpose(1,2)).transpose(1,2))
        AVW = F.dropout(AVW)
        # end
        AVW = AVW.transpose(1, 2).contiguous()###AV withj shape NxC,N=mxn
        b,c,n = AVW.shape
        AVW = AVW.view(b, c, h, -1)
        return AVW

class VGUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2,base_nc=64,fix_grad=True):
        super(VGUNet, self).__init__()
        self.fix_grad = fix_grad
        self.conv1 = DoubleConv(in_ch, base_nc)
        self.pool1 = nn.Conv2d(base_nc, base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv2 = DoubleConv(base_nc, 2 * base_nc)
        self.pool2 = nn.Conv2d(2 * base_nc, 2 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv3 = DoubleConv(2 * base_nc, 4 * base_nc)
        self.pool3 = nn.Conv2d(4 * base_nc, 4 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        # if self.fix_grad==True:
        #     for p in self.parameters():
        #         p.requires_grad=False
        self.sgcn3 = SpatialGCN(2 * base_nc)
        self.sgcn2 = SpatialGCN(4 * base_nc)
        self.sgcn1 = SpatialGCN(4 * base_nc)  ###changed with spatialGCN
        self.up6 = nn.ConvTranspose2d(4 * base_nc, 4 * base_nc, 2, stride=2,padding=0)  ##upsampling
        self.conv6 = DoubleConv(8 * base_nc, 4 * base_nc)
        self.up7 = nn.ConvTranspose2d(4 * base_nc, 2 * base_nc, 2, stride=2, padding=0)  ##upsampling
        self.conv7 = DoubleConv(4 * base_nc, 2 * base_nc)
        self.up8 = nn.ConvTranspose2d(2 * base_nc, base_nc, 2, stride=2,padding=0)  ##upsampling
        self.conv8 = DoubleConv(2 * base_nc, base_nc)
        self.conv9 = nn.Conv2d(base_nc, out_ch, kernel_size=1, padding=0)
    def forward(self,x):
        c1=self.conv1(x)  ## 2 nc
        p1=self.pool1(c1)  ##
        c2=self.conv2(p1) ##nc 2nc
        p2=self.pool2(c2)
        c3=self.conv3(p2) ##2nc 2nc
        p3=self.pool3(c3)
        c4=self.sgcn1(p3)   ###spatial gcn 4nc
        up_6= self.up6(c4)
        merge6 = torch.cat([up_6, self.sgcn2(c3)], dim=1)##gcn
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, self.sgcn3(c2)], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c1], dim=1)
        c8=self.conv8(merge8)
        c9= self.conv9(c8)

        return c9

