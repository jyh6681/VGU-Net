import torch
from torch import nn
from gcn_lib import Grapher

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d


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



class VGUnet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2,base_nc=64,fix_grad=True):
        super(VGUnet, self).__init__()
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
        self.sgcn3 = Grapher(2 * base_nc, kernel_size=200, dilation=1, conv='gin', act='relu', norm=None,bias=True,  
                            stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False)
        self.sgcn2 = Grapher(4 * base_nc, kernel_size=80, dilation=1, conv='gin', act='relu', norm=None,bias=True,  
                            stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False)
        self.sgcn1 = Grapher(4 * base_nc, kernel_size=20, dilation=1, conv='gin', act='relu', norm=None,bias=True,  
                            stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False)
        self.up6 = nn.ConvTranspose2d(4 * base_nc, 4 * base_nc, 2, stride=2,padding=0)  ##upsampling
        self.conv6 = DoubleConv(8 * base_nc, 4 * base_nc)
        self.up7 = nn.ConvTranspose2d(4 * base_nc, 2 * base_nc, 2, stride=2, padding=0)  ##upsampling
        self.conv7 = DoubleConv(4 * base_nc, 2 * base_nc)
        self.up8 = nn.ConvTranspose2d(2 * base_nc, base_nc, 2, stride=2,padding=0)  ##upsampling
        self.conv8 = DoubleConv(base_nc, base_nc)#DoubleConv(2 * base_nc, base_nc)
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
        merge8 = up_8#torch.cat([up_8, c1], dim=1)
        c8=self.conv8(merge8)
        c9= self.conv9(c8)

        return c9
