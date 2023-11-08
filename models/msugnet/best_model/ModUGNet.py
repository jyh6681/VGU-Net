import math
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from baseline_model import basicblock as B

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


class FeatureGCN(nn.Module):
    """
        Feature GCN
    """
    def __init__(self, planes, ratio=2):
        super(FeatureGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))   ###get the new VF 128*64

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ  ###adjacency matrix
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        # print("bbbb:::::", z_idt.shape,z.shape, b.shape,x_sqz.shape)
        y = torch.matmul(z, b)
        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)
        # print("shape:::::",feat.shape,x.shape,y.shape)
        g_out = F.relu_(x+y)
        return g_out

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
        self.bn1 = BatchNorm1d(out_plane)
        self.conv_wgl2 = nn.Linear(out_plane, out_plane)
        self.bn2 = BatchNorm1d(out_plane)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, test_mode=False):
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

        ####cosine=(a*b)/||a||*||b||
        # Adj = torch.bmm(node_k,node_q)#ab_ij=node_i*node_j
        # batch_row_norm = torch.norm(node_k,dim=-1).unsqueeze(-1)
        # Adj = torch.div(Adj,torch.bmm(batch_row_norm,batch_row_norm.permute(0,2,1)))

        Adj = self.softmax(Adj)  ###adjacency matrix of size b N N

        max = torch.max(Adj, dim=2)
        min = torch.min(Adj, dim=2)
        Adj = (Adj - min.values[:, :, None]) / max.values[:, :, None]  # normalized adjacency matrix
        # Adj[Adj<0.5]=0

        # ##attention map
        if test_mode == True:
            tmp = (Adj.squeeze()).cpu().numpy()
            np.save("./output/hair_adjmap" + str(int(224 / h)) + "_brats.npy", tmp)
            print("adj saved!")
            tmp = Adj.squeeze().cpu().numpy()
            reducer = umap.UMAP(random_state=42, n_neighbors=50, min_dist=0.1)  ##UMAP
            embedding = reducer.fit_transform(tmp)
            plt.scatter(embedding[:, 0], embedding[:, 1], cmap="Spectral", s=5)
            plt.gca().set_aspect('equal', 'datalim')
            # plt.colorbar()
            plt.title('UMAP projection of the Digits dataset')
            plt.savefig("./output/umap" + str(int(224 / h)) + "_brats.png")
        ##plt.show()

        AV = torch.bmm(Adj,node_v)###AX
        AVW = F.relu(self.bn1(self.conv_wgl(AV).transpose(1,2)).transpose(1,2))###AXW b n C
        AVW = F.dropout(AVW)
        #####add one more layer
        AV = torch.bmm(Adj,AVW)
        AVW = F.relu(self.bn2(self.conv_wgl2(AV).transpose(1,2)).transpose(1,2))
        AVW = F.dropout(AVW)
        ##end
        AVW = AVW.transpose(1, 2).contiguous()###AV withj shape NxC,N=mxn
        b,c,n = AVW.shape
        #AVW = nn.GroupNorm(4, 1444, eps=1e-6)(AVW)
        AVW = AVW.view(b, c, h, -1)

        return AVW

class UnsupGCN(nn.Module):
    def __init__(self, plane):
        super(UnsupGCN, self).__init__()
        inter_plane = plane #// 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.matrixnorm =BatchNorm1d(1024)

        #self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.conv_wgl = nn.Linear(inter_plane,inter_plane)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_q = self.node_k(x)
        node_v = self.node_v(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0,2,1)   ##b N C
        node_q = node_q.view(b, c, -1)  ###b c N
        node_v = node_v.view(b, c, -1).permute(0,2,1) ##b N C

        AV = torch.bmm(node_k,node_q)   ###Q*K^T adj
        # print("show1:", AV[0,0:10,0:10])
        # print("adj matrix:",AV.shape)#AV[0,1,2],AV[0,2,1])
        AV = self.matrixnorm(AV)
        AV = self.softmax(AV)
        # print("show2:",AV[0,0:10,0:10])

        # ##attention map
        tmp = (AV.squeeze()).cpu().numpy()
        feat = (node_v.squeeze()).cpu().numpy()
        np.save("./unsup_saved_model/adjmapbear2.npy",tmp)
        # np.save("./unsup_saved_model/featbear2.npy", feat)
        print("adj saved!")


        ##normalized adjacency matrix

        ###UMAP
        # tmp = node_v.squeeze().cpu().numpy()
        # reducer = umap.UMAP(random_state=42,n_neighbors=50,min_dist=0.1)
        # embedding = reducer.fit_transform(tmp)
        # plt.scatter(embedding[:,0],embedding[:,1],cmap = "Spectral",s=5)
        # plt.gca().set_aspect('equal', 'datalim')
        # # plt.colorbar()
        # plt.title('UMAP projection of the Digits dataset')
        # plt.show()

        # eye = torch.zeros_like(AV).cuda()
        # deg_mat = torch.zeros_like(AV).cuda()
        # for i in range(0, AV.shape[0]):
        #     eye[i, :, :] = torch.eye(AV.shape[1])
        #     deg_mat[i,:,:] = torch.diag(1/torch.sqrt(1+torch.sum(AV[i,:,:],dim=0)))
        # AV = AV + eye
        # AV = torch.bmm(deg_mat,AV)
        # AV = torch.bmm(AV,deg_mat)

        AV = torch.bmm(AV,node_v)
        #AV = AV.transpose(1, 2).contiguous()   ###AV withj shape NxC,N=mxn
        #print("shapeAV:",AV.shape)
        AVW = self.conv_wgl(AV)
        AVW = AVW.transpose(1, 2).contiguous()
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        return F.relu(AVW)


class GaussianGCN(nn.Module):
    def __init__(self, plane):
        super(GaussianGCN, self).__init__()
        inter_plane = plane #// 2

        # self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.conv_wgl = nn.Linear(inter_plane,inter_plane)
        self.bn_wg = BatchNorm1d(inter_plane)

    def forward(self, x):
        # b, c, h, w = x.size()
        b,c,h,w = x.size()
        node_k = x.view(b, c, -1).permute(0, 2, 1)   ##b N C
        node_q = x.view(b, c, -1)  ###b c N
        AV = torch.bmm(node_k,node_q)   ###Q*K^T
        for n in range(0,node_k.shape[1]):
            for j in range(n,node_k.shape[1]):
                AV[:,n,j] = torch.exp(torch.sum(torch.pow(node_k[:,n,:]-node_k[:,j,:],2),dim=1)/(-2*math.pi))
                AV[:,j,n] = AV[:,n,j]
        norm = 1
        if norm==1:
            eye = torch.zeros_like(AV)
            deg_mat = torch.zeros_like(AV)
            for i in range(0, AV.shape[0]):
                eye[i, :, :] = torch.eye(AV.shape[1])
                deg_mat[i,:,:] = torch.diag(1/torch.sqrt(1+torch.sum(AV[i,:,:],dim=0)))
            AV = AV + eye
            AV = torch.bmm(deg_mat,AV)
            AV = torch.bmm(AV,deg_mat)
        AV = torch.bmm(AV,node_k)
        # AV = AV.transpose(1, 2).contiguous()   ###AV withj shape NxC,N=mxn
        #print("shapeAV:",AV.shape)
        AVW = self.conv_wgl(AV)
        AVW = AVW.transpose(1, 2).contiguous()
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        return AVW



class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, nc, mode='C', bias=bias)             ##########nc should adapt with dual graph net


        self.model = B.sequential(m_head, *m_body, m_tail)
        self.add = B.conv(nc, out_nc, mode='C', bias=bias)
        #self.head = DualGCNHead(nc, nc, out_nc)

    def forward(self, x):
        n = self.model(x)   ####n has shape [batch,64,128,128]
        n = self.add(n)

        return n

class DUGnet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(DUGnet, self).__init__()
        base_nc = 64
        self.conv1 = DoubleConv(in_ch, base_nc)
        self.pool1 = nn.Conv2d(base_nc, base_nc, 2,stride=2, padding=0, bias=False)##downsampling
        self.conv2 = DoubleConv(base_nc, 2*base_nc)
        self.pool2 = nn.Conv2d(2*base_nc, 2*base_nc, 2,stride=2, padding=0, bias=False)##downsampling
        self.conv3 = DoubleConv(2*base_nc, 4*base_nc)
        self.pool3 = nn.Conv2d(4*base_nc, 4*base_nc, 2,stride=2, padding=0, bias=False)##downsampling
        self.sgcn = UnsupGCN(4*base_nc)   ###changed with spatialGCN
        self.up6_double = nn.ConvTranspose2d(4*base_nc, 4*base_nc, 2, stride=2)##upsampling
        self.conv6 = DoubleConv(8*base_nc, 4*base_nc)
        self.up7 = nn.ConvTranspose2d(4*base_nc, 2*base_nc, 2, stride=2)##upsampling
        self.conv7 = DoubleConv(4*base_nc, 2*base_nc)
        self.up8 = nn.ConvTranspose2d(2*base_nc, base_nc, 2, stride=2)##upsampling
        self.conv8 = DoubleConv(2*base_nc, base_nc)
        self.conv9 = nn.Conv2d(base_nc, out_ch, kernel_size=3,padding=0)
    def forward(self,x):
        c1=self.conv1(x)  ## 2 nc
        p1=self.pool1(c1)  ##
        c2=self.conv2(p1) ##nc 2nc
        p2=self.pool2(c2)
        c3=self.conv3(p2) ##2nc 2nc
        p3=self.pool3(c3)
        c4=self.sgcn(p3)   ###spatial gcn

        up_6= self.up6_double(c4)

        merge6 = torch.cat([up_6, c3], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c2], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c1], dim=1)
        c8=self.conv8(merge8)
        c9= self.conv9(c8)

        #c9 = self.batchnorm2(c9)
        #c9 = self.act(c9)
        return c9

class UGnet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(UGnet, self).__init__()
        base_nc = 32
        self.conv1 = DoubleConv(in_ch, base_nc)
        self.pool1 = nn.Conv2d(base_nc, base_nc, 2,stride=2, padding=0, bias=False)##downsampling
        self.conv2 = DoubleConv(base_nc, 2*base_nc)
        self.pool2 = nn.Conv2d(2*base_nc, 2*base_nc, 2,stride=2, padding=0, bias=False)##downsampling
        self.conv3 = DoubleConv(2*base_nc, 4*base_nc)
        self.pool3 = nn.Conv2d(4*base_nc, 4*base_nc, 2,stride=2, padding=0, bias=False)##downsampling
        self.sgcn2= SpatialGCN(4*base_nc)   ###changed with spatialGCN
        self.fgcn = FeatureGCN(4*base_nc)   ###changed with FeatureGCN
        self.up6_double = nn.ConvTranspose2d(4*base_nc, 4*base_nc, 2, stride=2)##upsampling
        self.conv6 = DoubleConv(8*base_nc, 4*base_nc)
        self.up7 = nn.ConvTranspose2d(4*base_nc, 2*base_nc, 2, stride=2,padding=0)##upsampling
        self.conv7 = DoubleConv(4*base_nc, 2*base_nc)
        self.up8 = nn.ConvTranspose2d(2*base_nc, base_nc, 2, stride=2)##upsampling
        self.conv8 = DoubleConv(2*base_nc, base_nc)
        self.conv9 = nn.Conv2d(base_nc, out_ch, kernel_size=1,padding=0)
    def forward(self,x):
        c1=self.conv1(x)  ## 2 nc
        p1=self.pool1(c1)  ##
        c2=self.conv2(p1) ##nc 2nc
        p2=self.pool2(c2)
        c3=self.conv3(p2) ##2nc 2nc
        p3=self.pool3(c3)
        c4=self.sgcn2(p3)   ###spatial gcn
        # feat_c4 = self.fgcn(p3)
        # c4 = torch.cat((c4,feat_c4),dim=1)

        up_6= self.up6_double(c4)

        merge6 = torch.cat([up_6, c3], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c2], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c1], dim=1)
        c8=self.conv8(merge8)
        c9= self.conv9(c8)

        #c9 = self.batchnorm2(c9)
        #c9 = self.act(c9)
        return c9

class PlainUnet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1,base_nc=64):
        super(PlainUnet, self).__init__()
        self.conv1 = DoubleConv(in_ch, base_nc)
        self.pool1 = nn.Conv2d(base_nc, base_nc, 2,stride=2, padding=0, bias=False)##downsampling
        self.conv2 = DoubleConv(base_nc, 2*base_nc)
        self.pool2 = nn.Conv2d(2*base_nc, 2*base_nc, 2,stride=2, padding=0, bias=False)##downsampling
        self.conv3 = DoubleConv(2*base_nc, 4*base_nc)
        self.pool3 = nn.Conv2d(4*base_nc, 4*base_nc, 2,stride=2, padding=0, bias=False)##downsampling
        self.bottleneck = DoubleConv(4*base_nc, 4*base_nc)   ###changed with spatialGCN
        self.up6 = nn.ConvTranspose2d(4*base_nc, 4*base_nc, 2, stride=2)##upsampling
        self.conv6 = DoubleConv(8*base_nc, 4*base_nc)
        self.up7 = nn.ConvTranspose2d(4*base_nc, 2*base_nc, 2, stride=2,padding=0)##upsampling
        self.conv7 = DoubleConv(4*base_nc, 2*base_nc)
        self.up8 = nn.ConvTranspose2d(2*base_nc, base_nc, 2, stride=2)##upsampling
        self.conv8 = DoubleConv(2*base_nc, base_nc)
        self.conv9 = nn.Conv2d(base_nc, out_ch, kernel_size=1,padding=0)
    def forward(self,x):
        c1=self.conv1(x)  ## 2 nc
        p1=self.pool1(c1)  ##
        c2=self.conv2(p1) ##nc 2nc
        p2=self.pool2(c2)
        c3=self.conv3(p2) ##2nc 2nc
        p3=self.pool3(c3)
        c4=self.bottleneck(p3)
        up_6= self.up6(c4)

        merge6 = torch.cat([up_6, c3], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c2], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c1], dim=1)
        c8=self.conv8(merge8)
        c9= self.conv9(c8)

        return c9

class LargePlainUnet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1,base_nc=64):
        super(LargePlainUnet, self).__init__()
        self.conv1 = DoubleConv(in_ch, base_nc)
        self.pool1 = nn.Conv2d(base_nc, base_nc, 2,stride=2, padding=0, bias=False)##downsampling
        self.conv2 = DoubleConv(base_nc, 2*base_nc)
        self.pool2 = nn.Conv2d(2*base_nc, 2*base_nc, 2,stride=2, padding=0, bias=False)##downsampling
        self.conv3 = DoubleConv(2*base_nc, 4*base_nc)
        self.pool3 = nn.Conv2d(4*base_nc, 4*base_nc, 2,stride=2, padding=0, bias=False)##downsampling
        self.conv4 = DoubleConv(4 * base_nc, 8 * base_nc)
        self.pool4 = nn.Conv2d(8 * base_nc, 8 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.bottleneck = DoubleConv(8*base_nc, 8*base_nc)
        self.up5 = nn.ConvTranspose2d(8 * base_nc, 8 * base_nc, 2, stride=2)  ##upsampling
        self.conv5 = DoubleConv(16 * base_nc, 4 * base_nc)
        self.up6 = nn.ConvTranspose2d(4*base_nc, 4*base_nc, 2, stride=2)##upsampling
        self.conv6 = DoubleConv(8*base_nc, 2*base_nc)
        self.up7 = nn.ConvTranspose2d(2*base_nc, 2*base_nc, 2, stride=2,padding=0)##upsampling
        self.conv7 = DoubleConv(4*base_nc, 1*base_nc)
        self.up8 = nn.ConvTranspose2d(1*base_nc, base_nc, 2, stride=2)##upsampling
        self.conv8 = DoubleConv(2*base_nc, base_nc)
        self.conv9 = nn.Conv2d(base_nc, out_ch, kernel_size=1,padding=0)
    def forward(self,x):
        c1=self.conv1(x)  ## 2 nc
        p1=self.pool1(c1)  ##
        c2=self.conv2(p1) ##nc 2nc
        p2=self.pool2(c2)
        c3=self.conv3(p2) ##2nc 2nc
        p3=self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)


        c5=self.bottleneck(p4)
        up_5= self.up5(c5)
        merge5 = torch.cat([up_5, c4], dim=1)


        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c3], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c2], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c1], dim=1)
        c8=self.conv8(merge8)
        c9= self.conv9(c8)

        return c9



class MulUGnet(nn.Module):
    def __init__(self,in_ch=2, out_ch=2,base_nc=64,fix_grad=True):
        super(MulUGnet, self).__init__()
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


class LargeMulUGnet(nn.Module):
    def __init__(self,in_ch=2, out_ch=2,base_nc=64,fix_grad=True):
        super(LargeMulUGnet, self).__init__()
        self.fix_grad = fix_grad
        self.conv1 = DoubleConv(in_ch, base_nc)
        self.pool1 = nn.Conv2d(base_nc, base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv2 = DoubleConv(base_nc, 2 * base_nc)
        self.pool2 = nn.Conv2d(2 * base_nc, 2 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv3 = DoubleConv(2 * base_nc, 4 * base_nc)
        self.pool3 = nn.Conv2d(4 * base_nc, 4 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        self.conv4 = DoubleConv(4 * base_nc, 8 * base_nc)
        self.pool4 = nn.Conv2d(8 * base_nc, 8 * base_nc, 2, stride=2, padding=0, bias=False)  ##downsampling
        # if self.fix_grad==True:
        #     for p in self.parameters():
        #         p.requires_grad=False
        self.sgcn3 = SpatialGCN(2 * base_nc)
        self.sgcn2 = SpatialGCN(4 * base_nc)
        self.sgcn1 = SpatialGCN(8 * base_nc)  ###changed with spatialGCN
        self.sgcn0 = SpatialGCN(8 * base_nc)  ###changed with spatialGCN

        self.up5 = nn.ConvTranspose2d(8 * base_nc, 8 * base_nc, 2, stride=2, padding=0)  ##upsampling
        self.conv5 = DoubleConv(16 * base_nc, 4 * base_nc)


        self.up6 = nn.ConvTranspose2d(4 * base_nc, 4 * base_nc, 2, stride=2,padding=0)  ##upsampling
        self.conv6 = DoubleConv(8 * base_nc, 2 * base_nc)
        self.up7 = nn.ConvTranspose2d(2 * base_nc, 2 * base_nc, 2, stride=2, padding=0)  ##upsampling
        self.conv7 = DoubleConv(4 * base_nc, 1 * base_nc)
        self.up8 = nn.ConvTranspose2d(1 * base_nc, base_nc, 2, stride=2,padding=0)  ##upsampling
        self.conv8 = DoubleConv(2 * base_nc, base_nc)
        self.conv9 = nn.Conv2d(base_nc, out_ch, kernel_size=1, padding=0)
    def forward(self,x):
        c1=self.conv1(x)  ## 2 nc
        p1=self.pool1(c1)  ##
        c2=self.conv2(p1) ##nc 2nc
        p2=self.pool2(c2)
        c3=self.conv3(p2) ##2nc 2nc
        p3=self.pool3(c3)
        c4=self.conv4(p3)   ###spatial gcn 4nc
        p4 = self.pool4(c4)

        c5 = self.sgcn0(p4)
        up_5 = self.up5(c5)
        merge5 = torch.cat([up_5, self.sgcn1(c4)], dim=1)

        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, self.sgcn2(c3)], dim=1)

        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, self.sgcn3(c2)], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c1], dim=1)
        c8=self.conv8(merge8)
        c9= self.conv9(c8)

        return c9
