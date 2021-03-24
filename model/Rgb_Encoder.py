import torch
import torch.nn as nn
from model.vgg import B2_VGG

########################################################################################
############################feature extraction block####################################
class CRU(nn.Module):
    def __init__(self, in_channel, out_channel,N):
        super(CRU, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = MGCN_rgb(in_channel,in_channel//4,out_channel,N)

        self.conv_cat = nn.Conv2d(5*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x_cat = torch.cat((x0, x1, x2,x3,x4), 1)
        x_cat = self.conv_cat(x_cat)
        x = self.relu(x_cat + self.conv_res(x))
        return x
########################################################################################

###############################global-reason######################################
class GCN(nn.Module):
    def __init__(self, dim_1_channels, dim_2_channels):
        super().__init__()

        self.conv1d_1 = nn.Conv1d(dim_1_channels, dim_1_channels, 1)
        self.conv1d_2 = nn.Conv1d(dim_2_channels, dim_2_channels, 1)

    def forward(self, x):
        h = self.conv1d_1(x)
        h = (x - h).permute(0, 2, 1)       ######自己加的
        return self.conv1d_2(h).permute(0, 2, 1)


class MGCN_sub(nn.Module):
    def __init__(self, in_channels, mid_channels, N):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.N = N

        self.phi = nn.Conv2d(in_channels, mid_channels, 1)
        self.theta = nn.Conv2d(in_channels, N, 1)
        self.gcn = GCN(N, mid_channels)
        self.phi_inv = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, x):
        batch_size, in_channels, h, w = x.shape
        mid_channels = self.mid_channels
        N = self.N

        B = self.theta(x).view(batch_size, N, -1)
        x_reduced = self.phi(x).view(batch_size, mid_channels, h * w)
        x_reduced = x_reduced.permute(0, 2, 1)
        v = B.bmm(x_reduced)
        z = self.gcn(v)
        y = B.permute(0, 2, 1).bmm(z).permute(0, 2, 1)
        y = y.view(batch_size, mid_channels, h, w)
        x = self.phi_inv(y)
        return  x


class MGCN_rgb(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, N):
        super().__init__()
        self.glore1 = MGCN_sub(in_channels, mid_channels, N)
        self.glore2 = MGCN_sub(in_channels, mid_channels, N// 2 )
        self.glore3 = MGCN_sub(in_channels, mid_channels, N //4)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        x1 = self.glore1(x)
        x2 = self.glore2(x)
        x3 = self.glore3(x)
        x_f = x+x1+x2+x3
        x = self.conv(x_f)
        return x

########################################################################################

###################################depth decoder block##################################

class VGG_Rgb(nn.Module):
    def __init__(self, channel=32):
        super( VGG_Rgb, self).__init__()
        self.vgg = B2_VGG()
        #depth stream
        self.rfb3_1 = CRU(256, channel,32*32)
        self.rfb4_1 = CRU(512, channel,16*16)
        self.rfb5_1 = CRU(512, channel,8*8)
    def forward(self, x):
        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)
        #depth stream
        x3_1 = x3
        x4_1 = self.vgg.conv4_1(x3_1)
        x5_1 = self.vgg.conv5_1(x4_1)
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        x5_1 = self.rfb5_1(x5_1)
        return x3_1, x4_1, x5_1
