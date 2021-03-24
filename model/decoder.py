import torch
import torch.nn as nn
from model.Focal_Encoder import VGG_Focal
from model.Rgb_Encoder import VGG_Rgb
from model.vgg import B2_VGG

########################################################################################
class SSA_Fusion(nn.Module):
    def __init__(self, inchannel,at_type='relation-attention'):
        super(SSA_Fusion, self).__init__()
        self.at_type = at_type
        self.inchannel = inchannel
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.6)

        self.alpha = nn.Sequential(nn.Linear(self.inchannel, 1),  ###512*1
                                   nn.Sigmoid())

        self.beta = nn.Sequential(nn.Linear(self.inchannel*2, 1),
                                  nn.Sigmoid())
        self.conv = nn.Sequential(
            nn.Conv2d(2 * inchannel, inchannel,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.parameter_initialization()

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x, AT_level='second_level',vectors='',vm=''):
        N,D,H,W = x.size()
        x = x.view(1,N,D,H,W)
        vs = []
        alphas = []
        assert AT_level == 'first_level' or AT_level == 'second_level'
        num_pair = x.size(1)
        for i in range(num_pair):
            f = x[:,i, :, :, :]  # x[128,3,224,224]   这里有疑问
            f1 = self.avgpool(f)
            f1 = f1.squeeze(3).squeeze(2)  # f[1, 512, 1, 1] ---> f[1, 512]    #####squeeze()用于删除指定的维度
            vs.append(f)
            alphas.append(self.alpha(self.dropout(f1)).view(1,1,1,1))       ######(1*512)*(512*1)=1*1

        vs_stack = torch.stack(vs, dim=1)
        alphas_stack = torch.stack(alphas, dim=1)

        if self.at_type == 'self-attention':
            vm1 = ((vs_stack.mul(alphas_stack)).sum(1)).div(alphas_stack.sum(1))
        if self.at_type == 'relation-attention':
            vm1 = vs_stack.mul(alphas_stack).sum(1).div(alphas_stack.sum(1))
            betas = []
            for i in range(len(vs)):
                vs[i] = torch.cat([vs[i], vm1], dim=1)
                vs1= self.avgpool(vs[i])
                vs1 = vs1.squeeze(3).squeeze(2)
                betas.append(self.beta(self.dropout(vs1)).view(1,1,1,1))
            cascadeVs_stack = torch.stack(vs, dim=1)
            betas_stack = torch.stack(betas, dim=1)
            output = cascadeVs_stack.mul(betas_stack * alphas_stack).sum(1).div((betas_stack * alphas_stack).sum(1))
            output = self.conv(output)
        return output

############################feature cross-pass-fusion  block####################################
class Cross_Fusion(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(Cross_Fusion, self).__init__()
        self.relu = nn.ReLU(True)
        self.focal_conv1 = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, 3, (2, 1, 1), padding=(1,1,1)),
            nn.Conv3d(in_channel, out_channel, 3, (2, 1, 1), padding=(1, 1, 1)),
            nn.Conv3d(out_channel, out_channel, (2,1,1), (2,1,1) ,padding=0),
        )

        self.rgb_conv1 =  nn.Conv2d(in_channel, out_channel, 1)

        self.focal_conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3,1,padding=1),
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.rgb_conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, padding=1),
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.cat_conv = SSA_Fusion(out_channel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
    def forward(self, x_rgb,x_focal):
        batch_size,channel,H,W = x_focal.shape
        x_focal_view = x_focal.view(1, batch_size,channel,H,W)
        x_focal_T = x_focal_view.permute(0,2,1,3,4)

        x_focal_conv1 = self.focal_conv1(x_focal_T)
        x_focal_rgb = x_focal_conv1.view(1,channel,H,W)
        x_rgb_conv1 = self.rgb_conv1(x_rgb)
        x_rgb_focal = torch.cat((x_rgb_conv1,x_rgb_conv1,x_rgb_conv1,x_rgb_conv1,x_rgb_conv1,x_rgb_conv1,x_rgb_conv1,
                                 x_rgb_conv1,x_rgb_conv1,x_rgb_conv1,x_rgb_conv1,x_rgb_conv1),dim=0)

        focal_add_out = self.focal_conv2(x_focal + x_rgb_focal)
        rgb_add_out = self.rgb_conv2( x_rgb + x_focal_rgb)
        out = torch.cat((rgb_add_out,focal_add_out),dim=0)
        out = self.cat_conv(out)
        return out
########################################################################################
###################################depth decoder block##################################
######################## aggragation three level depth features#########################
class decoder_d(nn.Module):
    def __init__(self, channel):
        super(decoder_d, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = nn.Conv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat4 = nn.Conv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat5 = nn.Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5_1 = nn.Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5_2 = nn.Conv2d(3 * channel,1, 1, 1)

        self.cross_fusion1 =  Cross_Fusion(channel,channel)
        self.cross_fusion2 = Cross_Fusion(channel,channel)
        self.cross_fusion3 = Cross_Fusion(channel,channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, rgb3, rgb4, rgb5,focal3,focal4,focal5):
        # x3: 1/16 x4: 1/8 x5: 1/4
        rgb3_1 = rgb3
        rgb4_1 = rgb4
        rgb5_1 = rgb5

        focal3_1 = focal3
        focal4_1 = focal4
        focal5_1 = focal5

        fusion1_out = self.cross_fusion1(rgb5_1,focal5_1)
        fusion1_out_up_conv = self.conv_upsample4(self.upsample(fusion1_out))

        fusion2_out = self.cross_fusion2(rgb4_1,focal4_1)
        cat1 = torch.cat((fusion1_out_up_conv,fusion2_out),dim=1)
        cat1_conv = self.conv_concat4(cat1)

        cat1_conv_up_conv = self.conv_upsample5(self.upsample( cat1_conv))
        fusion3_out = self.cross_fusion3(rgb3_1,focal3_1)
        cat2 = torch.cat((cat1_conv_up_conv,fusion3_out),dim=1)
        cat2_conv = self.conv_concat5(cat2)

        x = self.conv5_1(cat2_conv)
        out = self.conv5_2(x)
        return out

class Depth_Net(nn.Module):
    def __init__(self, channel=32):
        super(Depth_Net,self).__init__()
        self.vgg = B2_VGG()
        self.focal_net = VGG_Focal(channel)
        self.rgb_net = VGG_Rgb(channel)
        self.decoder_net = decoder_d(channel)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, x,focal):
        rgb3, rgb4, rgb5 = self.rgb_net(x)
        focal3, focal4, focal5 = self.focal_net(focal)
        decoder = self.decoder_net( rgb3, rgb4, rgb5,focal3, focal4, focal5)
        depth_out = self.upsample1(decoder)
        return  depth_out
