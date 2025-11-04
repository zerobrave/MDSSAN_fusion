import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
import cv2
import math
import itertools


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class ResSpectralExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResSpectralExtractor, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
                                  ResidualBlock(out_channels),
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU(negative_slope=0.0)
                                  )
        

    def forward(self, hsi_feats):
        out =self.conv(hsi_feats)
        return out

class ResSpatialExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResSpatialExtractor, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
                                  ResidualBlock(out_channels),
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU(negative_slope=0.0),
                                  )
        
    def forward(self, rgb_feats):
        out =self.conv(rgb_feats)
        return out
    

class ECABlock(nn.Module):
    """ECA module"""
    def __init__(self, channels, b=1, gamma=2):
        super(ECABlock,self).__init__()
        #自适应卷积核大小, 可设置为7
        self.kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        if self.kernel_size % 2 ==0 :
            self.kernel_size = self.kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=(self.kernel_size - 1)//2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
    
        b, c, h, w = x.size()
        z = self.avg_pool(x)

        # Two different branches of ECA module
        z = self.conv(z.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        z = self.sigmoid(z)
        return y*z.expand_as(y)  #维度扩展
    
class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                       nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x, y):
        """
        x: 输入特征以得到权重
        y: 用于注意力的特征
        """
        attention_mask = self.attention(x)
        features = self.conv(y)
        return torch.mul(features, attention_mask)

class AttentionFusedBlock(nn.Module):
    """ SA and CA for fusion of aggregated features """
    """ 空间特征和光谱特征在融合后做自注意力"""
    def __init__(self, in_channels, out_channels):
        super(AttentionFusedBlock, self).__init__()
        #输入输出通道为融合特征的通道
        self.in_channels = in_channels   #2c
        self.out_channels = out_channels  #c

        self.concat = nn.Conv2d(2*in_channels, in_channels)

        self.CA = ECABlock(self.in_channels)
        self.SA = SABlock(self.in_channels, self.in_channels)
        self.aggregate1 = nn.Conv2d(in_channels=2*self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)
        self.aggregate2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, fused_feats):
        #fused_feats = self.FE(fused_feats)        #
        fused_feats = self.concat(fused_feats)
        sa = self.SA(fused_feats, fused_feats)
        ca = self.CA(fused_feats, fused_feats)
        feats = torch.cat((sa, ca), dim=1)  #4c
        feats = self.aggregate1(feats)      #2c
        out = feats + fused_feats           #skip  2c
        out = self.aggregate2(out)          #c
        return out    

class RefineAttentionBlock(nn.Module):
    """ RefineAttention. fuse to apply CA&SA """
    """ 利用融合后特征对输入的空间和光谱特征进行细化"""
    def __init__(self, in_channels, out_channels):
        super(RefineAttentionBlock, self).__init__()
        self.in_channels = in_channels  #c
        self.out_channels = out_channels  #c
        self.FE = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, padding=0)

        self.CA = ECABlock(self.in_channels)
        self.SA = SABlock(self.in_channels, self.in_channels)
        self.aggregate = nn.Conv2d(3*self.in_channels, self.in_channels, kernel_size=1, padding=0)

    def forward(self, lr_feats, pan_feats, fused_feats):
        """
        lr_feats: 相同尺度下低分辨率高光谱特征   c
        pan_feats: 相同尺度下pan特征  c
        fused_feats: 该尺度下concat的特征, 通道数为2c
        """
        #transform
        fused_feats = self.FE(fused_feats)  #c
        #lr_feats = self.lr(lr_feats)
        #pan_feats = self.pan(pan_feats)
        #refine attention
        sa = self.SA(fused_feats, pan_feats)
        ca = self.CA(lr_feats, fused_feats)
        #feats = sa+ca+fused_feats          #c
        feats = torch.cat((sa, ca, fused_feats), dim=1)
        feats = self.aggregate(feats)      #2c
        return feats


class FastSpectralAttention(nn.Module):
    ''' Spectral transformer '''
    #通过亚像素卷积来完成特征图的采样，将邻域像素的信息转为通道信息，完成相关性矩阵的计算后进一步进行逆亚像素卷积变回原始尺寸
    def __init__(self, n_feats, rate):
        super().__init__()
        self.num_features = n_feats
        self.rate = rate
        self.md_features = int(n_feats/self.rate**2)
        self.query = nn.Sequential(nn.Conv2d(in_channels=self.num_features, out_channels=self.md_features, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(self.md_features), nn.PixelUnshuffle(self.rate), nn.LeakyReLU(negative_slope=0.0),
                                   )
        self.key = nn.Sequential(nn.Conv2d(in_channels=self.num_features, out_channels=self.md_features, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(self.md_features), nn.PixelUnshuffle(self.rate),nn.LeakyReLU(negative_slope=0.0),
                               
                                 )
        self.value = nn.Sequential(nn.Conv2d(in_channels=self.num_features, out_channels=self.md_features, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(self.md_features),nn.PixelUnshuffle(self.rate), nn.LeakyReLU(negative_slope=0.0),
                                  
                                   )
        #self.q_desub = nn.PixelUnshuffle(2) 
        #self.k_desub = nn.PixelUnshuffle(2) 
        #self.v_desub = nn.PixelUnshuffle(2) 
        
        self.tail = nn.Sequential(nn.Conv2d(in_channels=self.md_features, out_channels=self.num_features, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(self.num_features),
                                  nn.LeakyReLU(negative_slope=0.0),)
        
        self.aggregate = nn.PixelShuffle(self.rate)  #

    def forward(self, refine_hsi,lr_hsi, mask=None):
        b, c, H, W = refine_hsi.size(0), refine_hsi.size(1), refine_hsi.size(2), refine_hsi.size(3)
        h, w =int(H/self.rate), int(W/self.rate)
        q = self.query(refine_hsi).view(b, self.num_features, -1).permute(0, 2, 1)  # b c HW  ----b HW c
        k = self.key(lr_hsi).view(b, self.num_features, -1)  # b c HW/4
        v = self.value(lr_hsi).view(b, self.num_features, -1).permute(0, 2, 1)  # b HW/4 c

        correlation = torch.matmul(q, k)  # b HW HW/4
        correlation = ((self.num_features)** -0.5) * correlation
        if mask is not None:
            correlation = F.softmax(correlation/0.01, dim=1)*mask
        else:
            correlation = F.softmax(correlation, dim=1)

        spatial_spectral = torch.matmul(correlation, v)  # b HW c
        spatial_spectral = spatial_spectral.permute(0, 2, 1).contiguous()  # b c HW
        spatial_spectral = spatial_spectral.view(b, self.num_features, h, w)
        spatial_spectral = self.aggregate(spatial_spectral) 
        
        spatial_spectral = torch.cat((spatial_spectral, ))
        spatial_spectral = self.tail(spatial_spectral)  # b c H W
    
        return spatial_spectral
    



class MDSSAN(nn.Module):
    def __init__(self, config):
        super(MDSSAN, self).__init__()
        #考虑边缘光谱的align 
        #抽取hsi三波段来和rgb做可见匹配,
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]
        self.lr_size =  config[config["train_dataset"]]["LR_size"]
        self.hr_size =  config[config["train_dataset"]]["HR_size"]
        self.n_select_bands = config[config['train_dataset']]["msi_bands"]
        self.selected_sp_channels = [config[config['train_dataset']]['B'], config[config['train_dataset']]['G'], config[config['train_dataset']]['R']]

        self.cs = 32
        self.num_layers = config['num_layers'] + 1
        #self.num_layers = 3 + 1 #选择3层
        self.outchannels = list(itertools.repeat(self.cs, self.num_layers))
        self.num_blocks = len(self.outchannels)-1
        self.num_res_blocks = list(itertools.repeat(0, self.num_layers))
        self.res_scale = 1
        
        self.pixel_error = config['pixel_error']
        #self.mask = generate_mask(self.lr_size, k=self.pixel_error)
        self.mask = None
        self.msi = nn.Conv2d(self.n_select_bands, self.outchannels[0], kernel_size=3, padding=1, stride=1)
        self.msi1 = nn.Conv2d(3, self.outchannels[0], kernel_size=3, padding=1, stride=1)
        self.hsi = nn.Conv2d(self.in_channels, self.outchannels[0], kernel_size=3, padding=1, stride=1)
        
        self.up_align = config['up_align']
        self.sp_align = config['sp_align']
        
        self.hsi_first_fuse = nn.Conv2d(self.in_channels+self.cs, self.in_channels, 1)
        self.sampled_aggregation = nn.ModuleList()
        self.hsi_extractor = nn.ModuleList()
        self.rgb_extractor = nn.ModuleList()
        self.fused = nn.ModuleList()
        self.refined = nn.ModuleList()

        for i in range(self.num_blocks):
            self.sampled_aggregation.append(FastSpectralAttention(self.cs, 4))
            self.rgb_extractor.append(ResSpatialExtractor(self.outchannels[i], self.outchannels[i+1]))
            self.hsi_extractor.append(ResSpectralExtractor(self.outchannels[i], self.outchannels[i + 1]))
            self.fused.append(AttentionFusedBlock(self.outchannels[i], self.outchannels[i + 1]))
            self.refined.append(RefineAttentionBlock(self.outchannels[i], self.outchannels[i + 1]))

        
        self.conv_tail = nn.Conv2d(in_channels=self.outchannels[-1], out_channels=self.out_channels, kernel_size=3, padding=1)
        self.out = nn.Conv2d(in_channels=self.outchannels[-1], out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, hsi, rgb):
        hsi_up = F.interpolate(hsi, scale_factor=self.factor, mode='bilinear') 
        rgb_down = F.interpolate(rgb, scale_factor=1/self.factor, mode='bilinear') 
        
        b_hsi = hsi_up[:, self.selected_sp_channels[0], :, :].unsqueeze(1)
        g_hsi = hsi_up[:, self.selected_sp_channels[1], :, :].unsqueeze(1)
        r_hsi = hsi_up[:, self.selected_sp_channels[2], :, :].unsqueeze(1)
        rgb_hsi = torch.cat((r_hsi, g_hsi, b_hsi), dim=1) #抽取hsi 三波段用于可见匹配
        
        edge = self.edge_conv2d(rgb_hsi)
        edge = edge[:, 0, :, :].unsqueeze(1)  #边缘提取
        mask = self.mask #生成匹配搜索窗口， k为配准误差
        
        hsi_orig = hsi
        rgb = self.msi(rgb) #通道对齐
        hsi = self.hsi(hsi)  
        hsi_up_f = self.hsi(hsi_up)
        hsi_f = hsi_up
        base_up = hsi_f
           
        ref = []
        for i in range(self.num_blocks):
            rgb = self.rgb_extractor[i](rgb)  #RFEM,残差特征提取
            hsi = self.hsi_extractor[i](hsi)
            hsi_up_f = self.hsi_extractor[i](hsi_up_f) #hsi上采样以对齐spat_aggre

            align_rgb = self.sampled_aggregation[i](rgb, hsi)  #sampling and aggregation得到spat_aggre
            F_concat = torch.cat((hsi_up_f, align_rgb), dim=1)
            fused = self.fused[i](F_concat)
            refine = self.refined[i](hsi, rgb, fused)

            ref.append(refine)
            
        refine = ref[0]
        for i in range(len(ref)-1):
            refine = refine + ref[i+1]
        #refine = ref[0]+ref[1]+ref[2]+ref[3]
        out = self.conv_tail(refine)  + base_up
        out = self.out(out)
        return {'pred': out.contiguous(), "edge": edge}
    
    def edge_conv2d(self, im):
        im = im[:, :3,:, :]
        conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
        sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)
        conv_op.weight.data = torch.from_numpy(sobel_kernel)
        # print(conv_op.weight.size())
        # print(conv_op, '\n')
        conv_op = conv_op.cuda()
        edge_detect = conv_op(im)
        return edge_detect