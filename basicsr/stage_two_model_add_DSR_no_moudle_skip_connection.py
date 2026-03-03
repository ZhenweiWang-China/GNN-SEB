import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
#from sewar import rmse
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from datasets import PatchSet, load_image_pair, transform_image
from tools import cal_patch_index, test_fill_index, ReconstructionLoss, Average, ssim_numpy
#from stage_one import stage_one
from basicsr.archs.IPG_arch import PatchUnEmbed, MGB, GAL, IPG
    #PatchMerging
from base_models.swin_transformer import PatchMerging
from basicsr.ASSA_model import BasicASTLayer

class UpsamplePixelShuffle(nn.Module):
    def __init__(self, in_channels,):
        super(UpsamplePixelShuffle, self).__init__()
        self.conv = nn.Conv2d(in_channels, 2*in_channels, kernel_size=1)  # 调整通道数到 72
        self.pixel_shuffle = nn.PixelShuffle(2)  # 上采样 2 倍

    def forward(self, x):
        x = self.conv(x)  # 调整通道数
        x = self.pixel_shuffle(x)  # PixelShuffle 上采样
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.ConvBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU()
        )
    def forward(self, inputs):
        out = self.ConvBlock(inputs)
        return out

class first_GNN(nn.Module):
    def __init__(self, height, width):
        super(first_GNN, self).__init__()

        self.first_GNN = IPG(
            # upscale=4,#不变
            in_chans=12,  # 根据输入的参数数量确定
            img_size=(height, width),  # 用于token to patch或者patch to token
            window_size=16,  # GAL需要用
            img_range=1.,

            # len(depths)表示depths的长度，如果有6个元素（depths=[1, 1, 1, 1, 1, 1]），就等于6，这里表示MGB的数量
            depths=[2, 2, ],  # 里面的数（6）表示GAL的数量
            embed_dim=180,
            num_heads=[6, 6, ],#注意力机制的数量
            mlp_ratio=4,
            upsampler='pixelshuffle',  # 可以不设置
            resi_connection='1conv',  # 可以不变，设置为二维卷积
            # graph_flags 是一个列表，通常包含多个元素，每个元素对应网络中的一个阶段。每个元素可以是 1 或 0：
            # 1 表示在该阶段使用图结构进行计算或处理。
            # 0 表示该阶段不使用图结构，而是使用其他操作（如标准的卷积操作或其他方法）。
            graph_flags=[1, 1, 1, 1, 1, 1],
            #'GN' 和 'GS' 可能代表不同类型的图神经网络操作，或者是局部与全局采样方式等。
            #'GN' 可能表示“图神经网络（Graph Network）”操作，'GS' 可能表示“图聚合（Graph Sampling）”操作。
            stage_spec=[['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS']],
            #dist_type节点之间相似性度量的方式。余弦相似度（Cosine Similarity）
            #欧氏距离（Euclidean Distance）  曼哈顿距离（Manhattan Distance）
            dist_type='cossim',  # 不变
            top_k=256,# 不变
            #head_wise = 0：如果 head_wise 为 0，这意味着多个注意力头的计算是 共享的。换句话说，多个头将共同参与计算，不会为每个头分配独立的计算资源。通常情况下，这是默认行为，意味着不同的头会共享相同的输入和计算方式。
            #head_wise = 1：如果 head_wise 为 1，则表示每个注意力头的计算是 独立的。即每个头都会有自己的计算资源和计算路径，所有头之间没有共享的计算。这可能增加计算开销，但可以让每个头有更多的自由度来处理输入数据。
            head_wise=0,
            sample_size=32,
            graph_switch=1,  # 不变  graph_switch = 1：启用图计算;graph_switch = 0：禁用图计算。
            flex_type='interdiff_plain',#参数控制了图神经网络中图操作的 灵活性，影响图计算过程中所使用的策略或操作类型。
            FFNtype='basic-dwconv3',  # 不变
            conv_scale=0,
            conv_type='dwconv3-gelu-conv1-ca',
            diff_scales=[10, 1.5, ],
            fast_graph=1
        )

        self.second_GNN = IPG(
            # upscale=4,#不变
            in_chans=24,  # 根据输入的参数数量确定
            img_size=(height//2, width//2),  # 用于token to patch或者patch to token
            window_size=16,  # GAL需要用
            img_range=1.,

            # len(depths)表示depths的长度，如果有6个元素（depths=[1, 1, 1, 1, 1, 1]），就等于6，这里表示MGB的数量
            depths=[2, 2, ],  # 里面的数（6）表示GAL的数量
            embed_dim=180,
            num_heads=[6, 6, ],
            mlp_ratio=4,
            upsampler='pixelshuffle',  # 可以不设置
            resi_connection='1conv',  # 可以不变，设置为二维卷积
            graph_flags=[1, 1, 1, 1, 1, 1],
            stage_spec=[['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS']],
            dist_type='cossim',  # 不变
            top_k=256,
            head_wise=0,
            sample_size=32,
            graph_switch=1,  # 不变
            flex_type='interdiff_plain',
            FFNtype='basic-dwconv3',  # 不变
            conv_scale=0,
            conv_type='dwconv3-gelu-conv1-ca',
            diff_scales=[10, 1.5, ],
            fast_graph=1
        )

        self.third_GNN = IPG(
            # upscale=4,#不变
            in_chans=48,  # 根据输入的参数数量确定
            img_size=(height // 4, width // 4),  # 用于token to patch或者patch to token
            window_size=16,  # GAL需要用
            img_range=1.,

            # len(depths)表示depths的长度，如果有6个元素（depths=[1, 1, 1, 1, 1, 1]），就等于6，这里表示MGB的数量
            depths=[2, 2, ],  # 里面的数（6）表示GAL的数量
            embed_dim=180,
            num_heads=[6, 6, ],
            mlp_ratio=4,
            upsampler='pixelshuffle',  # 可以不设置
            resi_connection='1conv',  # 可以不变，设置为二维卷积
            graph_flags=[1, 1, 1, 1, 1, 1],
            stage_spec=[['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS']],
            dist_type='cossim',  # 不变
            top_k=256,
            head_wise=0,
            sample_size=32,
            graph_switch=1,  # 不变
            flex_type='interdiff_plain',
            FFNtype='basic-dwconv3',  # 不变
            conv_scale=0,
            conv_type='dwconv3-gelu-conv1-ca',
            diff_scales=[10, 1.5, ],
            fast_graph=1
        )

        self.fourth_GNN = IPG(
            # upscale=4,#不变
            in_chans=3,  # 根据输入的参数数量确定
            img_size=(height // 4, width // 4),  # 用于token to patch或者patch to token
            window_size=16,  # GAL需要用
            img_range=1.,

            # len(depths)表示depths的长度，如果有6个元素（depths=[1, 1, 1, 1, 1, 1]），就等于6，这里表示MGB的数量
            depths=[2, 2, ],  # 里面的数（6）表示GAL的数量
            embed_dim=180,
            num_heads=[6, 6, ],
            mlp_ratio=4,
            upsampler='pixelshuffle',  # 可以不设置
            resi_connection='1conv',  # 可以不变，设置为二维卷积
            graph_flags=[1, 1, 1, 1, 1, 1],
            stage_spec=[['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS']],
            dist_type='cossim',  # 不变
            top_k=256,
            head_wise=0,
            sample_size=32,
            graph_switch=1,  # 不变
            flex_type='interdiff_plain',
            FFNtype='basic-dwconv3',  # 不变
            conv_scale=0,
            conv_type='dwconv3-gelu-conv1-ca',
            diff_scales=[10, 1.5, ],
            fast_graph=1
        )

        self.fifth_GNN = IPG(
            # upscale=4,#不变
            in_chans=24,  # 根据输入的参数数量确定
            img_size=(height // 2, width // 2),  # 用于token to patch或者patch to token
            window_size=16,  # GAL需要用
            img_range=1.,

            # len(depths)表示depths的长度，如果有6个元素（depths=[1, 1, 1, 1, 1, 1]），就等于6，这里表示MGB的数量
            depths=[2, 2, ],  # 里面的数（6）表示GAL的数量
            embed_dim=180,
            num_heads=[6, 6, ],
            mlp_ratio=4,
            upsampler='pixelshuffle',  # 可以不设置
            resi_connection='1conv',  # 可以不变，设置为二维卷积
            graph_flags=[1, 1, 1, 1, 1, 1],
            stage_spec=[['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS']],
            dist_type='cossim',  # 不变
            top_k=256,
            head_wise=0,
            sample_size=32,
            graph_switch=1,  # 不变
            flex_type='interdiff_plain',
            FFNtype='basic-dwconv3',  # 不变
            conv_scale=0,
            conv_type='dwconv3-gelu-conv1-ca',
            diff_scales=[10, 1.5, ],
            fast_graph=1
        )

        self.sixth_GNN = IPG(
            # upscale=4,#不变
            in_chans=12,  # 根据输入的参数数量确定
            img_size=(height, width),  # 用于token to patch或者patch to token
            window_size=16,  # GAL需要用
            img_range=1.,

            # len(depths)表示depths的长度，如果有6个元素（depths=[1, 1, 1, 1, 1, 1]），就等于6，这里表示MGB的数量
            depths=[2, 2, ],  # 里面的数（6）表示GAL的数量
            embed_dim=180,
            num_heads=[6, 6, ],
            mlp_ratio=4,
            upsampler='pixelshuffle',  # 可以不设置
            resi_connection='1conv',  # 可以不变，设置为二维卷积
            graph_flags=[1, 1, 1, 1, 1, 1],
            stage_spec=[['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS']],
            dist_type='cossim',  # 不变
            top_k=256,
            head_wise=0,
            sample_size=32,
            graph_switch=1,  # 不变
            flex_type='interdiff_plain',
            FFNtype='basic-dwconv3',  # 不变
            conv_scale=0,
            conv_type='dwconv3-gelu-conv1-ca',
            diff_scales=[10, 1.5, ],
            fast_graph=1
        )

        self.downsample1 =PatchMerging(input_resolution=(height, width), dim=12, out_dim=24)#一般out_dim=6为dim的两倍
        self.downsample2 =PatchMerging(input_resolution=(height//2, width//2), dim=24, out_dim=48)#一般out_dim=6为dim的两倍
        self.downsample3 =PatchMerging(input_resolution=(height//4, width//4), dim=48, out_dim=96)#一般out_dim=6为dim的两倍

        self.upsample3 =UpsamplePixelShuffle(96)
        self.upsample2 =UpsamplePixelShuffle(48)
        self.upsample1 = UpsamplePixelShuffle(24)

        self.con11 = Conv(12, 12)

        self.con1 = Conv(24, 24)
        #self.con2 = Conv(6, 6)
        self.con3 = Conv(12, 12)
        self.con4 = Conv(12, 12)
        self.con5 = nn.Conv2d(
                            in_channels=12,   # 输入通道数（对应输入的第二个维度 9）
                            out_channels=1,  # 输出通道数（目标为 1）
                            kernel_size=3,    # 卷积核尺寸（3x3）
                            stride=1,         # 步长
                            padding=1         # 填充，用于保持空间维度不变
                            ).to(torch.float32)
        self.BN1=nn.BatchNorm2d(12)

        self.ASSA1 = BasicASTLayer(dim=12,
                            output_dim=12,
                            input_resolution=(height, width),
                            depth=2,
                            num_heads=4,
                            win_size=8,
                            mlp_ratio=4.,
                            qkv_bias=True, qk_scale=None,
                            drop=0, attn_drop=0,
                            drop_path=0.1*2,
                            norm_layer=nn.LayerNorm,
                            use_checkpoint=False,
                            token_projection='linear',token_mlp='leff',shift_flag=True,att=True,sparseAtt=True)

        self.ASSA2 = BasicASTLayer(dim=24,
                                   output_dim=24,
                                   input_resolution=(height//2, width//2),
                                   depth=2,
                                   num_heads=2,
                                   win_size=8,
                                   mlp_ratio=4.,
                                   qkv_bias=True, qk_scale=None,
                                   drop=0, attn_drop=0,
                                   drop_path=0.1 * 2,
                                   norm_layer=nn.LayerNorm,
                                   use_checkpoint=False,
                                   token_projection='linear', token_mlp='leff', shift_flag=True, att=True,
                                   sparseAtt=True)


    def forward(self, x):

        x = self.con11(x)
        mask=torch.isinf(x)
        x[mask]=0

        #print(x)
        first_result = self.first_GNN(x)
        #print('first_result', first_result)
        #print('first_result', first_result.shape)
        #downsample1 = self.downsample1(first_result)#输出为18个channel
        downsample1 = self.downsample1(first_result)  # 输出为18个channel
        #print('downsample1', downsample1)
        second_result = self.second_GNN(downsample1)
        #print('second_result', second_result)
        downsample2 = self.downsample2(second_result)#输出为36个channel
        #print('downsample2', downsample2)
        third_result = self.third_GNN(downsample2)#输出为36个channel
        #print('third_result', third_result)
        upsample2 = self.upsample2(third_result)#输出为18个channel
        #print('upsample2', upsample2)
        con1 =self.con1(upsample2)#输出为18个channel
        #print('con1', con1)
        #upsample2 = self.upsample2(con1)
        #downsample3 = self.downsample1(third_result)

        '''fourth_result = self.fourth_GNN(con1)
        upsample2 = self.upsample2(fourth_result)
        con2 = self.con2(upsample2)'''
        #print('second_result.shape', second_result.shape)
        #z = self.ASSA2(second_result)
        #print(z.shape)
        #fifth_result = self.fifth_GNN(con1)+second_result  # 输出为18个channel
        fifth_result = self.fifth_GNN(con1)   # 输出为18个channel
        #fifth_result = self.fifth_GNN(con1)+self.ASSA2(second_result)#输出为18个channel
        #print('fifth_result', fifth_result)
        upsample1 = self.upsample1(fifth_result)#输出为9个channel
        #print('upsample1', upsample1)
        con3 = self.con3(upsample1)#输出为9个channel
        #print('con3', con3)
        sixth_result =self.sixth_GNN(con3)
        #print('sixth_result', sixth_result)
        #con4 = self.con3(sixth_result)+x
        con4 = self.con3(sixth_result)
        #print('con4', con4)
        con5 = self.con5(con4)
        #print('con5', con5)
        #on5 = con5 * 150 + 200
        return con5

class MSE_loss(nn.Module):
    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, LST_pre, LST_lable, ):


        loss = torch.mean(torch.square(LST_pre - LST_lable))
        #loss_DD = torch.mean(torch.square(Mo_Eo_recnt - Mo_Eo_lbl))
        #loss = 0.6*loss_RT + 0.2*loss_TC + 0.2*loss_DD
        return loss


'''class IPG(nn.Module):
    def __init__(self,**kwargs):
        super(IPG, self).__init__()
        first_GNN = first_GNN(self, height, width)

    def first_GNN(self, height, width):
        x = IPG(
            # upscale=4,#不变
            in_chans=3,  # 根据输入的参数数量确定
            img_size=(height, width),  # 用于token to patch或者patch to token
            window_size=16,  # GAL需要用
            img_range=1.,

            # len(depths)表示depths的长度，如果有6个元素（depths=[1, 1, 1, 1, 1, 1]），就等于6，这里表示MGB的数量
            depths=[6, 6, ],  # 里面的数（6）表示GAL的数量
            embed_dim=180,
            num_heads=[6, 6, ],
            mlp_ratio=4,
            upsampler='pixelshuffle',  # 可以不设置
            resi_connection='1conv',  # 可以不变，设置为二维卷积
            graph_flags=[1, 1, 1, 1, 1, 1],
            stage_spec=[['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS']],
            dist_type='cossim',  # 不变
            top_k=256,
            head_wise=0,
            sample_size=32,
            graph_switch=1,  # 不变
            flex_type='interdiff_plain',
            FFNtype='basic-dwconv3',  # 不变
            conv_scale=0,
            conv_type='dwconv3-gelu-conv1-ca',
            diff_scales=[10, 1.5, ],
            fast_graph=1
        )

        return x
    def second_GNN(self, height, width):
        x = IPG(
            # upscale=4,#不变
            in_chans=3,  # 根据输入的参数数量确定
            img_size=(height//2, width//2),  # 用于token to patch或者patch to token
            window_size=16,  # GAL需要用
            img_range=1.,

            # len(depths)表示depths的长度，如果有6个元素（depths=[1, 1, 1, 1, 1, 1]），就等于6，这里表示MGB的数量
            depths=[6, 6, ],  # 里面的数（6）表示GAL的数量
            embed_dim=180,
            num_heads=[6, 6, ],
            mlp_ratio=4,
            upsampler='pixelshuffle',  # 可以不设置
            resi_connection='1conv',  # 可以不变，设置为二维卷积
            graph_flags=[1, 1, 1, 1, 1, 1],
            stage_spec=[['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS']],
            dist_type='cossim',  # 不变
            top_k=256,
            head_wise=0,
            sample_size=32,
            graph_switch=1,  # 不变
            flex_type='interdiff_plain',
            FFNtype='basic-dwconv3',  # 不变
            conv_scale=0,
            conv_type='dwconv3-gelu-conv1-ca',
            diff_scales=[10, 1.5, ],
            fast_graph=1
        )

        return x
    def third_GNN(self, height, width):
        x = IPG(
            # upscale=4,#不变
            in_chans=3,  # 根据输入的参数数量确定
            img_size=(height//4, width//4),  # 用于token to patch或者patch to token
            window_size=16,  # GAL需要用
            img_range=1.,

            # len(depths)表示depths的长度，如果有6个元素（depths=[1, 1, 1, 1, 1, 1]），就等于6，这里表示MGB的数量
            depths=[6, 6, ],  # 里面的数（6）表示GAL的数量
            embed_dim=180,
            num_heads=[6, 6, ],
            mlp_ratio=4,
            upsampler='pixelshuffle',  # 可以不设置
            resi_connection='1conv',  # 可以不变，设置为二维卷积
            graph_flags=[1, 1, 1, 1, 1, 1],
            stage_spec=[['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS']],
            dist_type='cossim',  # 不变
            top_k=256,
            head_wise=0,
            sample_size=32,
            graph_switch=1,  # 不变
            flex_type='interdiff_plain',
            FFNtype='basic-dwconv3',  # 不变
            conv_scale=0,
            conv_type='dwconv3-gelu-conv1-ca',
            diff_scales=[10, 1.5, ],
            fast_graph=1
        )

        return x'''



'''def first_GNN(height, width):
    x = IPG(
            # upscale=4,#不变
            in_chans=3,  # 根据输入的参数数量确定
            img_size=(height, width),  # 用于token to patch或者patch to token
            window_size=16,  # GAL需要用
            img_range=1.,

            # len(depths)表示depths的长度，如果有6个元素（depths=[1, 1, 1, 1, 1, 1]），就等于6，这里表示MGB的数量
            depths=[6, 6, ],  # 里面的数（6）表示GAL的数量
            embed_dim=180,
            num_heads=[6, 6, ],
            mlp_ratio=4,
            upsampler='pixelshuffle',  # 可以不设置
            resi_connection='1conv',  # 可以不变，设置为二维卷积
            graph_flags=[1, 1, 1, 1, 1, 1],
            stage_spec=[['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'],
                        ['GN', 'GS', 'GN', 'GS', 'GN', 'GS'], ['GN', 'GS', 'GN', 'GS', 'GN', 'GS']],
            dist_type='cossim',  # 不变
            top_k=256,
            head_wise=0,
            sample_size=32,
            graph_switch=1,  # 不变
            flex_type='interdiff_plain',
            FFNtype='basic-dwconv3',  # 不变
            conv_scale=0,
            conv_type='dwconv3-gelu-conv1-ca',
            diff_scales=[10, 1.5, ],
            fast_graph=1
    )



    return x'''

if __name__ == '__main__':
    #upscale = 4
    #设置输入patch的长、宽度
    #set KMP_DUPLICATE_LIB_OK = TRUE  # Windows命令行
    height = 32
    width = 32
    model = first_GNN(height, width)
    model.cuda()
    model.train()
    #print(model)
    #print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 12, 16, 16)).cuda().to(torch.float32)
    #print('x', x)
    y = model(x)

    print('y', y)
