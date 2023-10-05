import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import pdb
import math


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class TransferLayer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=[1,3,5], stride=1, padding=[0,1,2], dilation=1):
        super(TransferLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size=kernel_size[0], stride=stride,
                               padding=padding[0], dilation=dilation, bias=False)

        self.conv3 = nn.Conv2d(in_planes, out_planes,
                               kernel_size=kernel_size[1], stride=stride,
                               padding=padding[1], dilation=dilation, bias=False)

        self.conv5 = nn.Conv2d(in_planes, out_planes,
                               kernel_size=kernel_size[2], stride=stride,
                               padding=padding[2], dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.bn5 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x5 = self.conv5(x)
        x5 = self.bn5(x5)
        x = x1 + x3 + x5
        return x

class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class SAM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        #self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class DWConv_Mulit(nn.Module):
    def __init__(self, dim=768):
        super(DWConv_Mulit, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv_Mulit(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        #self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        #self.apply(self._init_weights)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, q, k


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Bottleneck, self).__init__()
        self.map = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False)
        self.conv0 = nn.Conv2d(in_planes, out_planes // 4, kernel_size=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(out_planes // 4, out_planes // 4, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes // 4, out_planes, kernel_size=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(out_planes // 4)
        self.bn1 = nn.BatchNorm2d(out_planes // 4)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn_map = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x):
        x_ = self.bn_map(self.map(x))
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(x_ + self.bn2(self.conv2(x)))
        return x

class Linear_Eca_block(nn.Module):
    """docstring for Eca_block"""
    def __init__(self):
        super(Linear_Eca_block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=5, padding=int(5/2), bias=False)
        self.sigmoid = nn.Sigmoid()
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv1d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()
    def forward(self, x, gamma=2, b=1):
        #N, C, H, W = x.size()
        y = self.avgpool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y.expand_as(x)

class HybridAttention(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(HybridAttention, self).__init__()

        self.eca = Linear_Eca_block()
        self.conv = BasicConv2d(in_planes // 2, out_planes // 2, 3, 1, 1)
        #self.sp = SpatialAttention()
        self.down_c = BasicConv2d(out_planes//2, 1, 3, 1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c = x.shape[1]
        x_t, x_c = torch.split(x, c // 2, dim=1)
        sa = self.sigmoid(self.down_c(x_c))
        gc = self.eca(x_t)
        x_c = self.conv(x_c)
        x_c = x_c * gc
        x_t = x_t * sa
        x = torch.cat((x_t, x_c), 1)
        return x

class HybridSenmentic(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(HybridSenmentic, self).__init__()

        self.patch_embed = OverlapPatchEmbed(img_size=224 // 4, patch_size=3, stride=1, in_chans=in_planes,
                                             embed_dim=out_planes)
        self.block = Block(dim=out_planes)
        self.norm = nn.LayerNorm(out_planes)
        self.gc = Linear_Eca_block()
        self.conv = Bottleneck(in_planes, out_planes)
        self.down_c = BasicConv2d(out_planes, 1, 3, 1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x):
        B = x.shape[0]
        #c = x.shape[1]
        #x_t, x_c = torch.split(x, c // 2, dim=1)
        x_t, H, W = self.patch_embed(x)
        x_t, q, k = self.block(x_t, H, W)
        x_t = self.norm(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        q = q.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        k = q.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        atten = q * k
        atten_c = self.gc(atten)
        atten_s = self.sigmoid(self.down_c(atten))
        x_t = x_t * atten_c * atten_s
        #x_t = self.upsample(x_t)
        x_c = self.conv(x)
        #x_c = self.upsample(x_c)
        x = x_t * x_c
        x = self.upsample(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        #self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x, H, W):
        msa, q, k = self.attn(self.norm1(x), H, W)
        x = x + msa
        x = x + self.mlp(self.norm2(x), H, W)

        return x, q, k


class Decoder(nn.Module):
    def __init__(self, img_size=224,  in_chans=3,  embed_dims=[512, 320, 128, 64],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[0, 0, 0, 0]):
        super(Decoder, self).__init__()
        # self.patch_embed1 = OverlapPatchEmb        self.block1 = Block(dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                     drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, sr_ratio=sr_ratios[0])ed(img_size=224 // 4, patch_size=3, stride=1, in_chans=embed_dims[0],
        #                                       embed_dim=320)
        # self.patch_embed2 = OverlapPatchEmbed(img_size=224 // 8, patch_size=3, stride=1, in_chans=embed_dims[1],
        #                                       embed_dim=128)
        # self.patch_embed3 = OverlapPatchEmbed(img_size=224 // 16, patch_size=3, stride=1, in_chans=embed_dims[2],
        #                                       embed_dim=64)
        # self.block1 = Block(dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                     drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, sr_ratio=sr_ratios[0])
        # self.norm1 = norm_layer(embed_dims[1])
        #
        # self.block2 = Block(dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                     drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, sr_ratio=sr_ratios[0])
        # self.norm2 = norm_layer(embed_dims[2])
        #
        # self.block3 = Block(dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                     drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, sr_ratio=sr_ratios[0])
        # self.norm3 = norm_layer(embed_dims[3])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.transferlayer = BasicConv2d(512, 512, 1, padding=0)
        # self.conv1 = Bottleneck(512, 320)
        # self.conv2 = Bottleneck(320, 128)
        # self.conv3 = Bottleneck(128, 64)
        # self.pooling = nn.AdaptiveAvgPool2d(1)
        # self.down_c1 = BasicConv2d(320, 1, 1, padding=0)
        # self.down_c2 = BasicConv2d(128, 1, 1, padding=0)
        # self.down_c3 = BasicConv2d(64, 1, 1, padding=0)
        self.hs0 = HybridSenmentic(512, 320)
        self.hs1 = HybridSenmentic(320, 128)
        self.hs2 = HybridSenmentic(128, 64)
        self.hb0 = HybridAttention(512, 512)
        self.hb1 = HybridAttention(320, 320)
        self.hb2 = HybridAttention(128, 128)
        self.hb3 = HybridAttention(64, 64)
        # self.gc1 = Linear_Eca_block()
        # self.gc2 = Linear_Eca_block()
        # self.gc3 = Linear_Eca_block()
        #
        # self.softmax = nn.Sigmoid()

    def forward(self, pvt):
        x1 = pvt[0] #64*64*64 img_size // 4
        x2 = pvt[1] #32*32*128 img_size // 8
        x3 = pvt[2] #16*16*320 img_size // 16
        x4 = pvt[3] #8*8*512   img_size // 32
        #x4 = self.hb0(x4)
        x_4 = self.transferlayer(x4)
        #x = self.upsample(x_4)

        x = self.hs0(x_4)
        x = x * self.hb1(x3)
        #x = self.upsample(x)

        x_3 = x


        x = self.hs1(x)
        x = x * self.hb2(x2)
        #x = self.upsample(x)
        x_2 = x


        x = self.hs2(x)
        x_1 = x * self.hb3(x1)
        #x_1 = self.upsample(x)


        return x_1, x_2, x_3, x_4




class HSNet(nn.Module):
    def __init__(self, channel=32):
        super(HSNet, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.decoder = Decoder()
        #self.decoder2 = Decoder()

        # self.Translayer2_0 = BasicConv2d(64, channel, 1)
        # self.Translayer2_1 = BasicConv2d(128, channel, 1)
        # self.Translayer3_1 = BasicConv2d(320, channel, 1)
        # self.Translayer4_1 = BasicConv2d(512, channel, 1)
        #
        # self.CFM = CFM(channel)
        # self.ca = ChannelAttention(64)
        # self.sa = SpatialAttention()
        # self.SAM = SAM()
        #
        # self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        # self.out_SAM = nn.Conv2d(channel, 1, 1)
        # self.out_CFM = nn.Conv2d(channel, 1, 1)
        self.out1 = nn.Conv2d(64, 1, 1)
        self.out2 = nn.Conv2d(128, 1, 1)
        self.out3 = nn.Conv2d(320, 1, 1)
        self.out4 = nn.Conv2d(512, 1, 1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dc1 = nn.Conv2d(64, 64, 1)
        self.dc2 = nn.Conv2d(128, 64, 1)
        self.dc3 = nn.Conv2d(320, 64, 1)
        self.dc4 = nn.Conv2d(512, 64, 1)
        self.bn_dc1 = nn.BatchNorm2d(64)
        self.bn_dc2 = nn.BatchNorm2d(64)
        self.bn_dc3 = nn.BatchNorm2d(64)
        self.bn_dc4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        # backbone
        #with torch.no_grad():
        pvt = self.backbone(x)
            #pvt1 = self.decoder1(pvt)
        #pvt[0]=self.drop(pvt[0])
        #pvt[1]=self.drop(pvt[1])
        #pvt[2]=self.drop(pvt[2])
        #pvt[3]=self.drop(pvt[3])
        x1, x2, x3, x4 = self.decoder(pvt)    
        #x1=x1.detach()
        #x2=x2.detach()
        #x3=x3.detach()
        #x4=x4.detach()
        B = x1.shape[0]
        y1 = self.pooling(self.bn_dc1(self.dc1(x1)))#.detach()
        y2 = self.pooling(self.bn_dc2(self.dc2(x2)))#.detach()
        y3 = self.pooling(self.bn_dc3(self.dc3(x3)))#.detach()
        y4 = self.pooling(self.bn_dc4(self.dc4(x4)))#.detach()
        y = y1 + y2 + y3 + y4
        
        coeff = self.sigmoid(self.fc2(self.relu(self.fc1(y.reshape(B, -1)))))
        prediction1 = self.out1(x1) * coeff[:,0].reshape(B, 1, 1, 1)
        prediction2 = self.out2(x2) * coeff[:,1].reshape(B, 1, 1, 1)
        prediction3 = self.out3(x3) * coeff[:,2].reshape(B, 1, 1, 1)
        prediction4 = self.out4(x4) * coeff[:,3].reshape(B, 1, 1, 1)
        # x1 = pvt[0]
        # x2 = pvt[1]
        # x3 = pvt[2]
        # x4 = pvt[3]
        #
        # # CIM
        # x1 = self.ca(x1) * x1  # channel attention
        # cim_feature = self.sa(x1) * x1  # spatial attention
        #
        # # CFM
        # x2_t = self.Translayer2_1(x2)
        # x3_t = self.Translayer3_1(x3)
        # x4_t = self.Translayer4_1(x4)
        # cfm_feature = self.CFM(x4_t, x3_t, x2_t)
        #
        # # SAM
        # T2 = self.Translayer2_0(cim_feature)
        # T2 = self.down05(T2)
        # sam_feature = self.SAM(cfm_feature, T2)
        #
        # prediction1 = self.out_CFM(cfm_feature)
        # prediction2 = self.out_SAM(sam_feature)
        #
        prediction1_4 = F.interpolate(prediction1, scale_factor=4, mode='bilinear')
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')
        prediction3_16 = F.interpolate(prediction3, scale_factor=16, mode='bilinear')
        prediction4_32 = F.interpolate(prediction4, scale_factor=32, mode='bilinear')
        #prediction=prediction1_4 + prediction2_8+prediction3_16+prediction4_32
        #x_2nd=x[:,1,:,:]
        #x_2nd=x_2nd[:,None,:,:]
        #pdb.set_trace()
        return prediction1_4, prediction2_8, prediction3_16, prediction4_32

class HSNet_with_aux(nn.Module):
    def __init__(self, channel=32):
        super(HSNet_with_aux, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.decoder = Decoder()
        #self.decoder2 = Decoder()

        # self.Translayer2_0 = BasicConv2d(64, channel, 1)
        # self.Translayer2_1 = BasicConv2d(128, channel, 1)
        # self.Translayer3_1 = BasicConv2d(320, channel, 1)
        # self.Translayer4_1 = BasicConv2d(512, channel, 1)
        #
        # self.CFM = CFM(channel)
        # self.ca = ChannelAttention(64)
        # self.sa = SpatialAttention()
        # self.SAM = SAM()
        #
        # self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        # self.out_SAM = nn.Conv2d(channel, 1, 1)
        # self.out_CFM = nn.Conv2d(channel, 1, 1)
        self.out1 = nn.Conv2d(64, 1, 1)
        self.out2 = nn.Conv2d(128, 1, 1)
        self.out3 = nn.Conv2d(320, 1, 1)
        self.out4 = nn.Conv2d(512, 1, 1)
        
        self.out1_a = nn.Conv2d(64, 3, 1)
        self.out2_a = nn.Conv2d(128, 3, 1)
        self.out3_a = nn.Conv2d(320, 3, 1)
        self.out4_a = nn.Conv2d(512, 3, 1)
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dc1 = nn.Conv2d(64, 64, 1)
        self.dc2 = nn.Conv2d(128, 64, 1)
        self.dc3 = nn.Conv2d(320, 64, 1)
        self.dc4 = nn.Conv2d(512, 64, 1)
        self.bn_dc1 = nn.BatchNorm2d(64)
        self.bn_dc2 = nn.BatchNorm2d(64)
        self.bn_dc3 = nn.BatchNorm2d(64)
        self.bn_dc4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        #with torch.no_grad():
        pvt = self.backbone(x)
        x1, x2, x3, x4 = self.decoder(pvt)    

        B = x1.shape[0]
        y1 = self.pooling(self.bn_dc1(self.dc1(x1)))#.detach()
        y2 = self.pooling(self.bn_dc2(self.dc2(x2)))#.detach()
        y3 = self.pooling(self.bn_dc3(self.dc3(x3)))#.detach()
        y4 = self.pooling(self.bn_dc4(self.dc4(x4)))#.detach()
        y = y1 + y2 + y3 + y4
        coeff = self.sigmoid(self.fc2(self.relu(self.fc1(y.reshape(B, -1)))))
        prediction1 = self.out1(x1) * coeff[:,0].reshape(B, 1, 1, 1)
        prediction2 = self.out2(x2) * coeff[:,1].reshape(B, 1, 1, 1)
        prediction3 = self.out3(x3) * coeff[:,2].reshape(B, 1, 1, 1)
        prediction4 = self.out4(x4) * coeff[:,3].reshape(B, 1, 1, 1)
        
        prediction1_a = self.out1_a(x1) 
        prediction2_a = self.out2_a(x2) 
        prediction3_a = self.out3_a(x3) 
        prediction4_a = self.out4_a(x4)
                
        prediction1_4 = F.interpolate(prediction1, scale_factor=4, mode='bilinear')
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')
        prediction3_16 = F.interpolate(prediction3, scale_factor=16, mode='bilinear')
        prediction4_32 = F.interpolate(prediction4, scale_factor=32, mode='bilinear')
        
        prediction1_4_a = F.interpolate(prediction1_a, scale_factor=4, mode='bilinear')
        prediction2_8_a = F.interpolate(prediction2_a, scale_factor=8, mode='bilinear')
        prediction3_16_a = F.interpolate(prediction3_a, scale_factor=16, mode='bilinear')
        prediction4_32_a = F.interpolate(prediction4_a, scale_factor=32, mode='bilinear')
        
        return prediction1_4, prediction2_8, prediction3_16, prediction4_32,prediction1_4_a, prediction2_8_a, prediction3_16_a, prediction4_32_a



if __name__ == '__main__':
    model = PolypPVT().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
