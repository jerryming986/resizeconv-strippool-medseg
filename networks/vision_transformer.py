# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import os

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def _get_convnext_pretrained(self, config):
        """获取 ConvNeXt-T 的 ImageNet 预训练 state_dict。
        优先用 config.MODEL.PRETRAIN_CKPT 指向的本地 convnext 权重文件；
        否则用 timm 下载/读取本地 HF 缓存。失败则返回 None。"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        path = config.MODEL.PRETRAIN_CKPT
        if path and os.path.exists(path) and 'convnext' in os.path.basename(path).lower():
            print("load convnext pretrained from file:{}".format(path))
            sd = torch.load(path, map_location=device)
            return sd.get('model', sd)
        try:
            import timm
            print("load convnext_tiny pretrained via timm (HF cache)")
            return timm.create_model('convnext_tiny', pretrained=True).state_dict()
        except Exception as e:
            print("WARNING: failed to obtain ConvNeXt pretrained weights ->", repr(e))
            return None

    def load_from(self, config):
        import re
        pre = self._get_convnext_pretrained(config)
        if pre is None:
            print("none pretrain (train from scratch)")
            return

        print("---start load ConvNeXt-T pretrained into encoder & mirrored decoder---")
        # 编码器：stem.* / stages.*.{downsample,conv_dw,norm,mlp,gamma} 直接同名命中。
        # BIM 增强分支(norm_bim/bim/gamma_bim)为全新模块、无预训练对应 -> 自动被过滤。
        full_dict = dict(pre)
        # 解码器：把编码器 stage 2/1/0 的 block 权重(conv_dw/norm/mlp/gamma)镜像到 up_stages 0/1/2
        for k, v in pre.items():
            m = re.match(r'stages\.(\d+)\.blocks\.(\d+)\.(norm|mlp|gamma|conv_dw).*', k)
            if m and int(m.group(1)) in (0, 1, 2):
                up = 2 - int(m.group(1))   # enc stage 2/1/0 -> decoder up_stages 0/1/2
                full_dict[k.replace("stages.%s" % m.group(1), "up_stages.%d" % up, 1)] = v

        model_dict = self.swin_unet.state_dict()
        full_dict = {k: v for k, v in full_dict.items()
                     if k in model_dict and v.shape == model_dict[k].shape}

        loaded = sum(model_dict[k].numel() for k in full_dict)
        total = sum(v.numel() for v in model_dict.values())
        print("matched tensors:{} | loaded params:{:.2f}M / {:.2f}M ({:.1%})".format(
            len(full_dict), loaded / 1e6, total / 1e6, loaded / total))

        msg = self.swin_unet.load_state_dict(full_dict, strict=False)
        from_scratch = sum(model_dict[k].numel() for k in msg.missing_keys)
        print("from-scratch params (BIM enhancement + up/down-sample + head):{:.2f}M".format(from_scratch / 1e6))
 