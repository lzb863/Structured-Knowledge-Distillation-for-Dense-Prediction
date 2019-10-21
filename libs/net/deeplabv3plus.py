# encoding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from libs.net.backbone import build_backbone
from libs.net.ASPP import ASPP


class deeplabv3plus(nn.Module):
    def __init__(self, args):
        super(deeplabv3plus, self).__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048

        args.TRAIN_BN_MOM = 0.1
        self.aspp = ASPP(dim_in=input_channel,
                         dim_out=args.MODEL_ASPP_OUTDIM,
                         rate=16 // args.MODEL_OUTPUT_STRIDE,
                         bn_mom=args.TRAIN_BN_MOM
                         )
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=args.MODEL_OUTPUT_STRIDE // 4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, args.MODEL_SHORTCUT_DIM, args.MODEL_SHORTCUT_KERNEL, 1,
                      padding=args.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(args.MODEL_SHORTCUT_DIM, momentum=args.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(args.MODEL_ASPP_OUTDIM + args.MODEL_SHORTCUT_DIM, args.MODEL_ASPP_OUTDIM, 3, 1, padding=1,
                      bias=True),
            SynchronizedBatchNorm2d(args.MODEL_ASPP_OUTDIM, momentum=args.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(args.MODEL_ASPP_OUTDIM, args.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(args.MODEL_ASPP_OUTDIM, momentum=args.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(args.MODEL_ASPP_OUTDIM, args.num_classes, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(args.MODEL_BACKBONE, os=args.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers() # 1/16
        # print("layers", layers[-1].size())
        feature_aspp = self.aspp(layers[-1])
        feature_aspp_1 = self.dropout1(feature_aspp) # 1/8
        feature_aspp_2 = self.upsample_sub(feature_aspp_1) # 1/4
        # print("feature_aspp ", feature_aspp.size())
        feature_shallow = self.shortcut_conv(layers[0])
        # print("layers[0]", layers[0].size())
        feature_cat = torch.cat([feature_aspp_2, feature_shallow], 1)
        # print("feature_cat ", feature_cat.size())
        result_1 = self.cat_conv(feature_cat)
        result_2 = self.cls_conv(result_1)
        result = self.upsample4(result_2)
        return [result, feature_cat, result_2, feature_aspp_1, feature_aspp_2, feature_shallow, feature_cat ]
    
    # for ts
    def forward_till_aspp(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers() # 1/16
        # print("layers", layers[-1].size())
        feature_aspp = self.aspp(layers[-1])
        return feature_aspp

    def aspp_to_catFeat(self, feature_aspp):
        feature_aspp = self.dropout1(feature_aspp) # 1/8
        feature_aspp = self.upsample_sub(feature_aspp) # 1/4
        layers = self.backbone.get_layers() # 1/16
        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        return feature_cat

    def catFeat_to_predict(self, feature_cat):
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result
