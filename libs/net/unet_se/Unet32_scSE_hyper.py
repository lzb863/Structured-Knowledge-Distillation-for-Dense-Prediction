import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import libs.loss.lovasz_losses as L
from libs.net.sync_batchnorm import SynchronizedBatchNorm2d

from libs.net.unet_se.resnets import *

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        #self.bn = SynchronizedBatchNorm2d(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x

class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = ConvBn2d(in_channels=out_channels,out_channels=1,kernel_size=1,padding=0)
    def forward(self,x):
        x=self.conv(x)
        #print('spatial',x.size())
        x=F.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = ConvBn2d(in_channels=out_channels,out_channels=int(out_channels/2),kernel_size=1,padding=0)
        self.conv2 = ConvBn2d(in_channels=int(out_channels/2),out_channels=out_channels,kernel_size=1,padding=0)
    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        #print('channel',x.size())
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.sigmoid(x)
        return x



class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x, e=None):
        if x.size()[3]!=128:
           x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        #print('x',x.size())
        #if e is not None: print('e',e.size())
        if e is not None:
            x = torch.cat([x,e],1)

        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        #print('x_new',x.size())
        g1 = self.spatial_gate(x)
        #print('g1',g1.size())
        g2 = self.channel_gate(x)
        #print('g2',g2.size())
        x = g1*x + g2*x

        return x

class Unet_scSE_hyper(nn.Module):

    def criterion(self,logit, truth):
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)
        loss = L.lovasz_hinge(logit, truth, per_image=True, ignore=None)
        return loss
    
    def criterion2(self,logit, truth):
        metric = torch.nn.BCEWithLogitsLoss(size_average=True, reduction='none')
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)
        loss = metric(logit, truth)
        return loss

    def __init__(self, cfg):
        super().__init__()
        self.resnet = resnet32()
        checkpoint = torch.load('')
        # original saved file with DataParallel
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        state_dict=checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
          name = k[7:] # remove `module.`
          new_state_dict[name] = v
        #print(checkpoint['state_dict'])
        self.resnet.load_state_dict(new_state_dict)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            nn.ReLU(inplace=True),
        )

        self.encoder2 = self.resnet.layer1 # 16
        self.encoder3 = self.resnet.layer2 #32
        self.encoder4 = self.resnet.layer3 #64
        #self.encoder5 = self.resnet.layer4 #512

        self.center = nn.Sequential(
            ConvBn2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        #self.decoder5 = Decoder(256+512,512,64)
        self.decoder4 = Decoder(64 +64,64,64)
        self.decoder3 = Decoder(64 +32,32,64)
        self.decoder2 = Decoder(64 +16 ,16 ,64)
        self.decoder1 = Decoder(64 +16 ,8 ,64)

        self.logit = nn.Sequential(
            nn.Conv2d(272, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, cfg.MODEL_NUM_CLASSES, kernel_size=1, padding=0),
        )

    def forward(self, x):
        mean=[0.485, 0.456, 0.406]
        std=[0.229,0.224,0.225]
        x=torch.cat([
           (x-mean[2])/std[2],
           (x-mean[1])/std[1],
           (x-mean[0])/std[0],
        ],1)

        e1 = self.conv1(x)
        #print("e1",e1.size())
        e2 = self.encoder2(e1)
        #print('e2',e2.size())
        e3 = self.encoder3(e2)
        #print('e3',e3.size())
        e4 = self.encoder4(e3)
        #print('e4',e4.size())
        #e5 = self.encoder5(e4)
        #print('e5',e5.size())

        f = self.center(e4)
        #print('f',f.size())
        #d5 = self.decoder5(f, e5)
        d4 = self.decoder4(f,e4)
        #print('d4',d4.size())
        d3 = self.decoder3(d4,e3)
        #print('d3',d3.size())
        d2 = self.decoder2(d3,e2)
        #print('d2',d4.size())
        d1 = self.decoder1(d2,e1)
        #print('d1',d1.size())
        #print('e1',e1.size())
        #print('d2',d2.size())
        #print('d3',d3.size())
        #print('d4',d4.size())
        #print('d1',d1.size())

        f = torch.cat((
            F.upsample(e1,scale_factor= 1, mode='bilinear',align_corners=False),
            d1,
            F.upsample(d2,scale_factor= 1, mode='bilinear',align_corners=False),
            F.upsample(d3,scale_factor= 2, mode='bilinear',align_corners=False),
            F.upsample(d4,scale_factor= 4, mode='bilinear',align_corners=False),
            #F.upsample(d5,scale_factor=16, mode='bilinear',align_corners=False),
        ),1)

        f = F.dropout2d(f,p=0.5)
        #print('f',f.size())
        logit = self.logit(f)
        #print('logit',logit.size())
        return logit


    def criterion1(self, logit, truth ):
        loss = FocalLoss2d(gamma=0.5)(logit, truth, type='sigmoid')
        return loss

    # def criterion(self,logit, truth):
    #     loss = F.binary_cross_entropy_with_logits(logit, truth)
    #     return loss

    def metric(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        #dice = dice_accuracy(prob, truth, threshold=threshold, is_average=True)
        dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        return dice

    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d) or isinstance(m,SynchronizedBatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False

        else:   
        	raise NotImplementedError
