import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import libs.loss.lovasz_losses as L
from libs.net.sync_batchnorm import SynchronizedBatchNorm2d


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
        self.conv = nn.Conv2d(in_channels=out_channels,out_channels=1,kernel_size=1,padding=0)
    def forward(self,x):
        x=self.conv(x)
        #print('spatial',x.size())
        x=F.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=out_channels,out_channels=int(out_channels/2),kernel_size=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=int(out_channels/2),out_channels=out_channels,kernel_size=1,padding=0)
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
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        #print('x',x.size())
        #print('e',e.size())
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

    def __init__(self, cfg):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )

        self.encoder2 = self.resnet.layer1 #256
        self.encoder3 = self.resnet.layer2 #512
        self.encoder4 = self.resnet.layer3 #1024
        self.encoder5 = self.resnet.layer4 #2048

        self.center = nn.Sequential(
            ConvBn2d(2048,2048,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(2048,1024,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.decoder5 = Decoder(256+512,512,64)
        self.decoder4 = Decoder(64 +256,256,64)
        self.decoder3 = Decoder(64 +128,128,64)
        self.decoder2 = Decoder(64 +64 ,64 ,64)
        self.decoder1 = Decoder(64     ,32 ,64)

        self.logit = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, padding=1),
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
        #print(e1.size())
        e2 = self.encoder2(e1)
        #print('e2',e2.size())
        e3 = self.encoder3(e2)
        #print('e3',e3.size())
        e4 = self.encoder4(e3)
        #print('e4',e4.size())
        e5 = self.encoder5(e4)
        #print('e5',e5.size())

        f = self.center(e5)
        #print('f',f.size())
        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5,e4)
        d3 = self.decoder3(d4,e3)
        d2 = self.decoder2(d3,e2)
        d1 = self.decoder1(d2)
        #print('d1',d1.size())

        f = torch.cat((
            F.upsample(e1,scale_factor= 2, mode='bilinear',align_corners=False),
            d1,
            F.upsample(d2,scale_factor= 2, mode='bilinear',align_corners=False),
            F.upsample(d3,scale_factor= 4, mode='bilinear',align_corners=False),
            F.upsample(d4,scale_factor= 8, mode='bilinear',align_corners=False),
            F.upsample(d5,scale_factor=16, mode='bilinear',align_corners=False),
        ),1)

        f = F.dropout2d(f,p=0.50)
        logit = self.logit(f)
        return logit


    def criterion1(self, logit, truth ):
        loss = FocalLoss2d()(logit, truth, type='sigmoid')
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

def accuracy(prob, truth, threshold=0.5,  is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size,-1)
    t = truth.detach().view(batch_size,-1)

    p = p>threshold
    t = t>0.5
    correct = ( p == t).float()
    accuracy = correct.sum(1)/p.size(1)

    if is_average:
        accuracy = accuracy.sum()/batch_size
        return accuracy
    else:
        return accuracy


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()


        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]

            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C

            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss
