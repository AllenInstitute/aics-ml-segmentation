import torch
from torchvision.models.segmentation import deeplabv3_resnet101
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# for debugging
import pdb


class deeplabV3(nn.Module):
    def __init__(self, nclass=[2], pretrain=True): # add to use pretrained or not 
        super(deeplabV3,self).__init__()
        
        self.num_class = nclass[0]
        model = deeplabv3_resnet101(pretrained=pretrain, progress=True, aux_loss=None) # using deeplabv3

        # change the classifier
        model.classifier[4] = nn.Conv2d(256, self.num_class, kernel_size=(1, 1), stride=(1, 1))
        self.deeplab = model
        self.final_activation = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.deeplab(x)['out'] 

        out = x.permute(0,2,3,1).contiguous()
        out = out.view(out.numel() // self.num_class, self.num_class)
        out = F.log_softmax(out, dim=1)

        return out

class deeplabV3_plus(nn.Module):
    '''
    TODO: Implement altered xception as a backbone
    '''
    def __init__(self, nclass=[2], pretrain=True):
        super(deeplabV3_plus,self).__init__()

        self.num_class = nclass[0]

        inital_model = deeplabv3_resnet101(pretrained=pretrain, progress=True, aux_loss=None) # using deeplabv3

        self.backbone = inital_model.backbone

        self.aspp = nn.Sequential(
                                inital_model.classifier[0],
                                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU()
        )
        self.last_layer = nn.Sequential(
                                        nn.Conv2d(2304, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.Dropout(0.5, inplace=False),
                                        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.Dropout(0.1, inplace=False),
                                        nn.Conv2d(256, self.num_class, kernel_size=(1, 1), stride=(1, 1))

        )

        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x_low_level_feat = self.backbone(x)['out'] # low level feature
        x_aspp = self.aspp(x_low_level_feat) # after differnt ratio of atrous convolution

        x_decoder = torch.cat((x_aspp, x_low_level_feat), dim=1) # add low and high level feature 
        x_decoder = self.last_layer(x_decoder) # fully connected layer
        
        x_decoder = F.interpolate(x_decoder, size=x.size()[2:], mode='bilinear', align_corners=False) # Last upsampling layer

        out = x_decoder.permute(0,2,3,1).contiguous()
        out = out.view(out.numel() // self.num_class, self.num_class)
        out = F.log_softmax(out, dim=1)
        
        return out

# class Xception(nn.Moduel):
#     def __init__(self, )

