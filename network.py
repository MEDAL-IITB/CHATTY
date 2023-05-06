import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb
import random


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152, "ViT_B_16":models.vit_b_16, "Swin_T":models.swin_t}

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class ResNetFc(nn.Module):
  def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](weights='ResNet50_Weights.DEFAULT')
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
    self.select_layers = nn.Sequential(self.layer3, self.layer4, self.avgpool)

    self.use_bottleneck = use_bottleneck
    self.sigmoid = nn.Sigmoid()
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            #dual bridges
            self.trans1 = nn.Linear(bottleneck_dim, class_num)
            self.trans2 = nn.Linear(bottleneck_dim, class_num)

            self.focal1 = nn.Linear( class_num,class_num)
            self.focal2 = nn.Linear( class_num,1)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)

            #initialize with different weights
            self.trans1.apply(init_weights)
            self.trans2.apply(init_weights)

            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)

            self.trans1 = nn.Linear(model_resnet.fc.in_features, class_num)
            self.trans1.apply(init_weights)
            self.trans2 = nn.Linear(model_resnet.fc.in_features, class_num)
            self.trans2.apply(init_weights)
            
            self.__in_features = model_resnet.fc.in_features
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features

  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)

    bridge1 = self.trans1(x)
    bridge2 = self.trans2(x)
    y = self.fc(x) #classifier
    y1 = y- bridge1
    y2 = y - bridge2

    return x, y1, y2, bridge1, bridge2

  def output_num(self):
    return self.__in_features

  def GLL(self):
    return self.fc.weight

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2},
                            {"params":self.trans1.parameters(), "lr_mult":10, 'decay_mult':2},
                            {"params":self.trans2.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list

class TransformerFc(nn.Module):
  def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(TransformerFc, self).__init__()
    self.model_resnet = resnet_dict[resnet_name](weights='ViT_B_16_Weights.DEFAULT')
    self.model_resnet.load_state_dict(torch.load('/home/chirag/imagenet21k_ViT-B_16.npz'))
    #self.conv_proj = model_resnet.conv_proj
    #self.encoder = model_resnet.encoder
    #self.heads = torch.nn.Linear(in_features=768, out_features=2048, bias=True)
    #self.feature_layers = nn.Sequential(self.conv_proj, self.encoder, self.heads)
    self.model_resnet.heads = torch.nn.Linear(in_features=768, out_features=2048, bias=True)
    #self.model_resnet.head.out_features = 2048
    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(2048, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            #dual bridges
            self.trans1 = nn.Linear(bottleneck_dim, class_num)
            self.trans2 = nn.Linear(bottleneck_dim, class_num)

            self.focal1 = nn.Linear( class_num,class_num)
            self.focal2 = nn.Linear( class_num,1)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)

            #initialize with different weights
            self.trans1.apply(init_weights)
            self.trans2.apply(init_weights)

            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(2048, class_num)
            self.fc.apply(init_weights)

            self.trans1 = nn.Linear(2048, class_num)
            self.trans1.apply(init_weights)
            self.trans2 = nn.Linear(2048, class_num)
            self.trans2.apply(init_weights)
            
            #self.__in_features = self.feature_layers.heads.in_features
    else:
        self.fc = model_resnet.heads
        self.__in_features = self.model_resnet.heads.in_features

  def forward(self, x):
    x = self.model_resnet(x)
    x = x.view(x.size(0), -1)
    #print(x.size())
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)

    bridge1 = self.trans1(x)
    bridge2 = self.trans2(x)
    y = self.fc(x) #classifier
    y1 = y- bridge1
    y2 = y - bridge2

    return x, y1, y2, bridge1, bridge2

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.model_resnet.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.model_resnet.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2},
                            {"params":self.trans1.parameters(), "lr_mult":10, 'decay_mult':2},
                            {"params":self.trans2.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.transd = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.dropout3 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0


  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    z = self.transd(x)
    return y,z

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]