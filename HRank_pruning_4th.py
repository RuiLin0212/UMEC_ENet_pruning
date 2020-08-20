# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:18:28 2020

@author: RuiLIN
"""
""" load pretrained ENet_lite0 model """""" import packages """
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import time
import os
import torchvision.datasets as datasets
import numpy as np
from torch.autograd import Variable

""" import necessary packages for ENet_lite0 """
from efficientnet_lite_pytorch import EfficientNet
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile

""" Load lite0_model """
weights_path = EfficientnetLite0ModelFile.get_model_file_path()
print(weights_path) # print .pth file's path
lite0_model = EfficientNet.from_pretrained('efficientnet-lite0', weights_path = weights_path) # load the model

""" expand conv """
def HRank_expand(index, data):
    new_weight = data[index,:,:,:]
    return new_weight

""" depthwise conv """
def HRank_depthwise(index, data):
    new_weight = data[index,:,:,:]
    return new_weight

""" project conv """
def HRank_project(index, data):
    new_weight = data[:,index,:,:]
    return new_weight

""" BN """
def HRank_BN(index, mean, var, weight, bias):
    new_mean = mean[index]
    new_var = var[index]
    new_weight = weight[index]
    new_bias = bias[index]
    return new_mean, new_var, new_weight, new_bias

""" prune the model """
def prune_layer(idx_block, num_channel):
    idx = np.sort(np.load("./Pruning_idx/4th/block_" + str(idx_block) + ".npy"))
    # epand conv
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._expand_conv.out_channels =" +  str(num_channel))
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._expand_conv.weight.data = HRank_expand(idx, lite0_model._blocks[" + str(idx_block) + "]._expand_conv.weight.data)")
    # bn
    exec("new_mean, new_var, new_weight, new_bias = HRank_BN(idx, lite0_model._blocks[" + str(idx_block) + "]._bn0.running_mean.data, lite0_model._blocks[" + str(idx_block) + "]._bn0.running_var.data, lite0_model._blocks[" + str(idx_block) + "]._bn0.weight.data, lite0_model._blocks[" + str(idx_block) + "]._bn0.bias.data)")
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._bn0 = nn.BatchNorm2d(" + str(num_channel) + ", eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)")
    exec("lite0_model._blocks[" + str(idx_block) + "]._bn0.running_mean.data = new_mean")
    exec("lite0_model._blocks[" + str(idx_block) + "]._bn0.running_var.data = new_var")
    exec("lite0_model._blocks[" + str(idx_block) + "]._bn0.weight.data = new_weight")
    exec("lite0_model._blocks[" + str(idx_block) + "]._bn0.bias.data = new_bias")
    # depthwise conv
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._depthwise_conv.groups = " + str(num_channel))
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._depthwise_conv.in_channels = " + str(num_channel))
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._depthwise_conv.out_channels = " + str(num_channel))
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._depthwise_conv.weight.data = HRank_depthwise(idx, lite0_model._blocks[" + str(idx_block) + "]._depthwise_conv.weight.data)")
    # bn1 
    exec("new_mean, new_var, new_weight, new_bias = HRank_BN(idx, lite0_model._blocks[" + str(idx_block) + "]._bn1.running_mean.data, lite0_model._blocks[" + str(idx_block) + "]._bn1.running_var.data, lite0_model._blocks[" + str(idx_block) + "]._bn1.weight.data, lite0_model._blocks[" + str(idx_block) + "]._bn1.bias.data)")
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._bn1 = nn.BatchNorm2d(" + str(num_channel) + ", eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)")
    exec("lite0_model._blocks[" + str(idx_block) + "]._bn1.running_mean.data = new_mean")
    exec("lite0_model._blocks[" + str(idx_block) + "]._bn1.running_var.data = new_var")
    exec("lite0_model._blocks[" + str(idx_block) + "]._bn1.weight.data = new_weight")
    exec("lite0_model._blocks[" + str(idx_block) + "]._bn1.bias.data = new_bias")
    # project conv 
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._project_conv.in_channels = " + str(num_channel))
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._project_conv.weight.data = HRank_project(idx, lite0_model._blocks[" + str(idx_block) + "]._project_conv.weight.data)")
    return

""" modify the model """
def modify_layer(idx_block, num_channel, expand_conv, depth_conv, project_conv):
    # epand conv
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._expand_conv.out_channels =" +  str(num_channel))
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._expand_conv.weight.data = torch.rand" + "(" + str(expand_conv) + ")")
    # bn
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._bn0 = nn.BatchNorm2d(" + str(num_channel) + ", eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)")
    # depthwise conv
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._depthwise_conv.groups = " + str(num_channel))
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._depthwise_conv.in_channels = " + str(num_channel))
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._depthwise_conv.out_channels = " + str(num_channel))
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._depthwise_conv.weight.data = torch.rand" + "(" + str(depth_conv) + ")")
    # bn1 
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._bn1 = nn.BatchNorm2d(" + str(num_channel) + ", eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)")
    # project conv 
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._project_conv.in_channels = " + str(num_channel))
    exec("lite0_model._blocks" + "[" + str(idx_block) + "]" + "._project_conv.weight.data = torch.rand" + "(" + str(project_conv) + ")")
    return

# Block 9
modify_layer(9, 557, [557,112,1,1], [557,1,5,5], [112,557,1,1])
# Block 10
modify_layer(10, 567, [567,112,1,1], [567,1,5,5], [112,567,1,1])
# Block 11
modify_layer(11, 588, [588,112,1,1], [588,1,5,5], [192,588,1,1])
# Block 12
modify_layer(12, 768, [768,192,1,1], [768,1,5,5], [192,768,1,1])
# Block 13
modify_layer(13, 768, [768,192,1,1], [768,1,5,5], [192,768,1,1])
# Block 14
modify_layer(14, 768, [768,192,1,1], [768,1,5,5], [192,768,1,1])
# Block 15
modify_layer(15, 365, [365,192,1,1], [365,1,3,3], [320,365,1,1])

""" load model """
# original saved file with DataParallel
state_dict = torch.load('./Normal_checkpoint_3rd/ReLU/Fine_tune_params_7.pth')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
lite0_model.load_state_dict(new_state_dict)

""" prune the model """
# block 12
prune_layer(12, 735)
# block 13
prune_layer(13, 737)
# block 14
prune_layer(14, 717)
# block 15
prune_layer(15, 258)

""" save the model """
torch.save(lite0_model.state_dict(), 'HRank_checkpoint_4th.pth')