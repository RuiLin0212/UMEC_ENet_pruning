# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:49:33 2020

@author: RuiLIN
"""
""" import packages """
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import re

""" import necessary packages for ENet_lite0 """
from efficientnet_lite_pytorch import EfficientNet
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile

""" Load lite0_model """
weights_path = EfficientnetLite0ModelFile.get_model_file_path()
print(weights_path) # print .pth file's path
lite0_model = EfficientNet.from_pretrained('efficientnet-lite0', weights_path = weights_path) # load the model

""" modify model function """
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

""" modify model """
# Block 9
modify_layer(9, 557, [557,112,1,1], [557,1,5,5], [112,557,1,1])
# Block 10
modify_layer(10, 567, [567,112,1,1], [567,1,5,5], [112,567,1,1])
# Block 11
modify_layer(11, 588, [588,112,1,1], [588,1,5,5], [192,588,1,1])
# Block 12
modify_layer(12, 735, [735,192,1,1], [735,1,5,5], [192,735,1,1])
# Block 13
modify_layer(13, 737, [737,192,1,1], [737,1,5,5], [192,737,1,1])
# Block 14
modify_layer(14, 717, [717,192,1,1], [717,1,5,5], [192,717,1,1])
# Block 15
modify_layer(15, 258, [258,192,1,1], [258,1,3,3], [320,258,1,1])

""" Replace ReLU6 by ReLU """
for layer in lite0_model.named_modules():
    if isinstance(layer[1],nn.ReLU6):
        if layer[0][1:].isalpha():
            exec("lite0_model." + layer[0] + "= nn.ReLU()")
        else:
            num = re.sub("\D", "", layer[0])
            exec("lite0_model." + layer[0][0:7] + "[" + num + "]" + layer[0][-7:] + "= nn.ReLU()")
print(lite0_model)

""" load model """
# original saved file with DataParallel
state_dict = torch.load('./checkpoint/ReLU/4th/Fine_tune_params_5.pth')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
lite0_model.load_state_dict(new_state_dict)

"""  Deploy model on GPU """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(lite0_model)
model = model.to(device)

""" Download Dataset """
def gen_loaders(path, BATCH_SIZE, NUM_WORKERS):
    # Data loading code
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)

    return (train_loader, val_loader)

train_loader, val_loader = gen_loaders('/data/DeepLearning/ILSVRC2012', 80, 4)


""" Optimizer & Criterion """
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
criterion = nn.CrossEntropyLoss()

""" fine tune """
def fine_tune(epoch, log_interval=200):
    global eigs1_epoch, eigs2_epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{:0>5d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx * len(data) / len(train_loader.dataset), loss.detach().item()))
    torch.save(model.state_dict(),'./checkpoint/ReLU/4th/resume/Fine_tune_params_' + str(epoch+1) + '.pth')
            
""" test """
def test(epoch):
    global new_predict
    model.eval()
    val_loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(val_loader):
        data = data.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = model(data)
            val_loss += criterion(output, target).item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum()

    val_loss /= len(val_loader) 
    accuracy = 100. * correct.to(torch.float32) / len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), accuracy))
    
""" train & test """
epochs = 50
for epoch in range(epochs):
    fine_tune(epoch)
    test(epoch)
