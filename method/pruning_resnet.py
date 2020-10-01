import torch.nn as nn
import torch
import copy
import numpy as np
import collections
from torch.autograd import Variable
def get_ori_shape(model):
    ori_shape = collections.OrderedDict()
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer,torch.nn.Linear):
            namex = name + '.weight'
            if layer.weight is not None:
                ori_shape[namex] = layer.weight.data.shape[0]
            else:
                ori_shape[namex] = layer.running_mean.data.shape[0]
    return ori_shape
def pruning_resnet_grad(model,config_prune,re_ind=None):

    downsample_Flag = False
    ori_shape = get_ori_shape(model)
    if config_prune['downsample'] is not None:
        downsample_config=config_prune['downsample']
        downsample_Flag=True

    index=None
    layer_index = collections.OrderedDict()
    ori_shape = collections.OrderedDict()
    ori_model=copy.deepcopy(model)
    #ori_model.cuda()
    for name,layer in model.named_modules():
        if isinstance(layer,torch.nn.Conv2d) or isinstance(layer,torch.nn.BatchNorm2d) or isinstance(layer,torch.nn.Linear):
            namex = name + '.weight'
            #print(namex)
            if downsample_Flag:
                if namex in downsample_config:
                    index = layer_index[downsample_config[namex]]
            if isinstance(layer,torch.nn.Conv2d):
                ori_shape[namex]=layer.weight.data.shape[0]
                layer, index = pruning_grad_resnet_layer(namex, layer,index,re_num=re_ind[namex])
                layer_index[namex]=index
            elif isinstance(layer,torch.nn.BatchNorm2d):
                ori_shape[namex] = layer.running_mean.data.shape[0]
                layer,index=pruning_grad_resnet_layer(namex,layer,index)
                layer_index[namex]=index
            elif isinstance(layer,torch.nn.Linear):
                ori_shape[namex] = layer.weight.data.shape[0]
                layer,index=pruning_grad_resnet_layer(namex,layer,index)
                layer_index[namex]=index
            else:
                continue
        else:
            continue
    return model
def pruning_grad_resnet_layer(namex,layer,index=None,re_num=None):
    if isinstance(layer, nn.Conv2d):
        if index is not None:
            layer.weight.data = layer.weight.data[:, index]
            layer.in_channels = index.shape[0]
        weight = layer.weight.data
        a = np.linspace(0, weight.shape[0]-1, num=weight.shape[0])
        if re_num is None:
            dsa=a
        else:
            dsa_ori=re_num
            dsa=np.setdiff1d(a,dsa_ori)
        dsa.sort()
        layer.weight.data = layer.weight.data[dsa]
        if layer.bias is not None:
            layer.bias.data = layer.bias.data[dsa]
        layer.out_channels = dsa.shape[0]
        return layer, dsa
    elif isinstance(layer, nn.BatchNorm2d):
        if layer.weight is not None:
            layer.weight.data=layer.weight.data[index]
        if layer.bias is not None:
            layer.bias.data=layer.bias.data[index]
        layer.running_mean.data = layer.running_mean.data[index]
        layer.running_var.data = layer.running_var.data[index]
        layer.num_features = index.shape[0]
        return layer, index
    elif isinstance(layer, torch.nn.Linear):
        layer.weight.data = layer.weight.data[:, index]
        layer.in_features = index.shape[0]
        return layer,index
    else:
        print(layer)