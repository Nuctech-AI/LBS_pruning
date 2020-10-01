import torch.nn as nn
import torch
import copy
import numpy as np
import collections
def pruning_grad_densent_layer(namex,layer,index=None,groups_y=None,re_num=None):
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
def get_ori_channels(model):
    channels = collections.OrderedDict()
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer,torch.nn.Linear):
            namex = name + '.weight'
            channels[namex] = layer.weight.data.shape[1]
    return channels
def pruning_googlenet_grad(model,config_prune=None,re_ind=None):

    ori_shape = get_ori_shape(model)
    ori_channels=get_ori_channels(model)
    index=None
    layer_index = collections.OrderedDict()
    ori_model=copy.deepcopy(model)
    #ori_model.cuda()
    concat_fig=config_prune['concat']

    layer_list = list(model.state_dict().keys())
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            ne=name+'.weight'
            if ne not in re_ind:
                #print(ne)
                re_ind[ne]=None

    offset=collections.OrderedDict()
    for index_i,i in enumerate(concat_fig):
        if index_i==0:
            continue
        for index_j,j in enumerate(i):
            offset[j]=concat_fig[index_i-1]
    for name_cnv,name_x in offset.items():
        for ll,each_x in enumerate(name_x):
            if 'branch3x3' in each_x:
                name_x[ll]=name_x[ll][:-8]+'3'+name_x[ll][-7:]
            if 'branch5x5' in each_x:
                name_x[ll] =name_x[ll][:-8]+'6'+name_x[ll][-7:]

    cov_index=collections.OrderedDict()
    for name_cnv, name_x in offset.items():
        offset1=0
        cov_index1=None
        for ll, each_x in enumerate(name_x):
            if ll==0:
                offset1=0
                cov_index1=copy.deepcopy(re_ind[each_x])
            else:
                offset1=offset1+ori_shape[name_x[ll-1]]
                if re_ind[each_x] is None:
                    continue
                if cov_index1 is None:
                    cov_index1=copy.deepcopy(re_ind[each_x])+offset1
                else:
                    cov_index1=np.hstack((cov_index1,(re_ind[each_x]+offset1)))
        if cov_index1 is not  None:
            cov_index1.sort()
        cov_index[name_cnv]=cov_index1


    index = None
    for name,layer in model.named_modules():
        if isinstance(layer,torch.nn.Conv2d) or isinstance(layer,torch.nn.BatchNorm2d) or isinstance(layer,torch.nn.Linear):
            namex = name + '.weight'
            #print(namex)
            if isinstance(layer,torch.nn.Conv2d):
                #ori_shape[namex]=layer.weight.data.shape[0]
                if namex in cov_index:
                    a = np.linspace(0, ori_channels[namex] - 1, num=ori_channels[namex])
                    if cov_index[namex] is not None:
                        index = np.setdiff1d(a, cov_index[namex])
                    else:
                        index=a
                layer, index = pruning_grad_densent_layer(namex, layer, index,re_num=re_ind[namex])
                layer_index[namex]=index
            elif isinstance(layer,torch.nn.BatchNorm2d):
                layer,index=pruning_grad_densent_layer(namex,layer,index)
                layer_index[namex]=index
            elif isinstance(layer,torch.nn.Linear):
                #ori_shape[namex] = layer.weight.data.shape[0]
                if namex in cov_index:
                    a = np.linspace(0, ori_channels[namex] - 1, num=ori_channels[namex])
                    if cov_index[namex] is not None:
                        index = np.setdiff1d(a, cov_index[namex])
                    else:
                        index=a
                layer,index=pruning_grad_densent_layer(namex,layer,index)
                layer_index[namex]=index
            else:
                continue
        else:
            continue
    return model