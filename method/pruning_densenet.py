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

def pruning_grad_densenet(model,config_prune=None,re_ind=None):

    ori_shape = get_ori_shape(model)
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
    for i in concat_fig:
        for index,j in enumerate(i):
            if index%2!=0:
                offset_name=[]
                kk=i[:index]
                for index2,jj in enumerate(kk):
                    if index2%2==0:
                        offset_name.append(jj)
                offset[j]=offset_name
    bn_idx=collections.OrderedDict()
    for bn_name,layer_up in offset.items():
        offset_num=0
        dsa=None

        for layer_i in layer_up:
            if dsa is None:
                if re_ind[layer_i] is None:
                    offset_num=offset_num+ori_shape[layer_i]
                    continue
                else:
                    dsa=re_ind[layer_i]+offset_num
            else:
                if re_ind[layer_i] is not None:
                    z=re_ind[layer_i]+offset_num
                    dsa=np.hstack((dsa,z))
                else:
                    offset_num = offset_num+ori_shape[layer_i]
                    continue
            offset_num = offset_num+ori_shape[layer_i]
        bn_idx[bn_name]=dsa

    index = None
    for name,layer in model.named_modules():
        if isinstance(layer,torch.nn.Conv2d) or isinstance(layer,torch.nn.BatchNorm2d) or isinstance(layer,torch.nn.Linear):
            namex = name + '.weight'
            #print(namex)


            if isinstance(layer,torch.nn.Conv2d):
                #ori_shape[namex]=layer.weight.data.shape[0]
                layer, index = pruning_grad_densent_layer(namex, layer, index,re_num=re_ind[namex])
                layer_index[namex]=index
            elif isinstance(layer,torch.nn.BatchNorm2d):
                #ori_shape[namex] = layer.running_mean.data.shape[0]

                if namex in bn_idx:
                    index=bn_idx[namex]
                    a = np.linspace(0, ori_shape[namex] - 1, num=ori_shape[namex])
                    index = np.setdiff1d(a, index)
                layer,index=pruning_grad_densent_layer(namex,layer,index)
                layer_index[namex]=index
            elif isinstance(layer,torch.nn.Linear):
                #ori_shape[namex] = layer.weight.data.shape[0]
                layer,index=pruning_grad_densent_layer(namex,layer,index)
                layer_index[namex]=index
            else:
                continue
        else:
            continue
    return model