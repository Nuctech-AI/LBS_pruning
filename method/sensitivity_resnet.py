import collections
import pickle
import time
import torch.nn as nn
import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import os
def get_i_i(each_name_weight,stage):
    for i,stage_ea in enumerate(stage):
        for j , block_ea in enumerate(stage_ea):
            if block_ea == each_name_weight:
                return i ,j
def get_score_resnet(model,test_loader,args,stage=None):
    gpus = list(map(int, args.gpu))
    if args.cuda:
        model.cuda()
    model.eval()
    correct = 0
    conv_name=collections.OrderedDict()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_name[name]=m.weight.data.shape[0]
    conv_name_list=list(conv_name.keys())
    score_all=collections.OrderedDict()
    test_loss_ori = 0
    for each_name in conv_name_list:
        num_f=conv_name[each_name]
        each_name_weight=each_name+'.weight'
        score_all[each_name_weight] = np.zeros(num_f)
    for data, target in test_loader:
        indx_target = target.clone()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        optimizer.zero_grad()
        #output = model(data)
        output = nn.parallel.data_parallel(model, data, device_ids=gpus)
        loss= F.cross_entropy(output, target)
        loss.backward()
        for name, m in model.named_modules():
            name_weight=name+'.weight'
            if isinstance(m, torch.nn.Conv2d) and name_weight in score_all:
                grads_each= m.weight.grad.view(m.weight.grad.shape[0], -1)
                ori_weight=m.weight.data.view(m.weight.data.shape[0],-1)
                grad_dot_weight=grads_each*ori_weight
                grad_norm_1=torch.norm(grad_dot_weight, p=1, dim=1)
                grad_norm_1_index=grad_norm_1.argsort()
                grad_list=grad_norm_1_index.cpu().numpy().tolist()
                num=int(score_all[name_weight].shape[0])
                for i in range(int(score_all[name_weight].shape[0])):
                    score_all[name_weight][i]+=(1-(num-grad_list.index(i))/num)
        test_loss_ori+=loss.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()
    del data
    del target
    #acc = 100. * correct.type(torch.FloatTensor) / len(test_loader.dataset)
    #print("Accuracy:{:.4f}%".format(acc.numpy()))
    with open(os.path.join(args.result_dir,'score_all.pickle'),'wb') as json_file:
        pickle.dump(score_all,json_file)
    return score_all
def test_acc_resnet(model,test_loader,args,config=None,stage=None,all_score=None):
    model_ori=copy.deepcopy(model)
    delta_loss=config['delta_loss']
    gpus = list(map(int, args.gpu))
    #duiying={'layer2.0.conv2.weight':'layer1.8.conv2.weight',
    #         'layer3.0.conv2.weight':'layer2.8.conv2.weight'}
    channel_cor=None
    if config["channel_select"]:
        channel_cor=config["channel_cor"]

    correct = 0
    conv_name=collections.OrderedDict()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    if stage is not None:
        stage_num=len(stage)
        block_num=[]
        for i in range(stage_num):
            block_num.append(len(stage[i])-1)
        all_stage_name=sum(stage,[])
        for name_s_list in stage:
            channel_score=np.zeros(all_score[name_s_list[0]].shape[0])
            for each_name_weight in name_s_list:
                stage_i, block_i = get_i_i(each_name_weight, stage)
                block_num_now = block_num[stage_i]
                if block_i < 1:
                    block_i=1
                channel_score+=(block_i/block_num_now)*all_score[each_name_weight]
            for each_name_weight in name_s_list:
                all_score[each_name_weight]=channel_score


    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_name[name]=m.weight.data.shape[0]
    conv_name_list=list(conv_name.keys())
    layer_list = list(model.state_dict().keys())
    conv_idenx, conv_name1 = get_name_indenx(model)
    result = collections.OrderedDict()
    computed_name=[]

    #test ori
    test_loss_ori=0

    if args.cuda:
        model.cuda()
    model.eval()

    for data, target in test_loader:
        indx_target = target.clone()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        optimizer.zero_grad()
        #output = model(data)
        output = nn.parallel.data_parallel(model, data, device_ids=gpus)
        loss = F.cross_entropy(output, target)
        loss.backward()
        test_loss_ori += loss.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()
    del data
    del target
    test_loss_ori = test_loss_ori / len(test_loader)
    print(test_loss_ori)

    ori_shape=get_ori_shape(model)
    #each_layer analysis

    for each_name in conv_name_list:
        each_name_weight = each_name + '.weight'
        if each_name_weight in computed_name:
            continue
        num_f=conv_name[each_name]
        of=0
        prued_cha = []
        if channel_cor is not None:
            if each_name_weight in channel_cor:
                dylist=channel_cor[each_name_weight]
                if dylist in result:
                    prued_cha+=(result[dylist]+ori_shape[each_name_weight]//4).tolist()
        #each_name_weight=each_name+'.weight'
        if each_name_weight in all_stage_name:
            stage_i,block_i=get_i_i(each_name_weight,stage)
            computed_name=computed_name+stage[stage_i]
        else:
            computed_name.append(each_name_weight)
        step=num_f//2
        on_num = step
        while step>=1:
            on_num=int(on_num)
            if on_num<1 or on_num>=num_f:
                break
            test_loss1 = 0
            model=copy.deepcopy(model_ori)
            if args.cuda:
                model.cuda()
            model.eval()
            if each_name_weight in all_stage_name:
                for each_name_weight in stage[stage_i]:
                    dsa = all_score[each_name_weight].argsort()[:on_num]
                    if prued_cha:
                        dsa_list=dsa.tolist()
                        dsa_list+=prued_cha
                        dsa_list_1=[]
                        [dsa_list_1.append(oo) for oo in dsa_list if not oo in dsa_list_1]
                        dsa=np.array(dsa_list_1)
                    conv_idex = conv_name1.index(each_name_weight)
                    next_conv = conv_idenx[conv_idex + 1]
                    conv_idex = conv_idenx[conv_idex]
                    change_conv_bn(model, conv_idex, next_conv, layer_list, dsa)
            else:
                dsa = all_score[each_name_weight].argsort()[:on_num]
                conv_idex = conv_name1.index(each_name_weight)
                next_conv = conv_idenx[conv_idex + 1]
                conv_idex = conv_idenx[conv_idex]
                change_conv_bn(model, conv_idex, next_conv, layer_list, dsa)
            for data, target in test_loader:
                indx_target = target.clone()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target)
                optimizer.zero_grad()
                #output = model(data)
                output = nn.parallel.data_parallel(model, data, device_ids=gpus)
                loss = F.cross_entropy(output, target)
                loss.backward()
                test_loss1 += loss.item()
            test_loss1 = test_loss1 / len(test_loader)
            print(each_name,step,on_num,test_loss1,test_loss_ori)
            step /= 2
            if (step // 1) != step and step>1:
                step = step + 0.5
            if abs(test_loss1-test_loss_ori)>delta_loss:
                on_num-=step
            else:
                result[each_name_weight] = dsa
                on_num+=step
            del model
            del data
            del target
        if prued_cha and each_name_weight not in result:
            result[each_name_weight]=np.array(prued_cha)
    import pickle
    with open(os.path.join(args.result_dir,'result_id.pickle'),'wb') as json_file1:
        pickle.dump(result,json_file1)
    #test_loss=test_loss/len(test_loader)
    #acc = 100. * correct.type(torch.FloatTensor) / len(test_loader.dataset)
    #print("Accuracy:{:.4f}%".format(acc.numpy()))
    #return acc

def test_accori(model,test_loader,args):
    if args.cuda:
        model.cuda()
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        indx_target = target.clone()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target).item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()
    acc = 100. * correct.type(torch.FloatTensor) / len(test_loader.dataset)
    print("Accuracy:{:.4f}%".format(acc.numpy()))
    test_loss = test_loss / len(test_loader)
    print(test_loss)
    del data
    del target
    del model
    return acc


def get_name_indenx(model):
    conv_idenx = []
    conv_name=[]
    layer_list = list(model.state_dict().keys())
    for ii, name in enumerate(layer_list):
        if len(list(model.state_dict()[name].size()))== 4:
            conv_idenx.append(ii)
            conv_name.append(name)
        if len(list(model.state_dict()[name].size())) == 2:
            conv_idenx.append(ii)
            conv_name.append(name)
            break
    return conv_idenx,conv_name
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

def change_conv_bn(model,conv_idex,next_conv,layer_list,dsa):
    for kkk in range(conv_idex, next_conv):
        name = layer_list[kkk]
        if 'var' in name:
            model.state_dict()[name].data[dsa] = 1
        else:
            if model.state_dict()[name].data.size():
                model.state_dict()[name].data[dsa] = 0
def sensitivity_grad_resnet(model=None,config=None,args=None,test_loader=None):
    if config['stage'] is not None:
        stage=config['stage']
    score_all=get_score_resnet(model,test_loader,args,stage)
    # import pickle
    # with open('score_all.pickle','rb') as json_file:
    #     score_all=pickle.load(json_file)

    test_acc_resnet(model,test_loader,args,config,stage,all_score=score_all)
