import argparse
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'utee'))
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.cifar10 import dataset
import copy
from utee import misc
import torch.nn as nn
from IPython import embed
import collections
import pickle
from thop import profile

from utee.get_config import get_yaml_data
from utee.tool import AverageMeter,ProgressMeter,accuracy
sys.path.append('..')
from network.cifar10.resnet_gal import resnet_56,resnet_110
from network.cifar10.resnet_gal_pruned import resnet_56_pruned,resnet_110_pruned
from network.cifar10.densenet_gal import densenet_40
from network.cifar10.googlenet import googlenet
from network.cifar10.vgg import vgg_16_bn
import torchvision.models as models
from method.pruning_resnet import pruning_resnet_grad
from method.pruning_densenet import pruning_grad_densenet
from method.pruning_googlenet import pruning_googlenet_grad
from method.pruning_vgg import pruning_vgg_grad

parser = argparse.ArgumentParser(description='LBS_pruning Example')
parser.add_argument('--dataset', default='cifar10', help='cifar10|imagenet')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
parser.add_argument('--network', default='resnet56', help='the network type')
parser.add_argument('--config_dir', default='config/cifar10/config_resnet56.yaml', help='the network type')
parser.add_argument('--data_dir', default='/mnt/MountVolume3/chensitong/data', help='learning rate (default: 1e-3)')
parser.add_argument('--gpu', default='0', help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--result_dir', default="result/cifar10/resnet56/0019",  help='the results dir of the sensetivity and the train step')
parser.add_argument('--model_dir', default="result/cifar10/resnet56/0019/best-235.pth",  help='the results dir of the sensetivity and the train step')
args = parser.parse_args()
# select gpu
args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)
gpus=list(map(int,args.gpu))
# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#load_config
config=get_yaml_data(args.config_dir)


def load_model(model, model_path):
    state_dict_ = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = {}
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        # else:
        # ppp=10
        # print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)


#creat  the original  model
if  args.network =='resnet56':
    model = resnet_56()
elif  args.network =='resnet110':
    model=resnet_110()
elif args.network=='densenet40':
    model=densenet_40()
elif args.network=='googlenet':
    model=googlenet()
elif args.network=='vgg_bn':
    model=vgg_16_bn()
elif args.network=='resnet50':
    model=models.resnet50()
# create  the data loader
if args.dataset == 'cifar10':
    train_loader = dataset.get10_mobile(batch_size=args.batch_size,data_root=args.data_dir,train=True, val=False, num_workers=1)
    test_loader = dataset.get10_mobile(batch_size=args.batch_size,data_root=args.data_dir,train=False, val=True, num_workers=1)
elif args.dataset == 'imagenet':
    test_loader=dataset.get_imagenet_valdata(batch_size=args.batch_size,data_root=args.data_dir)
print('prune')
model_ori=copy.deepcopy(model)
ori_shape={}
for name, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        if m.weight is not None:
            ori_shape[name+'.weight'] = m.weight.data.shape[0]
with open(os.path.join(args.result_dir,'result_id.pickle'),'rb') as json_file1:
    result=pickle.load(json_file1)
stage=config['stage']
if stage is not None:
    def get_i_i(each_name_weight, stage):
        for i, stage_ea in enumerate(stage):
            for j, block_ea in enumerate(stage_ea):
                if block_ea == each_name_weight:
                    return i, j
    if config['channel_select']:
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                namex=name+'.weight'
                if namex not in result:
                    stage_i,_=get_i_i(namex,stage)
                    if stage[stage_i][-1] not in result:
                        if stage_i!=0 and result[stage[stage_i-1][-1]] is not None :
                            result[namex]=result[stage[stage_i-1][-1]]+ori_shape[namex]//4
                        else:
                            result[namex]=None
                    else:
                        result[namex]=result[stage[stage_i][-1]]
        pruned_num=[]
        name_bolck_end=config['name_bolck_end']
        for sd in name_bolck_end:
            pruned_num.append(result[sd])
    else:
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                namex = name + '.weight'
                if namex not in result:
                    stage_i, _ = get_i_i(namex, stage)
                    if stage[stage_i][-1] not in result:
                        result[namex] = None
                    else:
                        result[namex] = result[stage[stage_i][-1]]

if args.dataset=="cifar10":
    input=torch.randn(1,3,32,32)
    flops1,params1=profile(model,inputs=(input,))
    #print(flops1)
    #print(params1)
    if  args.network =='resnet56':
        model=resnet_56_pruned(ma=pruned_num)
        model=pruning_resnet_grad(model,config, result)
        load_model(model,args.model_dir)
    elif args.network =='resnet110':
        model=resnet_110_pruned(ma=pruned_num)
        model=pruning_resnet_grad(model,config, result)
        load_model(model,args.model_dir)
    elif args.network =='densenet40':
        model = pruning_grad_densenet(model, config, result)
        load_model(model, args.model_dir)
    elif args.network =='googlenet':
        model = pruning_googlenet_grad(model, config, result)
        load_model(model, args.model_dir)
    elif args.network =='vgg_bn':
        model=pruning_vgg_grad(model,config, result)
        load_model(model, args.model_dir)
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            print(name + '  ' + str(m.weight.data.shape[0]))
    input=torch.randn(1,3,32,32)
    flops,params=profile(model,inputs=(input,))
    #print(flops)
    #print(params)

    print("FLOPs of the original network: {:6.2f}".format(flops1))
    print("PARAMS of the original network: {:6.2f}".format(params1))
    print("FLOPs of the pruned network: {:6.2f}".format(flops))
    print("PARAMS of the pruned network: {:6.2f}".format(params))


    if args.cuda:
        model.cuda()
    print('eval')
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    model.cuda()
    with torch.no_grad():
        for index1111, (data, target) in enumerate(test_loader):
            #print(index1111)
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            #utput=nn.parallel.data_parallel(model,data,device_ids=args.gpu)
            output=nn.parallel.data_parallel(model,data,device_ids=gpus)
            #output = model(data)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))



elif args.dataset=="imagenet":

    model.cuda()
    input=torch.randn(1,3,224,224).cuda()
    flops,params=profile(model,inputs=(input,))
    print(flops)
    print(params)

    model=pruning_resnet_grad(model,config, result)
    load_model(model,args.model_dir)
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            print(name + '  ' + str(m.weight.data.shape[0]))
    input=torch.randn(1,3,224,224).cuda()
    flops,params=profile(model,inputs=(input,))
    print(flops)
    print(params)

    print('test')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    model.cuda()
    model.eval()
    with torch.no_grad():
        for index1111, (data, target) in enumerate(test_loader):
            #print(index1111)
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            #utput=nn.parallel.data_parallel(model,data,device_ids=args.gpu)
            output=nn.parallel.data_parallel(model,data,device_ids=gpus)
            #output = model(data)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))