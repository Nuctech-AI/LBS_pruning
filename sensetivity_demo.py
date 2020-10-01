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
from network.cifar10.densenet_gal import densenet_40
from network.cifar10.googlenet import googlenet
from network.cifar10.vgg import vgg_16_bn
import torchvision.models as models

from method.sensitivity_resnet import sensitivity_grad_resnet



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
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
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
#creat  the original  model
if  args.network =='resnet56':
    model = resnet_56()
    model.load_state_dict(torch.load(args.model_dir, map_location='cuda:0')['state_dict'])
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
if 'resnet' in args.network:
    sensitivity_grad_resnet(model=model, config=config, args=args, test_loader=test_loader)