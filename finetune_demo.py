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
import torch.optim as optim
import numpy as np
from IPython import embed
import collections
import pickle
import time
from thop import profile

from utee.get_config import get_yaml_data
from utee.tool import AverageMeter,ProgressMeter,accuracy
sys.path.append('..')
from network.cifar10.resnet_gal import resnet_56
from network.cifar10.resnet_gal_pruned import resnet_56_pruned
from method.pruning_resnet import pruning_resnet_grad

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
parser.add_argument('--model_dir', default="pre_models/cifar10/resnet56/resnet_56.pt",  help='the results dir of the sensetivity and the train step')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='finetune_result', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='100,200,300', help='decreasing strategy')


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
model = resnet_56()
model.load_state_dict(torch.load(args.model_dir, map_location='cuda:0')['state_dict'])
# create  the cifar10 data loader

train_loader = dataset.get10_mobile(batch_size=args.batch_size,data_root=args.data_dir,train=True, val=False, num_workers=1)
test_loader = dataset.get10_mobile(batch_size=args.batch_size,data_root=args.data_dir,train=False, val=True, num_workers=1)

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


# input=torch.randn(1,3,32,32)
# flops,params=profile(model,inputs=(input,))
# print(flops)
# print(params)
model=resnet_56_pruned(ma=pruned_num)
model=pruning_resnet_grad(model,config, result)

# input=torch.randn(1,3,32,32)
# flops,params=profile(model,inputs=(input,))
# print(flops)
# print(params)
print('train')
if args.cuda:
    model.cuda()
model.eval()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
print('decreasing_lr: ' + str(decreasing_lr))
best_acc=0
old_file=None
try:
    # ready to go
    for epoch in range(args.epochs):
        model.train()
        ppp=[x for x in model.parameters()]
        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        t_begin=time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output=nn.parallel.data_parallel(model,data,device_ids=gpus)
            loss = F.cross_entropy(output, target)
            loss.backward()

            optimizer.step()

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                # correct = pred.cpu().eq(indx_target).sum()
                correct = sum(np.array(pred.cpu())==np.array(indx_target.data.cpu()))
                acc = (correct * 1.0) / (1.0 * len(data))
                print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    loss.item(), acc, optimizer.param_groups[0]['lr']))

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time
        print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))
        misc.model_snapshot(model, os.path.join(args.logdir, 'latest.pth'))

        if epoch % args.test_interval == 0:
            model.eval()
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                indx_target = target.clone()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target)
                output=nn.parallel.data_parallel(model,data,device_ids=gpus)
                test_loss += F.cross_entropy(output, target).item()
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.cpu().eq(indx_target).sum()

            test_loss = test_loss / len(test_loader) # average over number of mini-batch
            acc = 100. * correct.type(torch.FloatTensor) / len(test_loader.dataset)
            print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
                test_loss, correct, len(test_loader.dataset), acc))
            if acc > best_acc:
                new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
                misc.model_snapshot(model, new_file, old_file=old_file, verbose=True)
                best_acc = acc
                old_file = new_file
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))

