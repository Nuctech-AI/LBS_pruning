# LBS_pruning

PyTorch implementation for LBS.


## Abstract

Filter pruning has drawn more attention since resource constrained platform requires more compact model for deployment. However, current pruning methods suﬀer either from the inferior performance of one-shot methods, or the expensive time cost of iterative training methods. In this paper, we propose a balanced ﬁlter pruning method for performance and pruning speed. Based on the ﬁlter importance criteria, our method is able to prune a layer with approximate layer-wise optimal pruning rate at given loss variation. The network is pruned in the layer-wise way without the time consuming prune-retrain iteration. If a pre-deﬁned pruning rate for entire network is given, we also introduce a method to ﬁnd the corresponding loss variation threshold with fast converging speed. Moreover, we propose the layer group pruning and channel selection mechanism for channel alignment in network with short connections. The proposed pruning method is widely applicable to common architectures and does not involve any additional training except the ﬁnal ﬁne-tuning. Comprehensive experiments show that our method outperforms many state-of-the-art approaches.


## Running Code

In this code, you can test the performance of our model on cifar10 or Imagenet dataset, and you can also complete the entire pruning process of resnet56 on cifar10. The code has been tested by Python 3.6, [Pytorch 1.4.0](https://pytorch.org/) and CUDA 10.0 on Ubuntu 16.04.


## Installation

Clone this repo, and we'll call the directory that you cloned as ${LBS_ROOT}
After you installed the python 3.6 and pytorch >=1.2.0, you can use the following command to install the environment of this code:
```shell
cd ${LBS_ROOT}
pip install -r requirements.txt
````


## Pretrained models and baseline model

The Pretrained models and baseline model can be downloaded here:[[Baidu],code:dee4](https://pan.baidu.com/s/1HV3LfwZCmHRJDJzxmChQQg)


### Run examples

We provide some samples to test the performance of our baseline model,them can be find in ${LBS_ROOT}/experiment

Before using the shell script, make sure that you have changed the `data_dir`, `result_dir`, 'model_dir', 'config_dir' to the place where each of them is located.

**For Evaluate**

You can run the test_demo_cifar10_densenet40.sh, test_demo_imagenet_resnet50.sh and so on in ${LBS_ROOT}/experiment


**For Analysising The Resnet56 On Cifar10**

You can run the sensetivity_demo_cifar10_resnet56.sh in ${LBS_ROOT}/experiment

if you want to get models with different pruning rates, you can change the value of dalta_loss in config file.

**For Fine-tune The Pruned Model for Resnet56 On Cifar10**

You can run the finetune_demo_cifar10_resnet56.sh in ${LBS_ROOT}/experiment


## Tips

If you find any problems, please feel free to contact to the authors (chensitong@nuctech.com or li.dong@nuctech.com).