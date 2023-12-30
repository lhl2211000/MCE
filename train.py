import argparse
import collections
import pprint
import time
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
import random
from PIL import ImageFilter
from data_loader.imbalance_cifar import IMBALANCECIFAR100


deterministic = False

if deterministic:
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)


# class GaussianBlur(object):
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

#     def __init__(self, sigma=[.1, 2.]):
#         self.sigma = sigma

#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x

# class TwoCropsTransform:
#     """Take two random crops of one image as the query and key."""

#     def __init__(self, base_transform):
#         self.base_transform = base_transform

#     def __call__(self, x):
#         q = self.base_transform(x)
#         k = self.base_transform(x)
#         return [q, k]

# class TestAgnosticImbalanceCIFAR100DataLoader(DataLoader):
#     """
#     Imbalance Cifar100 Data Loader
#     """

#     def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False,
#                  retain_epoch_size=True, imb_type='exp', imb_factor=0.01, test_imb_factor=0, reverse=False):
#         normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                          std=[0.2023, 0.1994, 0.2010])
#         train_trsfm = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(15),
#             transforms.ToTensor(),
#             normalize,
#         ])
#         test_trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             normalize,
#         ])
#         test_dataset = datasets.CIFAR100(data_dir, train=False, download=False, transform=test_trsfm)  # test set

#         if training:
#             dataset = IMBALANCECIFAR100(data_dir, train=True, download=False, transform=train_trsfm, imb_type=imb_type,imb_factor=imb_factor)
#             train_dataset = IMBALANCECIFAR100(data_dir, train=True, download=False,transform=TwoCropsTransform(train_trsfm), imb_type=imb_type,imb_factor=test_imb_factor, reverse=reverse)
            
#             val_dataset = test_dataset
#         else:  
#             dataset = IMBALANCECIFAR100(data_dir, train=True, download=False, transform=train_trsfm, imb_type=imb_type,imb_factor=test_imb_factor, reverse=reverse)
#             val_dataset = IMBALANCECIFAR100(data_dir, train=False, download=False, transform=test_trsfm,imb_type=imb_type, imb_factor=test_imb_factor, reverse=reverse)

#         self.dataset = dataset
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset

#         num_classes = len(np.unique(dataset.targets))
#         assert num_classes == 100

#         cls_num_list = [0] * num_classes
#         for label in dataset.targets:
#             cls_num_list[label] += 1

#         self.cls_num_list = cls_num_list

#         self.shuffle = shuffle
#         self.init_kwargs = {
#             'batch_size': batch_size,
#             'num_workers': num_workers
#         }

#         super().__init__(dataset=self.dataset, **self.init_kwargs)  # Note that sampler does not apply to validation set

#     def train_set(self):
#         return DataLoader(dataset=self.train_dataset, shuffle=True, **self.init_kwargs)

#     def test_set(self):
#         return DataLoader(dataset=self.val_dataset, shuffle=False, **self.init_kwargs)

def learing_rate_scheduler(optimizer, config):
    if "type" in config._config["lr_scheduler"]: 
        if config["lr_scheduler"]["type"] == "CustomLR": # linear learning rate decay
            lr_scheduler_args = config["lr_scheduler"]["args"]
            gamma = lr_scheduler_args["gamma"] if "gamma" in lr_scheduler_args else 0.1
            print("Scheduler step1, step2, warmup_epoch, gamma:", (lr_scheduler_args["step1"], lr_scheduler_args["step2"], lr_scheduler_args["warmup_epoch"], gamma))
            def lr_lambda(epoch):
                if epoch >= lr_scheduler_args["step2"]:
                    lr = gamma * gamma
                elif epoch >= lr_scheduler_args["step1"]:
                    lr = gamma
                else:
                    lr = 1

                """Warmup"""
                warmup_epoch = lr_scheduler_args["warmup_epoch"]
                if epoch < warmup_epoch:
                    lr = lr * float(1 + epoch) / warmup_epoch
                return lr
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)  # cosine learning rate decay
    else:
        lr_scheduler = None
    return lr_scheduler


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)#import data_loader.data_loaders as module_data
    valid_data_loader = data_loader.split_validation()
   
    model = config.init_obj('arch', module_arch)
    loss_class = getattr(module_loss, config["loss"]["type"])
    if hasattr(loss_class, "require_num_experts") and loss_class.require_num_experts:
        criterion = config.init_obj('loss', module_loss, cls_num_list=data_loader.cls_num_list, num_experts=config["arch"]["args"]["num_experts"])
    else:
        criterion = config.init_obj('loss', module_loss, cls_num_list=data_loader.cls_num_list)

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
   
    lr_scheduler = learing_rate_scheduler(optimizer, config)

  
    checkpoint_path = config.resume
    start_epoch = 1
    if checkpoint_path is not None:
        print(checkpoint_path)
        time.sleep(10)
        
        checkpoint = torch.load(checkpoint_path)
        
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint['epoch']

    trainer = Trainer(model, criterion, metrics, optimizer,config=config,data_loader= data_loader,
                      valid_data_loader=valid_data_loader,lr_scheduler=lr_scheduler,
                      cls_num_list=data_loader.cls_num_list,
                      start_epoch = start_epoch)

    trainer.train()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--load_crt', default=None, type=str, help='crt')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--step1'], type=int, target='lr_scheduler;args;step1'),
        CustomArgs(['--step2'], type=int, target='lr_scheduler;args;step2'),
        CustomArgs(['--warmup'], type=int, target='lr_scheduler;args;warmup_epoch'),
        CustomArgs(['--gamma'], type=float, target='lr_scheduler;args;gamma'),
        CustomArgs(['--save_period'], type=int, target='trainer;save_period'),
        CustomArgs(['--reduce_dimension'], type=int, target='arch;args;reduce_dimension'),
        CustomArgs(['--layer2_dimension'], type=int, target='arch;args;layer2_output_dim'),
        CustomArgs(['--layer3_dimension'], type=int, target='arch;args;layer3_output_dim'),
        CustomArgs(['--layer4_dimension'], type=int, target='arch;args;layer4_output_dim'),
        CustomArgs(['--num_experts'], type=int, target='arch;args;num_experts') 
    ]
    config = ConfigParser.from_args(args, options)
    pprint.pprint(config)
    main(config)
