import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from base import BaseDataLoader
from PIL import Image
from PIL import ImageFilter

class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size
    
    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets]) # Actually we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num # Ensures every instance has the chance to be visited in an epoch

class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels # Sampler needs to use targets
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label

# class iNaturalistDataLoader(DataLoader):
#     """
#     iNaturalist Data Loader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False, retain_epoch_size=True, 
#                 #  train_txt= './data_txt/iNaturalist_test/iNaturalist18_train.txt', 
#                 #  eval_txt= './data_txt/iNaturalist_test/iNaturalist18_val.txt'):
#         train_trsfm = transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
#         ])
#         test_trsfm = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
#         ])

#         if training:
#             dataset = LT_Dataset(data_dir, train_txt , train_trsfm)
#             val_dataset = LT_Dataset(data_dir, eval_txt, test_trsfm)
#         else: # test
#             dataset = LT_Dataset(data_dir, eval_txt, test_trsfm)
#             val_dataset = None

#         self.dataset = dataset
#         self.val_dataset = val_dataset

#         self.n_samples = len(self.dataset)

#         num_classes = 8142
#         # num_classes = 13

#         cls_num_list = [0] * num_classes
#         for label in dataset.targets:
#             cls_num_list[label] += 1

#         self.cls_num_list = cls_num_list

#         if balanced:
#             if training:
#                 buckets = [[] for _ in range(num_classes)]
#                 for idx, label in enumerate(dataset.targets):
#                     buckets[label].append(idx)
#                 sampler = BalancedSampler(buckets, retain_epoch_size)
#                 shuffle = False
#             else:
#                 print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
#         else:
#             sampler = None
        
#         self.shuffle = shuffle
#         self.init_kwargs = {
#             'batch_size': batch_size,
#             'shuffle': self.shuffle,
#             'num_workers': num_workers
#         }

#         super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler) # Note that sampler does not apply to validation set

#     def split_validation(self):
#         # return None
#         # If you want to validate:
#         return DataLoader(dataset=self.val_dataset, **self.init_kwargs)

class MCE_TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform1,base_transform2,base_transform3):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.base_transform3 = base_transform3

    def __call__(self, x):
        q = self.base_transform1(x)
        k = self.base_transform2(x)
        v = self.base_transform3(x)
        return [q, k ,v]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class MCE_iNaturalistDataLoader(DataLoader):
    """
    iNaturalist Data Loader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=4, training=True, balanced=False,
                 retain_epoch_size=True,
                 train_txt='./data_txt/iNaturalist/iNaturalist18_train.txt',
                 eval_txt='./data_txt/iNaturalist/iNaturalist18_val.txt'):
                #  train_txt='./data_txt/iNaturalist_test/iNaturalist18_train.txt',
                #  eval_txt='./data_txt/iNaturalist_test/iNaturalist18_val.txt'):
        train_trsfm1 = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
        ])

        train_trsfm2 = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
        ])

        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
        ])

        if training:
            dataset = LT_Dataset(data_dir, train_txt, MCE_TwoCropsTransform(train_trsfm1,train_trsfm2,train_trsfm2))
            val_dataset = LT_Dataset(data_dir, eval_txt, test_trsfm)
        else:  # test
            dataset = LT_Dataset(data_dir, eval_txt, test_trsfm)
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset

        self.n_samples = len(self.dataset)

        # num_classes = 8142
        num_classes = 13

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs,
                         sampler=sampler)  # Note that sampler does not apply to validation set

    def split_validation(self):
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)