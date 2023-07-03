import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, load_state_dict, rename_parallel_state_dict, autocast, use_fp16
import model.model as module_arch
import torch.nn as nn
import random
import math

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None,cls_num_list=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config

        # add_extra_info will return info about individual experts. This is crucial for individual loss. If this is false, we can only get a final mean logits.
        self.add_extra_info = config._config.get('add_extra_info', False)#self.add_extra_info = True
        print("self.add_extra_info",self.add_extra_info)

        self.data_loader = data_loader
        # self.data_loader2 = data_loader2
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        if use_fp16:
            self.logger.warn("FP16 is enabled. This option should be used with caution unless you make sure it's working and we do not provide guarantee.")
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None#self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        #######################有监督的对比损失######################
        self.criterion_scl=nn.CosineSimilarity(dim=0).to(self.device)
        self.criterion_scl_unsupervised = nn.CosineSimilarity(dim=2).to(self.device)
        self.cls_num_list = cls_num_list
        prior = np.array(cls_num_list) / np.sum(cls_num_list)  # P(Ci)
        self.prior = torch.tensor(prior).float().cuda()
        self.inverse_prior1=self.inverse_prior(self.prior)

        ##########################################################

    def inverse_prior(self, prior):
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        return inverse_prior

    def get_inverse_cls_num_list(self, cls_num_list):
        value, idx0 = torch.sort(torch.tensor(cls_num_list))
        _, idx1 = torch.sort(idx0)
        idx2 = torch.tensor(cls_num_list).shape[0]-1-idx1 # reverse the order
        inverse_cls_num_list = value.index_select(0,idx2)
        return inverse_cls_num_list

    """
    # scl for single contrastive model
    """
    def compute_supervised_contrastive_loss_sm(self, features, feat_stop_grad,view_features,view_feat_sg, target,k):  # features.shape=128*3*48  ,features_view,feat_view_stop_grad
        # 创建一个二维列表：cls_idx_list
        cls_idx_list = []
        for _ in range(len(self.cls_num_list)):  # 100是cifar-100的类别数量
            cls_idx_list.append([])

        current_batch_size = len(features)
        for j in range(current_batch_size):
            cls_idx_list[target[j].to(torch.long)].append(j)
        # for i in range(current_batch_size):
        #     features[i]=features[i]+torch.log(self.prior[target[i]]+1e-9)
        #     view_features[i] = view_features[i] + torch.log(self.prior[target[i]] + 1e-9)
        #     feat_stop_grad[i]=feat_stop_grad[i] + torch.log(self.prior[target[i]] + 1e-9)
        #     view_feat_sg[i]=view_feat_sg[i] + torch.log(self.prior[target[i]] + 1e-9)
        #首先计算无监督
        # loss_scl_unsupervised = - (self.criterion_scl_unsupervised(features, view_feat_sg).mean() + self.criterion_scl_unsupervised(view_features, feat_stop_grad).mean()) * 0.5

        #然后计算有监督对比损失
        loss_scl = 0.0
        for i in range(current_batch_size):
            current_cls_list = cls_idx_list[target[i].to(torch.long)]
            # e2=torch.log(self.prior[target[i]] + 1e-9)
            uscl =- (self.criterion_scl(features[i][0], view_feat_sg[i][0]).mean() + self.criterion_scl(view_features[i][0], feat_stop_grad[i][0]).mean()) * 0.5
            if len(current_cls_list) >= 2:  # 当前batch中有同类别的样本
                # select all idx for current_cls_list
                if len(current_cls_list) <= k + 1:  # 同类别下的样本数量≤k+1
                    for cu in current_cls_list:
                        if cu != i:
                            loss_scl = loss_scl - (self.criterion_scl(features[i][0], feat_stop_grad[cu][0]).mean() + self.criterion_scl(features[cu][0], feat_stop_grad[i][0]).mean()) * 0.5

                    loss_scl = loss_scl / (len(current_cls_list) - 1) +uscl#

                else:  # 同类别下的样本数量>k+1
                    positive_samples_list = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list.append(cc)
                        if len(positive_samples_list) == k:
                            break

                    for p in positive_samples_list:
                        loss_scl = loss_scl - (self.criterion_scl(features[i][0],feat_stop_grad[p][0]).mean() + self.criterion_scl(features[p][0], feat_stop_grad[i][0]).mean()) * 0.5

                    loss_scl = loss_scl / k +uscl#

        return 1 + loss_scl / current_batch_size   #+ loss_scl_unsupervised

    """
    # 每个专家分别随机采样，k=x(2),最终版的损失函数
    """
    def compute_supervised_contrastive_loss(self, features ,feat_stop_grad,view_features,view_feat_sg, target,k):  # features.shape=128*3*48  ,features_view,feat_view_stop_grad
        # 创建一个二维列表：cls_idx_list
        cls_idx_list = []
        for _ in range(len(self.cls_num_list)):  # 100是cifar-100的类别数量
            cls_idx_list.append([])

        current_batch_size = len(features)
        for j in range(current_batch_size):
            cls_idx_list[target[j].to(torch.long)].append(j)

        loss_scl = 0.0
        # 首先计算无监督对比损失
        loss_scl_unsupervised =  - ( self.criterion_scl_unsupervised(features,view_feat_sg).mean() +  self.criterion_scl_unsupervised(view_features,feat_stop_grad).mean()) * 0.5
        # loss_scl =1+loss_scl_unsupervised/current_batch_size

        # 然后计算监督对比损失
        for i in range(current_batch_size):
            current_cls_list = cls_idx_list[target[i].to(torch.long)]
            if len(current_cls_list) >= 2:  # 当前batch中有同类别的样本
                # select all idx for current_cls_list
                if len(current_cls_list) <= k + 1:  # 同类别下的样本数量≤k+1
                    for cu in current_cls_list:
                        if cu != i:
                            loss_scl = loss_scl - \
                                         (self.criterion_scl(features[i][0],feat_stop_grad[cu][0]).mean() + self.criterion_scl(features[cu][0], feat_stop_grad[i][0]).mean()) * 0.5 \
                                       - (self.criterion_scl(features[i][1],feat_stop_grad[cu][1]).mean() + self.criterion_scl(features[cu][1], feat_stop_grad[i][1]).mean()) * 0.5 \
                                       - (self.criterion_scl(features[i][2],feat_stop_grad[cu][2]).mean() + self.criterion_scl(features[cu][2], feat_stop_grad[i][2]).mean()) * 0.5

                    # loss_scl = loss_scl / (len(current_cls_list) - 1)

                else:  # 同类别下的样本数量>k+1
                    l1=0.0
                    positive_samples_list_1 = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list_1.append(cc)
                        if len(positive_samples_list_1) == k:
                            break

                    for p in positive_samples_list_1:
                        # loss_scl = loss_scl - (self.criterion_scl(features[i][0],feat_stop_grad[p][0]).mean() + self.criterion_scl(features[p][0], feat_stop_grad[i][0]).mean()) * 0.5
                        l1 = l1 - (self.criterion_scl(features[i][0],feat_stop_grad[p][0]).mean() + self.criterion_scl(features[p][0], feat_stop_grad[i][0]).mean()) * 0.5
                    # l1=l1/(k+2)
                    random.shuffle(current_cls_list)

                    l2=0.0
                    positive_samples_list_2 = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list_2.append(cc)
                        if len(positive_samples_list_2) == k:
                            break

                    for p in positive_samples_list_2:
                        # loss_scl = loss_scl - (self.criterion_scl(features[i][1],feat_stop_grad[p][1]).mean() + self.criterion_scl(features[p][1], feat_stop_grad[i][1]).mean()) * 0.5
                        l2 = l2 - (self.criterion_scl(features[i][1],feat_stop_grad[p][1]).mean() + self.criterion_scl(features[p][1], feat_stop_grad[i][1]).mean()) * 0.5
                    # l2 = l2 / (k + 1)
                    random.shuffle(current_cls_list)

                    l3=0.0
                    positive_samples_list_3 = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list_3.append(cc)
                        if len(positive_samples_list_3) == k:
                            break

                    for p in positive_samples_list_3:
                        # loss_scl = loss_scl - (self.criterion_scl(features[i][2],feat_stop_grad[p][2]).mean() + self.criterion_scl(features[p][2], feat_stop_grad[i][2]).mean()) * 0.5
                        l3 = l3 - (self.criterion_scl(features[i][2],feat_stop_grad[p][2]).mean() + self.criterion_scl(features[p][2], feat_stop_grad[i][2]).mean()) * 0.5
                    # l3=l3/k

                    # loss_scl = loss_scl / k
                    loss_scl = loss_scl+l1+l2+l3

        return 1 + loss_scl / current_batch_size   + loss_scl_unsupervised

    """
    # 每个专家分别随机采样，k=1,
    """
    def compute_supervised_contrastive_loss_k1(self, features ,feat_stop_grad,view_features,view_feat_sg, target,k):  # features.shape=128*3*48  ,features_view,feat_view_stop_grad
        # 创建一个二维列表：cls_idx_list
        cls_idx_list = []
        for _ in range(len(self.cls_num_list)):  # 100是cifar-100的类别数量
            cls_idx_list.append([])

        current_batch_size = len(features)
        for j in range(current_batch_size):
            cls_idx_list[target[j].to(torch.long)].append(j)

        loss_scl = 0.0
        loss_scl_unsupervised =0.0

        for i in range(current_batch_size):
            current_cls_list = cls_idx_list[target[i].to(torch.long)]
            if len(current_cls_list) >= 2:  # 当前batch中有同类别的样本，计算有监督对比损失
                # select all idx for current_cls_list
                if len(current_cls_list) <= k + 1:  # 同类别下的样本数量≤k+1
                    for cu in current_cls_list:
                        if cu != i:
                            loss_scl = loss_scl - \
                                         (self.criterion_scl(features[i][0],feat_stop_grad[cu][0]).mean() + self.criterion_scl(features[cu][0], feat_stop_grad[i][0]).mean()) * 0.5 \
                                       - (self.criterion_scl(features[i][1],feat_stop_grad[cu][1]).mean() + self.criterion_scl(features[cu][1], feat_stop_grad[i][1]).mean()) * 0.5 \
                                       - (self.criterion_scl(features[i][2],feat_stop_grad[cu][2]).mean() + self.criterion_scl(features[cu][2], feat_stop_grad[i][2]).mean()) * 0.5

                    # loss_scl = loss_scl / (len(current_cls_list) - 1)

                else:  # 同类别下的样本数量>k+1
                    l1=0.0
                    positive_samples_list_1 = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list_1.append(cc)
                        if len(positive_samples_list_1) == k:
                            break

                    for p in positive_samples_list_1:
                        # loss_scl = loss_scl - (self.criterion_scl(features[i][0],feat_stop_grad[p][0]).mean() + self.criterion_scl(features[p][0], feat_stop_grad[i][0]).mean()) * 0.5
                        l1 = l1 - (self.criterion_scl(features[i][0],feat_stop_grad[p][0]).mean() + self.criterion_scl(features[p][0], feat_stop_grad[i][0]).mean()) * 0.5
                    # l1=l1/(k+2)
                    random.shuffle(current_cls_list)

                    l2=0.0
                    positive_samples_list_2 = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list_2.append(cc)
                        if len(positive_samples_list_2) == k:
                            break

                    for p in positive_samples_list_2:
                        # loss_scl = loss_scl - (self.criterion_scl(features[i][1],feat_stop_grad[p][1]).mean() + self.criterion_scl(features[p][1], feat_stop_grad[i][1]).mean()) * 0.5
                        l2 = l2 - (self.criterion_scl(features[i][1],feat_stop_grad[p][1]).mean() + self.criterion_scl(features[p][1], feat_stop_grad[i][1]).mean()) * 0.5
                    # l2 = l2 / (k + 1)
                    random.shuffle(current_cls_list)

                    l3=0.0
                    positive_samples_list_3 = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list_3.append(cc)
                        if len(positive_samples_list_3) == k:
                            break

                    for p in positive_samples_list_3:
                        # loss_scl = loss_scl - (self.criterion_scl(features[i][2],feat_stop_grad[p][2]).mean() + self.criterion_scl(features[p][2], feat_stop_grad[i][2]).mean()) * 0.5
                        l3 = l3 - (self.criterion_scl(features[i][2],feat_stop_grad[p][2]).mean() + self.criterion_scl(features[p][2], feat_stop_grad[i][2]).mean()) * 0.5
                    # l3=l3/k

                    # loss_scl = loss_scl / k
                    loss_scl = loss_scl+l1+l2+l3
            elif len(current_cls_list)==1:#计算无监督对比损失
                loss_scl_unsupervised =loss_scl_unsupervised - \
                                       (self.criterion_scl(features[i][0],view_feat_sg[i][0]).mean() + self.criterion_scl(view_features[i][0], feat_stop_grad[i][0]).mean()) * 0.5 \
                                      -(self.criterion_scl(features[i][1],view_feat_sg[i][1]).mean() + self.criterion_scl(view_features[i][1], feat_stop_grad[i][1]).mean()) * 0.5 \
                                      -(self.criterion_scl(features[i][2],view_feat_sg[i][2]).mean() + self.criterion_scl(view_features[i][2], feat_stop_grad[i][2]).mean()) * 0.5
        return 1 + (loss_scl+ loss_scl_unsupervised) / current_batch_size

    """
    # balanced supervised contrastive loss
    """

    def compute_bal_supervised_contrastive_loss(self, features, feat_stop_grad, view_features, view_feat_sg, target,k):  # features.shape=128*3*48  ,features_view,feat_view_stop_grad
        # 创建一个二维列表：cls_idx_list
        cls_idx_list = []
        for _ in range(len(self.cls_num_list)):  # 100是cifar-100的类别数量
            cls_idx_list.append([])

        current_batch_size = len(features)
        for j in range(current_batch_size):
            cls_idx_list[target[j].to(torch.long)].append(j)

        loss_scl = 0.0
        # 首先计算无监督对比损失
        # loss_scl_unsupervised = - (self.criterion_scl_unsupervised(features, view_feat_sg).mean() + self.criterion_scl_unsupervised(view_features, feat_stop_grad).mean()) * 0.5
        # loss_scl =1+loss_scl_unsupervised/current_batch_size
        loss_scl_unsupervised= 0.0
        # 然后计算监督对比损失
        for i in range(current_batch_size):
            e2 = torch.log(self.prior[target[i]]+1e-9)
            e3 = torch.log(self.prior[target[i]]+1e-9)-2*torch.log(self.inverse_prior1[target[i]]+1e-9)
            u1 = (self.criterion_scl(features[i][0], view_feat_sg[i][0]).mean() + self.criterion_scl(view_features[i][0],feat_stop_grad[i][0]).mean()) * 0.5
            u2 = (self.criterion_scl(features[i][1], view_feat_sg[i][1]).mean() + self.criterion_scl(view_features[i][1],feat_stop_grad[i][1]).mean()) * 0.5-e2
            u3 = (self.criterion_scl(features[i][2], view_feat_sg[i][2]).mean() + self.criterion_scl(view_features[i][2],feat_stop_grad[i][2]).mean()) * 0.5-e3
            loss_scl_unsupervised =loss_scl_unsupervised  -u1-u2-u3
            current_cls_list = cls_idx_list[target[i].to(torch.long)]
            if len(current_cls_list) >= 2:  # 当前batch中有同类别的样本
                # select all idx for current_cls_list
                if len(current_cls_list) <= k + 1:  # 同类别下的样本数量≤k+1
                    for cu in current_cls_list:
                        if cu != i:
                            s1=(self.criterion_scl(features[i][0],feat_stop_grad[cu][0]).mean() + self.criterion_scl(features[cu][0], feat_stop_grad[i][0]).mean()) * 0.5
                            s2=(self.criterion_scl(features[i][1],feat_stop_grad[cu][1]).mean() + self.criterion_scl(features[cu][1], feat_stop_grad[i][1]).mean()) * 0.5-e2
                            s3=(self.criterion_scl(features[i][2],feat_stop_grad[cu][2]).mean() + self.criterion_scl(features[cu][2], feat_stop_grad[i][2]).mean()) * 0.5-e3
                            loss_scl = loss_scl -s1-s2-s3

                    # loss_scl = loss_scl / (len(current_cls_list) - 1)

                else:  # 同类别下的样本数量>k+1
                    l1 = 0.0
                    positive_samples_list_1 = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list_1.append(cc)
                        if len(positive_samples_list_1) == k:
                            break

                    for p in positive_samples_list_1:
                        # loss_scl = loss_scl - (self.criterion_scl(features[i][0],feat_stop_grad[p][0]).mean() + self.criterion_scl(features[p][0], feat_stop_grad[i][0]).mean()) * 0.5
                        l1 = l1 - (self.criterion_scl(features[i][0], feat_stop_grad[p][0]).mean() + self.criterion_scl(features[p][0], feat_stop_grad[i][0]).mean()) * 0.5
                    # l1=l1/(k+2)
                    random.shuffle(current_cls_list)

                    l2 = 0.0
                    positive_samples_list_2 = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list_2.append(cc)
                        if len(positive_samples_list_2) == k:
                            break

                    for p in positive_samples_list_2:
                        # loss_scl = loss_scl - (self.criterion_scl(features[i][1],feat_stop_grad[p][1]).mean() + self.criterion_scl(features[p][1], feat_stop_grad[i][1]).mean()) * 0.5
                        l2 = l2 - (self.criterion_scl(features[i][1], feat_stop_grad[p][1]).mean() + self.criterion_scl(features[p][1], feat_stop_grad[i][1]).mean()) * 0.5-e2
                    # l2 = l2 / (k + 1)
                    random.shuffle(current_cls_list)

                    l3 = 0.0
                    positive_samples_list_3 = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list_3.append(cc)
                        if len(positive_samples_list_3) == k:
                            break

                    for p in positive_samples_list_3:
                        # loss_scl = loss_scl - (self.criterion_scl(features[i][2],feat_stop_grad[p][2]).mean() + self.criterion_scl(features[p][2], feat_stop_grad[i][2]).mean()) * 0.5
                        l3 = l3 - (self.criterion_scl(features[i][2], feat_stop_grad[p][2]).mean() + self.criterion_scl(features[p][2], feat_stop_grad[i][2]).mean()) * 0.5-e3
                    # l3=l3/k

                    # loss_scl = loss_scl / k
                    loss_scl = loss_scl + l1 + l2 + l3

        return ( loss_scl / current_batch_size + loss_scl_unsupervised   / current_batch_size)/2

    """
    # selfsupcon
    """
    def compute_self_supervised_contrastive_loss(self, features, feat_stop_grad, views_features, views_feat_stop_grad):  # features.shape=128*3*48  ,features_view,feat_view_stop_grad
        loss_scl_unsupervised =  - ( self.criterion_scl_unsupervised(features,views_feat_stop_grad).mean() +  self.criterion_scl_unsupervised(views_features,feat_stop_grad).mean()) * 0.5

        return  loss_scl_unsupervised

    """
    # two-stage training:SelfSupCon+SupCon
    """
    def two_stage_compute_supervised_contrastive_loss(self, features, feat_stop_grad, target,k):  # features.shape=128*3*48  ,features_view,feat_view_stop_grad
        # 创建一个二维列表：cls_idx_list
        cls_idx_list = []
        for _ in range(len(self.cls_num_list)):  # 100是cifar-100的类别数量
            cls_idx_list.append([])

        current_batch_size = len(features)
        for j in range(current_batch_size):
            cls_idx_list[target[j].to(torch.long)].append(j)

        loss_scl = 0.0
        # 首先计算无监督对比损失
        # loss_scl_unsupervised =  - ( self.criterion_scl_unsupervised(features,feat_view_stop_grad).mean() +  self.criterion_scl_unsupervised(features_view,feat_stop_grad).mean()) * 0.5
        # loss_scl =1 + loss_scl/current_batch_size

        # 然后计算监督对比损失
        for i in range(current_batch_size):
            current_cls_list = cls_idx_list[target[i].to(torch.long)]
            if len(current_cls_list) >= 2:  # 当前batch中有同类别的样本
                ####################################################################
                # e2 = torch.log(self.prior[target[i]]+1e-9)
                # e3 = torch.log(self.prior[target[i]]+1e-9)-2*torch.log(self.inverse_prior1[target[i]]+1e-9)
                # e2 = self.prior[target[i]] + 1e-9
                # e3 = (self.prior[target[i]] + 1e-9)/(self.inverse_prior1[target[i]] + 1e-9)*(self.inverse_prior1[target[i]] + 1e-9)
                ####################################################################
                # select all idx for current_cls_list
                if len(current_cls_list) <= k + 1:  # 同类别下的样本数量≤k+1
                    for cu in current_cls_list:
                        if cu != i:
                            loss_scl = loss_scl - \
                                         (self.criterion_scl(features[i][0],feat_stop_grad[cu][0]).mean() + self.criterion_scl(features[cu][0], feat_stop_grad[i][0]).mean()) * 0.5 \
                                       - (self.criterion_scl(features[i][1],feat_stop_grad[cu][1]).mean() + self.criterion_scl(features[cu][1], feat_stop_grad[i][1]).mean()) * 0.5 \
                                       - (self.criterion_scl(features[i][2],feat_stop_grad[cu][2]).mean() + self.criterion_scl(features[cu][2], feat_stop_grad[i][2]).mean()) * 0.5

                    # loss_scl = loss_scl / (len(current_cls_list) - 1)

                else:  # 同类别下的样本数量>k+1
                    l1=0.0
                    positive_samples_list_1 = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list_1.append(cc)
                        if len(positive_samples_list_1) == k:
                            break

                    for p in positive_samples_list_1:
                        # loss_scl = loss_scl - (self.criterion_scl(features[i][0],feat_stop_grad[p][0]).mean() + self.criterion_scl(features[p][0], feat_stop_grad[i][0]).mean()) * 0.5
                        l1 = l1 - (self.criterion_scl(features[i][0],feat_stop_grad[p][0]).mean() + self.criterion_scl(features[p][0], feat_stop_grad[i][0]).mean()) * 0.5
                    # l1=l1/(k+2)
                    random.shuffle(current_cls_list)

                    l2=0.0
                    positive_samples_list_2 = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list_2.append(cc)
                        if len(positive_samples_list_2) == k:
                            break

                    for p in positive_samples_list_2:
                        # loss_scl = loss_scl - (self.criterion_scl(features[i][1],feat_stop_grad[p][1]).mean() + self.criterion_scl(features[p][1], feat_stop_grad[i][1]).mean()) * 0.5
                        l2 = l2 - (self.criterion_scl(features[i][1],feat_stop_grad[p][1]).mean() + self.criterion_scl(features[p][1], feat_stop_grad[i][1]).mean()) * 0.5
                    # l2 = l2 / (k + 1)
                    random.shuffle(current_cls_list)

                    l3=0.0
                    positive_samples_list_3 = []
                    for cc in current_cls_list:
                        if cc != i:
                            positive_samples_list_3.append(cc)
                        if len(positive_samples_list_3) == k:
                            break

                    for p in positive_samples_list_3:
                        # loss_scl = loss_scl - (self.criterion_scl(features[i][2],feat_stop_grad[p][2]).mean() + self.criterion_scl(features[p][2], feat_stop_grad[i][2]).mean()) * 0.5
                        l3 = l3 - (self.criterion_scl(features[i][2],feat_stop_grad[p][2]).mean() + self.criterion_scl(features[p][2], feat_stop_grad[i][2]).mean()) * 0.5
                    # l3=l3/k

                    # loss_scl = loss_scl / k
                    loss_scl = loss_scl+l1+l2+l3

        return 1 + loss_scl / current_batch_size  # + loss_scl_unsupervised
    """
    #对称版特征损失,每个专家统一对称
    """
    # def compute_supervised_contrastive_loss(self, features, feat_stop_grad, target,k):  # features.shape=128*3*48  ,features_view,feat_view_stop_grad
    #     # 创建一个二维列表：cls_idx_list
    #     cls_idx_list = []
    #     for _ in range(len(self.cls_num_list)):  # 100是cifar-100的类别数量
    #         cls_idx_list.append([])
    #
    #     current_batch_size = len(features)
    #     for j in range(current_batch_size):
    #         cls_idx_list[target[j].to(torch.long)].append(j)
    #     # cls_idx_list1 = cls_idx_list.copy()
    #     # cls_idx_list2 = cls_idx_list.copy()
    #     # cls_idx_list3 = cls_idx_list.copy()
    #     loss_scl = 0.0
    #
    #     sample_idx_list=[i for i in range(current_batch_size) ]
    #
    #     for i in sample_idx_list:
    #         current_cls_list = cls_idx_list[target[i].to(torch.long)]
    #         # current_cls_list1 = cls_idx_list1[target[i].to(torch.long)]#专家1的anchor的正样本列表
    #         # current_cls_list2 = cls_idx_list2[target[i].to(torch.long)]#专家2的anchor的正样本列表
    #         # current_cls_list3 = cls_idx_list3[target[i].to(torch.long)]#专家3的anchor的正样本列表
    #         if len(current_cls_list) >= 2:  # 当前batch中有同类别的样本
    #             for cu in current_cls_list:
    #                 if cu != i:
    #                     loss_scl = loss_scl \
    #                                -(self.criterion_scl(features[i][0],feat_stop_grad[cu][0]).mean() + self.criterion_scl(features[cu][0], feat_stop_grad[i][0]).mean()) * 0.5\
    #                                -(self.criterion_scl(features[i][1],feat_stop_grad[cu][1]).mean() + self.criterion_scl(features[cu][1], feat_stop_grad[i][1]).mean()) * 0.5\
    #                                -(self.criterion_scl(features[i][2],feat_stop_grad[cu][2]).mean() + self.criterion_scl(features[cu][2], feat_stop_grad[i][2]).mean()) * 0.5
    #                     cls_idx_list[target[i]].remove(cu)
    #                     cls_idx_list[target[i]].remove(i)
    #                     sample_idx_list.remove(cu)
    #                     # sample_idx_list.remove(i)
    #                     break
    #         # if len(current_cls_list1) >= 2:  # 当前batch中有同类别的样本
    #         #     for cu in current_cls_list1:
    #         #         if cu != i:
    #         #             loss_scl = loss_scl - (self.criterion_scl(features[i][0],feat_stop_grad[cu][0]).mean() + self.criterion_scl(features[cu][0], feat_stop_grad[i][0]).mean()) * 0.5
    #         #             cls_idx_list1[target[i]].remove(cu)
    #         #             cls_idx_list1[target[i]].remove(i)
    #         #             break
    #         # if len(current_cls_list2) >= 2:  # 当前batch中有同类别的样本
    #         #     for cu in current_cls_list2:
    #         #         if cu != i:
    #         #             loss_scl = loss_scl - (self.criterion_scl(features[i][0], feat_stop_grad[cu][0]).mean() + self.criterion_scl(features[cu][0], feat_stop_grad[i][0]).mean()) * 0.5
    #         #             cls_idx_list2[target[i]].remove(cu)
    #         #             cls_idx_list2[target[i]].remove(i)
    #         #             break
    #         # if len(current_cls_list3) >= 2:  # 当前batch中有同类别的样本
    #         #     for cu in current_cls_list3:
    #         #         if cu != i:
    #         #             loss_scl = loss_scl - (self.criterion_scl(features[i][0], feat_stop_grad[cu][0]).mean() + self.criterion_scl(features[cu][0], feat_stop_grad[i][0]).mean()) * 0.5
    #         #             cls_idx_list3[target[i]].remove(cu)
    #         #             cls_idx_list3[target[i]].remove(i)
    #         #             break
    #
    #     return 1 + loss_scl / current_batch_size

    #每个专家分别随机采样，scl=-logexp版
    # def compute_supervised_contrastive_loss(self, features, feat_stop_grad, target,k):  # features.shape=128*3*48  ,features_view,feat_view_stop_grad
    #     # 创建一个二维列表：cls_idx_list
    #     cls_idx_list = []
    #     for _ in range(len(self.cls_num_list)):  # 100是cifar-100的类别数量
    #         cls_idx_list.append([])
    #
    #     current_batch_size = len(features)
    #     for j in range(current_batch_size):
    #         cls_idx_list[target[j].to(torch.long)].append(j)
    #
    #     loss_scl = 0.0
    #     # 首先计算无监督对比损失
    #     # loss_scl_unsupervised =  - ( self.criterion_scl_unsupervised(features,feat_view_stop_grad).mean() +  self.criterion_scl_unsupervised(features_view,feat_stop_grad).mean()) * 0.5
    #     # loss_scl =1 + loss_scl/current_batch_size
    #
    #     # 然后计算监督对比损失
    #     for i in range(current_batch_size):
    #         current_cls_list = cls_idx_list[target[i].to(torch.long)]
    #         if len(current_cls_list) >= 2:  # 当前batch中有同类别的样本
    #             ####################################################################
    #             # e2 = torch.log(self.prior[target[i]]+1e-9)
    #             # e3 = torch.log(self.prior[target[i]]+1e-9)-2*torch.log(self.inverse_prior1[target[i]]+1e-9)
    #             ####################################################################
    #             temp=0.1
    #             # select all idx for current_cls_list
    #             if len(current_cls_list) <= k + 1:  # 同类别下的样本数量≤k+1
    #                 for cu in current_cls_list:
    #                     if cu != i:
    #                         numerator1 = torch.exp((self.criterion_scl(features[i][0],feat_stop_grad[cu][0]).mean() + self.criterion_scl(features[cu][0], feat_stop_grad[i][0]).mean()) * 0.5/temp)
    #                         numerator2 = torch.exp((self.criterion_scl(features[i][1],feat_stop_grad[cu][1]).mean() + self.criterion_scl(features[cu][1], feat_stop_grad[i][1]).mean()) * 0.5/temp)
    #                         numerator3 = torch.exp((self.criterion_scl(features[i][2],feat_stop_grad[cu][2]).mean() + self.criterion_scl(features[cu][2], feat_stop_grad[i][2]).mean()) * 0.5/temp)
    #                         denominator1 = 0.0
    #                         denominator2 = 0.0
    #                         denominator3 = 0.0
    #                         for neg_list in cls_idx_list:
    #                             if len(neg_list)>=1 :
    #                                 random_neg_ind1=random.randint(0, len(neg_list) - 1)
    #                                 neg1=neg_list[random_neg_ind1]#负样本当前batch下的索引
    #                                 if neg1!=i:
    #                                     denominator1=denominator1+torch.exp((self.criterion_scl(features[i][0],feat_stop_grad[neg1][0]).mean() + self.criterion_scl(features[neg1][0], feat_stop_grad[i][0]).mean()) * 0.5/temp)
    #
    #                                 random_neg_ind2 = random.randint(0, len(neg_list) - 1)
    #                                 neg2 = neg_list[random_neg_ind2]  # 负样本当前batch下的索引
    #                                 if neg2 != i:
    #                                     denominator2 = denominator2 + torch.exp((self.criterion_scl(features[i][1],feat_stop_grad[neg2][1]).mean() + self.criterion_scl(features[neg2][1], feat_stop_grad[i][1]).mean()) * 0.5/temp)
    #
    #                                 random_neg_ind3 = random.randint(0, len(neg_list) - 1)
    #                                 neg3 = neg_list[random_neg_ind3]  # 负样本当前batch下的索引
    #                                 if neg3 != i:
    #                                     denominator3 = denominator3 + torch.exp((self.criterion_scl(features[i][2],feat_stop_grad[neg3][2]).mean() + self.criterion_scl(features[neg3][2], feat_stop_grad[i][2]).mean()) * 0.5/temp)
    #                         loss_scl = loss_scl - torch.log(numerator1/denominator1) - torch.log(numerator2/denominator2) - torch.log(numerator3/denominator3)
    #
    #                 loss_scl = loss_scl / (len(current_cls_list) - 1)
    #
    #             else:  # 同类别下的样本数量>k+1
    #                 positive_samples_list_1 = []
    #                 for cc in current_cls_list:
    #                     if cc != i:
    #                         positive_samples_list_1.append(cc)
    #                     if len(positive_samples_list_1) == k:
    #                         break
    #
    #                 l1=0.0
    #                 for p in positive_samples_list_1:
    #                     numerator1 = torch.exp((self.criterion_scl(features[i][0],feat_stop_grad[p][0]).mean() + self.criterion_scl(features[p][0], feat_stop_grad[i][0]).mean()) * 0.5/temp)
    #                     denominator1 = 0.0
    #                     for neg_list in cls_idx_list:
    #                         if len(neg_list) >= 1:
    #                             random_neg_ind1 = random.randint(0, len(neg_list) - 1)
    #                             neg1 = neg_list[random_neg_ind1]  # 负样本当前batch下的索引
    #                             if neg1!=i:
    #                                 denominator1 = denominator1 + torch.exp((self.criterion_scl(features[i][0],feat_stop_grad[neg1][0]).mean() + self.criterion_scl(features[neg1][0], feat_stop_grad[i][0]).mean()) * 0.5 / temp)
    #                     l1= l1+torch.log(numerator1/denominator1)
    #
    #                 random.shuffle(current_cls_list)
    #
    #                 positive_samples_list_2 = []
    #                 for cc in current_cls_list:
    #                     if cc != i:
    #                         positive_samples_list_2.append(cc)
    #                     if len(positive_samples_list_2) == k:
    #                         break
    #                 l2=0.0
    #                 for p in positive_samples_list_2:
    #                     numerator2 = torch.exp((self.criterion_scl(features[i][1],feat_stop_grad[p][1]).mean() + self.criterion_scl(features[p][1], feat_stop_grad[i][1]).mean()) * 0.5/temp)
    #                     denominator2 = 0.0
    #                     for neg_list in cls_idx_list:
    #                         if len(neg_list) >= 1:
    #                             random_neg_ind2 = random.randint(0, len(neg_list) - 1)
    #                             neg2 = neg_list[random_neg_ind2]  # 负样本当前batch下的索引
    #                             if neg2 != i:
    #                                 denominator2 = denominator2 + torch.exp((self.criterion_scl(features[i][1],feat_stop_grad[neg2][1]).mean() + self.criterion_scl(features[neg2][1], feat_stop_grad[i][1]).mean()) * 0.5 / temp)
    #                     l2 = l2 + torch.log(numerator2 / denominator2)
    #
    #                 random.shuffle(current_cls_list)
    #
    #                 positive_samples_list_3 = []
    #                 for cc in current_cls_list:
    #                     if cc != i:
    #                         positive_samples_list_3.append(cc)
    #                     if len(positive_samples_list_3) == k:
    #                         break
    #                 l3=0.0
    #                 for p in positive_samples_list_3:
    #                     numerator3 = torch.exp((self.criterion_scl(features[i][0],feat_stop_grad[p][0]).mean() + self.criterion_scl(features[p][0], feat_stop_grad[i][0]).mean()) * 0.5 / temp)
    #                     denominator3 = 0.0
    #                     for neg_list in cls_idx_list:
    #                         if len(neg_list) >= 1:
    #                             random_neg_ind3 = random.randint(0, len(neg_list) - 1)
    #                             neg3 = neg_list[random_neg_ind3]  # 负样本当前batch下的索引
    #                             if neg3 != i:
    #                                 denominator3 = denominator3 + torch.exp((self.criterion_scl(features[i][0],feat_stop_grad[neg3][0]).mean() + self.criterion_scl(features[neg3][0], feat_stop_grad[i][0]).mean()) * 0.5 / temp)
    #                     l3 = l3 + torch.log(numerator3 / denominator3)
    #
    #                 loss_scl = (loss_scl-l1-l2-l3) / k
    #
    #     return 1 + loss_scl / current_batch_size  # + loss_scl_unsupervised


    #对于每一个类别，有几个样本用几个样本
    # def compute_supervised_contrastive_loss(self, features, feat_stop_grad, target,k):  # features.shape=128*3*48  ,features_view,feat_view_stop_grad
    #     # 创建一个二维列表：cls_idx_list
    #     cls_idx_list = []
    #     for _ in range(len(self.cls_num_list)):  # 100是cifar-100的类别数量
    #         cls_idx_list.append([])
    #
    #     current_batch_size = len(features)
    #     for j in range(current_batch_size):
    #         cls_idx_list[target[j].to(torch.long)].append(j)
    #
    #     loss_scl = 0.0
    #
    #     for i in range(current_batch_size):#外层循环控制batch中样本的迭代
    #         current_cls_list = cls_idx_list[target[i].to(torch.long)]
    #         # i1=ramdomly select k idx for current_cls_list
    #         if len(current_cls_list) >= 2:  # 同类别下的样本数量大于等于2
    #             for cu in current_cls_list:#内层循环控制同类别下其他样本的迭代
    #                 if cu != i:  # / self.cls_num_list[target[i]]*inverse_cls_num_list[target[i]]
    #                     loss_scl = loss_scl \
    #                                - (self.criterion_scl(features[i][0],feat_stop_grad[cu][0]).mean() + self.criterion_scl(features[cu][0], feat_stop_grad[i][0]).mean()) * 0.5 \
    #                                - (self.criterion_scl(features[i][1],feat_stop_grad[cu][1]).mean() + self.criterion_scl(features[cu][1], feat_stop_grad[i][1]).mean()) * 0.5 \
    #                                - (self.criterion_scl(features[i][2],feat_stop_grad[cu][2]).mean() + self.criterion_scl(features[cu][2], feat_stop_grad[i][2]).mean()) * 0.5
    #
    #             loss_scl = loss_scl / (len(current_cls_list) - 1) / 3  # 对于一个样本而言，应该除以同类别下的样本数量，还应该除以专家数量(3)
    #
    #     return 1 + loss_scl / current_batch_size

    def adjust_learning_rate_simsiam(self,optimizer, init_lr, epoch, total_epochs):
        #计算当前学习率
        step1=160
        step2=180
        warmup_epoch=5
        gamma=0.1
        print("Scheduler step1, step2, warmup_epoch, gamma:",( step1,  step2, warmup_epoch, gamma))

        if epoch >= 181:
            cur_lr = gamma * gamma*0.1
        elif epoch >= 161:
            cur_lr = gamma*0.1
        else:
            cur_lr = 0.1

        """Warmup"""
        warmup_epoch = warmup_epoch
        if epoch < warmup_epoch:
            cur_lr = cur_lr * float(epoch) / warmup_epoch

        #更新学习率
        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        #####   依然采用TADE的学习率模式，仅仅是让predictor的学习率固定为常数   #####
        # self.adjust_learning_rate_simsiam(self.optimizer, 0.025, epoch, 200)
        ####################################################################
        self.model.train()#resnet32开启训练模式
        self.real_model._hook_before_iter()
        self.train_metrics.reset()

        if hasattr(self.criterion, "_hook_before_epoch"):
            self.criterion._hook_before_epoch(epoch)

        ########   第1个epoch初始化特征中心,第161个epoch将特征中心应用于scl   ########
        # compute_feat_epoch=1
        # if epoch>=compute_feat_epoch:
        #     feat_dim = 1024
        #     # 专家特征队列
        #     self.all_features_stop_grad = torch.empty((0, feat_dim)).cuda()#单模型的第二个维度是1###3,
        #     # 样本标签队列
        #     self.all_targets = torch.empty(0, dtype=torch.long).cuda()
        ######################################################################

        # if epoch<=200:
        #     data_loader = self.data_loader2
        # else:
        #     data_loader = self.data_loader
        for batch_idx, data in enumerate(self.data_loader):
        # for batch_idx, data in enumerate(data_loader):
            data, target = data
            # data, target = data.to(self.device), target.to(self.device)
            data[0] = data[0].to(self.device)
            data[1] = data[1].to(self.device)
            data[2] = data[2].to(self.device)
            target=target.to(self.device)

            self.optimizer.zero_grad()

            with autocast():#这一行代码是什么意思
                if self.real_model.requires_target:#未执行
                    output = self.model(data, target=target)

                    output, loss = output   
                else:
                    extra_info = {}
                    # output = self.model(data)#output=3个专家的logit的平均值,特征,三个专家的logit
                    output = self.model(data[0],data[1],data[2])
                    # output = self.model(data[0], view=None,view1=None)


                    ##################################    用于计算特征中心    ############################################
                    # if epoch >=compute_feat_epoch:
                    #     #将专家习得的特征添加至专家1特征队列
                    #     self.all_features_stop_grad = torch.cat((self.all_features_stop_grad, output["feat_stop_grad"][:,0,:]))
                    #    # 将标签添加至标签队列
                    #     self.all_targets = torch.cat((self.all_targets, target))
                    ###################################################################################################


                    if self.add_extra_info:#执行
                        if isinstance(output, dict):#执行
                            logits = output["logits"]
                            # logits = (output["logits"] + output_view["logits"]) * 0.5
                            extra_info.update({
                                "logits": logits.transpose(0, 1)
                            })
                        else:#未执行
                            extra_info.update({
                                "logits": self.real_model.backbone.logits
                            })

                    if isinstance(output, dict):#执行,网络的输出output就是一个dict(字典)
                        output_logits = output["output"]#取3个专家的logit平均值
                        # output_logits = (output["output"] + output_view["output"]) * 0.5  # 取3个专家的两个视图的logit平均值


                    if self.add_extra_info:#执行，计算损失
                        ############################################################################################
                        # loss_cls = self.criterion(output_logits=output_logits, target=target, extra_info=extra_info)
                        # k=1#同类别下的样本数量
                        # loss_scl = self.compute_supervised_contrastive_loss(output["feat"],output["feat_stop_grad"],target,k)#,output_view["feat"],output_view["feat_stop_grad"],epoch,compute_feat_epoch
                        # # alpha =1-(epoch/200)*(epoch/200)
                        # alpha = 0.5
                        # # if epoch>=61:
                        # #     alpha=0.0
                        # loss_all=alpha*loss_cls+(1-alpha)*loss_scl
                        # # loss_all = alpha * loss_scl +  loss_cls
                        # if epoch<=200:
                        k=1
                        loss_scl = self.compute_supervised_contrastive_loss(output["feat"],output["feat_stop_grad"],output["view_feat"],output["view_feat_stop_grad"],target, k)
                        # loss_scl = self.compute_self_supervised_contrastive_loss(output["feat"],output["feat_stop_grad"],output["view_feat"],output["view_feat_stop_grad"])
                        loss_cls = self.criterion(output_logits=output_logits, target=target, extra_info=extra_info)
                        loss_all=0.5*loss_scl+0.5*loss_cls
                        # loss_all = loss_cls
                        # else:
                        #     loss_cls = self.criterion(output_logits=output_logits, target=target, extra_info=extra_info)
                        #     loss_all=loss_cls
                        ############################################################################################
                    else:#单模型训练走这个分支
                        loss_cls = self.criterion(output_logits=output_logits, target=target)
                        k=1#同类别下的样本数量
                        loss_scl = self.compute_supervised_contrastive_loss_sm(output["feat"],output["feat_stop_grad"],output["view_feat"],output["view_feat_stop_grad"],target,k)#,output_view["feat"],output_view["feat_stop_grad"]##,epoch,compute_feat_epoch
                        alpha = 0.5
                        loss_all=alpha*loss_scl+(1-alpha)*loss_cls
                        # loss_all = alpha *loss_scl +  loss_cls
                        # loss_all =  loss_cls
            if not use_fp16:#执行
                loss_all.backward()
                self.optimizer.step()
            else:#未执行
                self.scaler.scale(loss_all).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss_all.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_logits, target, return_length=True))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} max group LR: {:.4f} min group LR: {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    # loss_cls.item(),
                    loss_all.item(),
                    max([param_group['lr'] for param_group in self.optimizer.param_groups]),
                    min([param_group['lr'] for param_group in self.optimizer.param_groups])))
                ################################    暂时删除    ##################################
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                #################################################################################

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:#执行
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
        ###############################################################################
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        ###############################################################################
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
           
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data,view=None,view1=None)
                if isinstance(output, dict):
                    output = output["output"]
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, return_length=True))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
