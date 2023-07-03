# From https://github.com/kaidic/LDAM-DRW/blob/master/models/resnet_cifar.py
'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

import random

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):#48,100
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):#torch.mm就是矩阵乘积运算
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class NormedLinear_without_bias(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear_without_bias, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):#torch.mm就是矩阵乘积运算
        out = x.mm(F.normalize(self.weight, dim=0))
        return out

class NormedLinear_with_bias(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear_with_bias, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.bias = Parameter(torch.Tensor(out_features))
        self.bias.data.uniform_(-1, 1)#.renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):#torch.mm就是矩阵乘积运算
        out = x.mm(F.normalize(self.weight, dim=0))+self.bias
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.planes = planes
                self.in_planes = in_planes
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                self.shortcut = LambdaLayer(lambda x:F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2), "constant", 0))
                
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_s(nn.Module):
    #根据作者在github上的回答，使用s=30是为了调整余弦分类器的logit幅度
    #              BasicBlock, [5, 5, 5],           3,            100,                   True,                   None,                   None,           True,              True,             None, s=30)
    def __init__(self, block, num_blocks, num_experts, num_classes=10, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, returns_feat=True, use_experts=None, s=30):
        super(ResNet_s, self).__init__()
        
        self.in_planes = 16
        self.num_experts = num_experts

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        #                               BasicBlock,           5
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        #####################    将并行结构上移1/2层    #########################
        # self.conv1s=nn.ModuleList( [self._make_conv1s() for _ in range(num_experts)] )
        # self.layer1s=nn.ModuleList( [self._make_layer(block, 16, num_blocks[0], stride=1) for _ in range(num_experts)] )
        #####################################################################
        self.in_planes = self.next_in_planes

        if layer2_output_dim is None:#执行
            if reduce_dimension:#执行
                layer2_output_dim = 24
            else:
                layer2_output_dim = 32

        if layer3_output_dim is None:#执行
            if reduce_dimension:#执行
                layer3_output_dim = 48
            else:
                layer3_output_dim = 64

        #self.layer2s和self.layer3s这两个变量最后的s表示复数的意思
        self.layer2s = nn.ModuleList([self._make_layer(block, layer2_output_dim, num_blocks[1], stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        ################################# simsiam：用于监督对比损失的组件 ###################################################
        self.projector_final_dim=1024
        self.bottleneck_dim=256
        # print('===============================================')
        # print('self.projector_final_dim:',self.projector_final_dim)
        # print('===============================================')
        # self.projector = nn.Sequential( nn.Linear(layer3_output_dim, layer3_output_dim, bias=False),
        #                                 nn.BatchNorm1d(layer3_output_dim),
        #                                 nn.ReLU(inplace=True), # first layer
        #                                 nn.Linear(layer3_output_dim, layer3_output_dim, bias=False),
        #                                 nn.BatchNorm1d(layer3_output_dim),
        #                                 nn.ReLU(inplace=True), # second layer
        #                                 # nn.Linear(layer3_output_dim, layer3_output_dim),
        #                                 nn.Linear(layer3_output_dim, self.projector_final_dim),
        #                                 nn.BatchNorm1d(self.projector_final_dim, affine=False)) # output layer
        # self.projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
        #
        # # build a 2-layer predictor
        # self.predictor = nn.Sequential( nn.Linear(self.projector_final_dim, self.bottleneck_dim, bias=False),#
        #                                 nn.BatchNorm1d(self.bottleneck_dim),
        #                                 nn.ReLU(inplace=True), # hidden layer
        #                                 nn.Linear(self.bottleneck_dim, self.projector_final_dim)) # output layer #

        # self.projectors = nn.ModuleList([self.projector for _ in range(num_experts)])
        # self.predictors = nn.ModuleList([self.predictor for _ in range(num_experts)])
        self.projectors = nn.ModuleList([self._make_projector(layer3_output_dim,self.projector_final_dim) for _ in range(num_experts)])
        self.predictors = nn.ModuleList([self._make_predictor(self.projector_final_dim, self.bottleneck_dim) for _ in range(num_experts)])
        ################################################################################################################

        if use_norm:#执行
            self.linears = nn.ModuleList([NormedLinear(layer3_output_dim, num_classes) for _ in range(num_experts)])
            # self.linears = nn.ModuleList([NormedLinear(2048, num_classes) for _ in range(num_experts)])
        else:
            self.linears = nn.ModuleList([nn.Linear(layer3_output_dim, num_classes) for _ in range(num_experts)])
            s = 1

        if use_experts is None:#执行
            self.use_experts = list(range(num_experts))#self.use_experts=[0, 1, 2]
        elif use_experts == "rand":#未执行
            self.use_experts = None
        else:#未执行
            self.use_experts = [int(item) for item in use_experts.split(",")]

        self.s = s#self.s = 30
        self.returns_feat = returns_feat#True
        self.apply(_weights_init)#线性层和卷积层的权重初始化

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)#[1]*4=[1,1,1,1];[1]+[1,1,1,1]=[1,1,1,1,1]
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:#[1,1,1,1,1]
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion#block.expansion=1

        return nn.Sequential(*layers)#*号表示取列表中元素的意思    设l1=[1,2,3],则*l1=1 2 3

    def _make_projector(self,layer3_output_dim, projector_final_dim):
        projector=nn.Sequential(nn.Linear(layer3_output_dim, projector_final_dim, bias=False),
                                # NormedLinear_without_bias(layer3_output_dim, layer3_output_dim),
                                nn.BatchNorm1d(projector_final_dim),
                                nn.ReLU(inplace=True), # first layer
                                nn.Linear(projector_final_dim, projector_final_dim, bias=False),
                                # NormedLinear_without_bias(layer3_output_dim, layer3_output_dim),
                                nn.BatchNorm1d(projector_final_dim),
                                nn.ReLU(inplace=True), # second layer
                                nn.Linear(projector_final_dim, projector_final_dim),
                                # NormedLinear_with_bias(layer3_output_dim, projector_final_dim),
                                nn.BatchNorm1d(projector_final_dim, affine=False)) # output layer
        projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # print("第一个projector=====================================")
        # print(projector)
        # print("第一个projector====================================")

        return projector


    def _make_predictor(self, projector_final_dim, bottleneck_dim):
        predictor=nn.Sequential(nn.Linear(projector_final_dim, bottleneck_dim, bias=False),
                                # NormedLinear_without_bias(projector_final_dim, bottleneck_dim),
                                nn.BatchNorm1d(bottleneck_dim),
                                nn.ReLU(inplace=True), # hidden layer
                                nn.Linear(bottleneck_dim, projector_final_dim)) # output layer
                                # NormedLinear_with_bias(bottleneck_dim, projector_final_dim))
        # print("第二个predictor=====================================")
        # print(predictor)
        # print("第二个predictor=======================================")

        return predictor

    # def _make_conv1s(self):
    #     conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    #     bn1 = nn.BatchNorm2d(16)
    #     return nn.Sequential(conv1,bn1)

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _separate_part(self, x,view,view1, ind):#x是神经网络前半部分共同的输出,ind是专家索引
        out = x
        out = (self.layer2s[ind])(out)#(self.layer2s[ind])表示的是ModuleList中的第ind个专家，是一个Sequential，这一句也就是在调用Sequential(x)
        out = (self.layer3s[ind])(out)
        self.feat_before_GAP.append(out)##feat_before_GAP是一个list，每一个元素是一个tensor，每一个tensor的shape=128*48*8*8
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        self.raw_feat.append(out)
        ####################################################
        if view != None:
            out_view = view
            out_view = (self.layer2s[ind])(out_view)
            out_view = (self.layer3s[ind])(out_view)
            # self.feat_before_GAP.append(out_view)
            out_view = F.avg_pool2d(out_view, out_view.size()[3])
            out_view = out_view.view(out_view.size(0), -1)

            out_view1 = view1
            out_view1 = (self.layer2s[ind])(out_view1)
            out_view1 = (self.layer3s[ind])(out_view1)
            # self.feat_before_GAP.append(out_view)
            out_view1 = F.avg_pool2d(out_view1, out_view1.size()[3])
            out_view1 = out_view1.view(out_view1.size(0), -1)
        ####################################################
        ###################  让原始的特征依次通过projector,predictor  ##################
        # out_feat = (self.projectors[ind])(out)
            out_feat = (self.projectors[ind])(F.normalize(out_view,dim=1))
            self.feat_stop_grad.append(out_feat.detach())
            out_feat = (self.predictors[ind])(out_feat)
            self.feat.append(out_feat)#feat是一个list，每一个元素是一个tensor，每一个tensor的shape=128*48

            out_feat1 = (self.projectors[ind])(F.normalize(out_view1, dim=1))
            self.feat_stop_grad1.append(out_feat1.detach())
            out_feat1 = (self.predictors[ind])(out_feat1)
            self.feat1.append(out_feat1)
        ############################################################################
        #######################    加入监督对比损是之后的两种模型结构    ############################
        out = (self.linears[ind])(out)#分支型模型
        # out = (self.linears[ind])(out_feat)#直线型模型
        #######################################################################################
        out = out * self.s
        return out
    ##############################################################################
    # def _separate_part1(self, x, ind):
    #     out = x
    #     out = (self.layer1s[ind])(out)
    #     return out
    #
    # def _separate_part2(self, x, ind):
    #     out = x
    #     out = (self.conv1s[ind])(out)
    #     return F.relu(out)
    ##############################################################################

    def forward(self, x, view,view1):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        ####################################################
        if view!=None:
            out_view = F.relu(self.bn1(self.conv1(view)))
            out_view = self.layer1(out_view)

            out_view1 = F.relu(self.bn1(self.conv1(view1)))
            out_view1 = self.layer1(out_view1)
        else:
            out_view=None
            out_view1 = None
        ####################################################
        
        outs = []
        self.feat = []
        self.feat1 = []
        #####################################################################
        self.raw_feat=[]
        self.feat_stop_grad = []
        self.feat_stop_grad1 = []
        #####################################################################

        self.logits = outs
        self.feat_before_GAP = []
        
        if self.use_experts is None:#未执行
            use_experts = random.sample(range(self.num_experts), self.num_experts - 1)
        else:
            use_experts = self.use_experts
        ################################################################
        # out2=[]
        # for ind in use_experts:  # [0,1,2]
        #     out2.append(self._separate_part2(x, ind))
        # out1=[]
        # for ind in use_experts:  # [0,1,2]
        #     out1.append(self._separate_part1(out2[ind], ind))
        ################################################################
        for ind in use_experts:#[0,1,2]
            outs.append(self._separate_part(out,out_view,out_view1 ,ind))#得到了3个专家的输出：tensor.shape=128*100(批处理数量*数据集类别数量)
            # outs.append(self._separate_part(out1[ind], ind))
        if view != None:
            self.feat = torch.stack(self.feat, dim=1)
        #####################################################################
            self.feat_stop_grad = torch.stack(self.feat_stop_grad, dim=1)
        # self.raw_feat=torch.stack(self.raw_feat, dim=1)

            self.feat1 = torch.stack(self.feat1, dim=1)
            #####################################################################
            self.feat_stop_grad1 = torch.stack(self.feat_stop_grad1, dim=1)
        #####################################################################

        # self.feat_before_GAP = torch.stack(self.feat_before_GAP, dim=1)
        final_out = torch.stack(outs, dim=1).mean(dim=1)#stack导致tensor.shape变成了128*3*100，mean导致tensor.shape变回了128*100
        # if self.returns_feat:#执行
        #     return {
        #         "output": final_out,
        #         "raw_feat":self.raw_feat,
        #         "feat": self.feat,
        #         "feat_stop_grad": self.feat_stop_grad,#我加上的
        #         "logits": torch.stack(outs, dim=1)
        #     }
        # else:#未执行
        #     return final_out
        if view!=None:#执行
            return {
                "output": final_out,
                "raw_feat":self.raw_feat,
                "feat": self.feat,
                "feat_stop_grad": self.feat_stop_grad,#我加上的

                "view_feat": self.feat1,
                "view_feat_stop_grad": self.feat_stop_grad1,  # 我加上的

                "logits": torch.stack(outs, dim=1)
            }
        else:#未执行
            self.feat = torch.stack(self.raw_feat, dim=1)
            return {
                "output": final_out,
                "raw_feat": self.raw_feat,
                # "feat": self.feat,
                # "feat_stop_grad": self.feat_stop_grad,  # 我加上的
                "logits": torch.stack(outs, dim=1)
            }

def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])

def resnet32(num_classes=10, use_norm=False):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)

def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])

def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])

def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])

def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])

def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()