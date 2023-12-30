"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import autocast


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_experts, dropout=None, num_classes=1000, use_norm=False,
                 reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, share_layer3=False,
                 returns_feat=False, s=30):
        self.inplanes = 64
        self.num_experts = num_experts
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        self.share_layer3 = share_layer3

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 192
            else:
                layer3_output_dim = 256

        if layer4_output_dim is None:
            if reduce_dimension:
                layer4_output_dim = 384
            else:
                layer4_output_dim = 512

        if self.share_layer3:
            self.layer3 = self._make_layer(block, layer3_output_dim, layers[2], stride=2)
        else:
            self.layer3s = nn.ModuleList(
                [self._make_layer(block, layer3_output_dim, layers[2], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.layer4s = nn.ModuleList(
            [self._make_layer(block, layer4_output_dim, layers[3], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.use_dropout = True if dropout else False

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)
       
        if num_classes==365:
            self.raw_feature = 2048
        else:
            self.raw_feature = 1536 
        self.projector_final_dim = 2048
        self.bottleneck_dim = 512

        self.projectors = nn.ModuleList(
            [self._make_projector(self.raw_feature, self.projector_final_dim) for _ in range(num_experts)])
        self.predictors = nn.ModuleList(
            [self._make_predictor(self.projector_final_dim, self.bottleneck_dim) for _ in range(num_experts)])
        num = 0
        for para in self.projectors.parameters():
            num = num + para.numel()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if use_norm:
            self.linears = nn.ModuleList(
                [NormedLinear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
        else:
            self.linears = nn.ModuleList(
                [nn.Linear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
            s = 1

        self.s = s

        self.returns_feat = returns_feat

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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.next_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.next_inplanes, planes))

        return nn.Sequential(*layers)

    def _make_projector(self, raw_feature, projector_final_dim):
        projector = nn.Sequential(nn.Linear(raw_feature, projector_final_dim, bias=False),
                                  # NormedLinear_without_bias(layer3_output_dim, layer3_output_dim),
                                  nn.BatchNorm1d(projector_final_dim),
                                  nn.ReLU(inplace=True),  # first layer
                                  nn.Linear(projector_final_dim, projector_final_dim, bias=False),
                                  # NormedLinear_without_bias(layer3_output_dim, layer3_output_dim),
                                  nn.BatchNorm1d(projector_final_dim),
                                  nn.ReLU(inplace=True),  # second layer
                                  nn.Linear(projector_final_dim, projector_final_dim),
                                  # NormedLinear_with_bias(layer3_output_dim, projector_final_dim),
                                  nn.BatchNorm1d(self.projector_final_dim, affine=False))  # output layer
        projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        return projector

    def _make_predictor(self, projector_final_dim, bottleneck_dim):
        predictor = nn.Sequential(nn.Linear(projector_final_dim, bottleneck_dim, bias=False),
                                  # NormedLinear_without_bias(projector_final_dim, bottleneck_dim),
                                  nn.BatchNorm1d(bottleneck_dim),
                                  nn.ReLU(inplace=True),  # hidden layer
                                  nn.Linear(bottleneck_dim, projector_final_dim))  # output layer
        # NormedLinear_with_bias(bottleneck_dim, projector_final_dim))
        return predictor

    def _separate_part(self, x, view, view1, ind):
        if not self.share_layer3:
            x = (self.layer3s[ind])(x)

        x = (self.layer4s[ind])(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        if self.use_dropout:
            x = self.dropout(x)

        if view != None:
            if not self.share_layer3:
                out_view = (self.layer3s[ind])(view)
            out_view = (self.layer4s[ind])(out_view)

            out_view = self.avgpool(out_view)

            out_view = out_view.view(out_view.size(0), -1)

            if self.use_dropout:
                out_view = self.dropout(out_view)

            if not self.share_layer3:
                out_view1 = (self.layer3s[ind])(view1)
            out_view1 = (self.layer4s[ind])(out_view1)

            out_view1 = self.avgpool(out_view1)

            out_view1 = out_view1.view(out_view1.size(0), -1)

            if self.use_dropout:
                out_view1 = self.dropout(out_view1)

            out_feat = self.projectors[ind](F.normalize(out_view, dim=1))
            self.feat_stop_grad.append(out_feat.detach())
            out_feat = self.predictors[ind](out_feat)
            self.feat.append(out_feat)

            out_feat1 = self.projectors[ind](F.normalize(out_view1, dim=1))
            self.feat_stop_grad1.append(out_feat1.detach())
            out_feat1 = self.predictors[ind](out_feat1)
            self.feat1.append(out_feat1)

        else:
            out_feat = self.projectors[ind](F.normalize(x,dim=1))
            self.feat_stop_grad.append(out_feat.detach())
            out_feat = self.predictors[ind](out_feat)
            self.feat.append(out_feat)
        
        x = (self.linears[ind])(x)
        x = x * self.s
        return x

    def forward(self, x, view, view1):
        with autocast():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            if self.share_layer3:
                x = self.layer3(x)

            if view != None:
                view = self.conv1(view)
                view = self.bn1(view)
                view = self.relu(view)
                view = self.maxpool(view)

                view = self.layer1(view)
                view = self.layer2(view)
                if self.share_layer3:
                    view = self.layer3(view)

                view1 = self.conv1(view1)
                view1 = self.bn1(view1)
                view1 = self.relu(view1)
                view1 = self.maxpool(view1)

                view1 = self.layer1(view1)
                view1 = self.layer2(view1)
                if self.share_layer3:
                    view1 = self.layer3(view1)
            else:
                pass

            outs = []
            self.feat = []
            self.feat1 = []
            self.feat_stop_grad = []
            self.feat_stop_grad1 = []

            for ind in range(self.num_experts):
                outs.append(self._separate_part(x, view, view1, ind))

            self.feat_stop_grad = torch.stack(self.feat_stop_grad, dim=1)
            if view != None:
                self.feat_stop_grad1 = torch.stack(self.feat_stop_grad1, dim=1)

            final_out = torch.stack(outs, dim=1).mean(dim=1)

        if view != None:  
            return {
                "output": final_out,
                "feat": torch.stack(self.feat, dim=1),
                "feat_stop_grad": self.feat_stop_grad,
                "view_feat": torch.stack(self.feat1, dim=1),
                "view_feat_stop_grad": self.feat_stop_grad1,
                "logits": torch.stack(outs, dim=1)
            }
        else:  
            return {
                "output": final_out,
                "feat": torch.stack(self.feat, dim=1),
                "feat_stop_grad": self.feat_stop_grad, 
                "logits": torch.stack(outs, dim=1)
            }
