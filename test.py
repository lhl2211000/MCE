import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser


def main(config):#读入的config是相应的pth文件同级路径下的config.json
    logger = config.get_logger('test')
    # setup data_loader instances，返回cifar100的测试集
    data_loader = getattr(module_data, config['data_loader']['type'])(# config['data_loader']['type']=ImbalanceCIFAR100DataLoader
        config['data_loader']['args']['data_dir'],
        batch_size=256,
        shuffle=False,
        training=False,
        num_workers=0#源代码是12
    )

    # build model architecture
    if 'returns_feat' in config['arch']['args']:
    #############################################################################################
        # model = config.init_obj('arch', module_arch, allow_override=True, returns_feat=False)
        model = config.init_obj('arch', module_arch, allow_override=True, returns_feat=True)
    #############################################################################################
    else:
        model = config.init_obj('arch', module_arch) 

    # get function handles of loss and metrics 
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    # print("=============================MCE中的三个专家的分类器权重===================================")
    # weight_e1 = torch.norm(model.backbone.linears[0].weight,p=2,dim=0)
    # weight_e2 = torch.norm(model.backbone.linears[1].weight, p=2, dim=0)
    # weight_e3 = torch.norm(model.backbone.linears[2].weight, p=2, dim=0)
    # print(model.backbone.linears[0].weight.shape)
    # print("=================================================================================")
    # print("=============================单模型的分类器权重===================================")
    # print(model)
    # # weight = torch.norm(model.backbone.linears[0].weight,p=2,dim=0)
    # print("=================================================================================")
 
    total_metrics = torch.zeros(len(metric_fns))

    num_classes = config._config["arch"]["args"]["num_classes"]
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()


    get_class_acc = True
    if get_class_acc:
        train_data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=256,
            training=True
        ) 
        train_cls_num_list = np.array(train_data_loader.cls_num_list)
        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
        few_shot = train_cls_num_list < 20

    with torch.no_grad():
        feature_list = []
        for i in range(100):
            feature_list.append([])
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            # output = model(data)
            output = model(data,view=None,view1=None)

            raw_feat = output["raw_feat"]
            # print("=========================原始特征=====================================")
            # print(len(raw_feat))
            # print(len(raw_feat[0]))
            # print(len(raw_feat[0][0]))
            # print("====================================================================")
            for rt in zip(raw_feat[2], target):
                feature_list[rt[1]].append(rt[0])
            #####   用于单模型和集成模型的集成结果的测试   #######
            # output=output["output"]
            #############################
            ###############################################################
            # expert3_raw_feat=output["raw_feat"][:,2,:]
            # e33_logits=model.backbone.linears[1](expert3_raw_feat)
            output["logits"]=output["logits"].transpose(0, 1)
            expert1_logits = output['logits'][2]
            # expert2_logits = output['logits'][1]
            # expert3_logits = output['logits'][2]
            ###############################################################
            batch_size = data.shape[0] 
            for i, metric in enumerate(metric_fns):
                #############################################################
                # total_metrics[i] += metric(output, target) * batch_size
                total_metrics[i] += metric(expert1_logits, target) * batch_size
                # total_metrics[i] += metric(e33_logits, target) * batch_size
                #############################################################
            #########################################################################
            # for t, p in zip(target.view(-1), output.argmax(dim=1).view(-1)):
            for t, p in zip(target.view(-1), expert1_logits.argmax(dim=1).view(-1)):
            # for t, p in zip(target.view(-1), e33_logits.argmax(dim=1).view(-1)):
            #########################################################################
                confusion_matrix[t.long(), p.long()] += 1

        d=[]
        for f1 in range(100):
            sum=0.0
            for f2 in feature_list[f1]:
                # print(model.backbone.linears[0].weight.shape)
                # print("f2=",f2.shape)
                # print("model.backbone.linears[0].weight[0:48,f1]=", model.backbone.linears[0].weight[0:48,f1].shape)
                sum=sum+torch.norm(model.backbone.linears[2].weight[0:48,f1]-f2,p=2,dim=0).item()
            d.append(sum/100.0)
        d1=[float('{:.4f}'.format(i)) for i in d]
        print("d3=",d1)
        print("d1'len=", len(d1))

        # class_mean_feature_norm = []
        # for f1 in range(100):
        #     temp = 0.0
        #     for f2 in feature_list[f1]:
        #         temp = temp + f2
        #     class_mean_feature_norm.append(torch.norm(temp / 100, p=2, dim=0).item())
        # class_mean_feature_norm1 = [float('{:.4f}'.format(i)) for i in class_mean_feature_norm]
        # print("==========================类别平均特征================================================")
        # print(class_mean_feature_norm1)
        # print(len(class_mean_feature_norm1))
        # print("===================================================================================")


    acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1) 
    acc = acc_per_class.cpu().numpy() 
  
    many_shot_acc = acc[many_shot].mean()
    medium_shot_acc = acc[medium_shot].mean()
    few_shot_acc = acc[few_shot].mean() 

    n_samples = len(data_loader.sampler)
    log = {}
    log.update( {met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)} )

    if get_class_acc:
        log.update({
            "many_class_num": many_shot.sum(),
            "medium_class_num": medium_shot.sum(),
            "few_class_num": few_shot.sum(),
            "many_shot_acc": many_shot_acc,
            "medium_shot_acc": medium_shot_acc,
            "few_shot_acc": few_shot_acc,
        })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
