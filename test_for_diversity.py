import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser


def main(config):  
    logger = config.get_logger('test')
    # setup data_loader 
    data_loader = getattr(module_data, config['data_loader']['type'])(
        # config['data_loader']['type']=ImbalanceCIFAR100DataLoader
        config['data_loader']['args']['data_dir'],
        batch_size=256,
        shuffle=False,
        training=False,
        num_workers=0  # 源代码是12
    )

    # build model architecture
    if 'returns_feat' in config['arch']['args']:
        model = config.init_obj('arch', module_arch, allow_override=True, returns_feat=True)
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

    bool_list1=[]
    bool_list2 = []
    bool_list3 = []
    for i in range(10000):
        bool_list1.append(1)
        bool_list2.append(1)
        bool_list3.append(1)
    bl1=0
    bl2 = 0
    bl3 = 0

    with torch.no_grad():
        feature_list = []
        for i in range(100):
            feature_list.append([])
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            # output = model(data)
            output = model(data, view=None, view1=None)

            output["logits"]=output["logits"].transpose(0, 1)
            expert1_logits = output['logits'][0]
            expert2_logits = output['logits'][1]
            expert3_logits = output['logits'][2]

            for t1, p1 in zip(target.view(-1), expert1_logits.argmax(dim=1).view(-1)):
                if t1.long()!=p1.long():
                    bool_list1[bl1]=0
                bl1=bl1+1
            for t2, p2 in zip(target.view(-1), expert2_logits.argmax(dim=1).view(-1)):
                if t2.long()!=p2.long():
                    bool_list2[bl2]=0
                bl2=bl2+1
            for t3, p3 in zip(target.view(-1), expert3_logits.argmax(dim=1).view(-1)):
                if t3.long()!=p3.long():
                    bool_list3[bl3]=0
                bl3=bl3+1

    sum1=0.0
    sum2=0.0
    sum3=0.0

    for i1 in range(len(bool_list1)):
        if bool_list1[i1]==0 and bool_list2[i1]==0:
            sum1=sum1+1.0

    for i2 in range(len(bool_list1)):
        if bool_list1[i2]==0 and bool_list3[i2]==0:
            sum2=sum2+1.0

    for i3 in range(len(bool_list1)):
        if bool_list2[i3]==0 and bool_list3[i3]==0:
            sum3=sum3+1.0
    sum_all=(sum1+sum2+sum3)/30000.0
    print("sum_all=",sum_all)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
