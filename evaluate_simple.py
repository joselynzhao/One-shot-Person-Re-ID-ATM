from my_reid.eug import *
from my_reid import datasets
from my_reid import models
import numpy as np
import torch
import argparse
import os

import warnings

warnings.filterwarnings("ignore")

from my_reid.utils.logging import Logger
import os.path as osp
import sys
from torch.backends import cudnn
from my_reid.utils.serialization import load_checkpoint
from torch import nn
import time
import pickle
import torch.distributed as dist
from torch.nn.parallel import  DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from pathlib import  Path



def main(args):
    father = Path('/mnt/')
    if father.exists():  # 是在服务器上
        data_dir = Path('/mnt/share/datasets/RE-ID/data')  # 服务器
        # logs_dir = Path('/mnt/home/{}'.format(args.log_name))  # 服务器
    else:  # 本地
        data_dir = Path('/home/joselyn/workspace/ATM_SERIES/data')  # 本地跑用这个
        # logs_dir = Path('/home/joselyn/workspace/ATM_SERIES/{}'.format(args.log_name))  # 本地跑用这个

    cudnn.benchmark = True
    cudnn.enabled = True
    save_path = args.save_path
    sys.stdout = Logger(osp.join(save_path, 'log_evaluate' + time.strftime(".%m_%d_%H:%M:%S") + '.txt'))
    data_file = open(osp.join(save_path, 'data.txt'), 'a')  # 保存性能数据.
    # 数据格式为 mAP, rank-1,rank-5,rank-10,rank-20

    dataset_all = datasets.create(args.dataset, osp.join(data_dir, args.dataset))
    eval_list = args.eval_No
    if len(eval_list)==0: #一个也没有的适合,测试全部.
        import re
        eval_list = []
        pattern = re.compile(r'step_(\d+)\.ckpt')  #设定匹配格式.
        files = os.listdir(save_path)
        files.sort()
        for filename in files:
            try:
                iter_ = int(pattern.search(filename).groups()[0])  #找到的中间的数字.
                # print(iter_)
                eval_list.append(iter_)
            except:
                continue
    if len(eval_list)==0:
        print("{} has no files to evaluate".format(save_path))
        return
    eval_list.sort()  #排个序,输出更好看一些.
    print(eval_list)  # 这里可以看到 总共会评估哪些模型.
    # initial the EUG algorithm
    eug = EUG(batch_size=args.batch_size, num_classes=dataset_all.num_train_ids,
              dataset=dataset_all, l_data=[], u_data=[], save_path=save_path, max_frames=args.max_frames,
              embeding_fea_size=args.fea, momentum=args.momentum, lamda=args.lamda)

    for step in eval_list:
        ckpt_file = osp.join(save_path,'step_{}.ckpt'.format(step))
        eug.resume(ckpt_file,step)
        mAP, rank1, rank5, rank10, rank20 = eug.evaluate(dataset_all.query, dataset_all.gallery)
        data_file.write(
            '{} {:.2%} {:.2%} {:.2%} {:.2%} {:.2%}\n'.format(step, mAP, rank1, rank5, rank10, rank20,))
    data_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progressive Learning for One-Example re-ID')
    parser.add_argument('-d', '--dataset', type=str, default='mars',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--save_path', type=str, default='defalut')
    parser.add_argument('--eval_No', type=list, default=[])  # 默认为空则全部测试.
    parser.add_argument('--max_frames', type=int, default=100)
    parser.add_argument('--loss', type=str, default='ExLoss', choices=['CrossEntropyLoss', 'ExLoss'])
    parser.add_argument('-f', '--fea', type=int, default=1024)
    parser.add_argument('-m', '--momentum', type=float, default=0.5)
    parser.add_argument('--lamda', type=float, default=0.5)
    main(parser.parse_args())
