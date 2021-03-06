#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/8/30 上午11:29
# @Author  : Joselynzhao
# @Email   : zhaojing17@forxmail.com
# @File    : fsr_atmpro12.py
# @Software: PyCharm
# @Desc    :

from reid.eug import *
from reid import datasets
from reid import models
import numpy as np
import torch
import argparse
import os

import warnings

warnings.filterwarnings("ignore")

from reid.utils.logging import Logger
import os.path as osp
import sys
from torch.backends import cudnn
from reid.utils.serialization import load_checkpoint
from torch import nn
import time
import pickle
import torch.distributed as dist
from torch.nn.parallel import  DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from  pathlib import  Path



def resume(savepath):
    import re
    pattern = re.compile(r'step_(\d+)\.ckpt')
    start_step = -1
    ckpt_file = ""

    # find start step
    files = os.listdir(savepath)
    files.sort()
    for filename in files:
        try:
            iter_ = int(pattern.search(filename).groups()[0])
            print(iter_)
            if iter_ > start_step:
                start_step = iter_
                ckpt_file = osp.join(savepath, filename)
        except:
            continue

    # if need resume
    if start_step >= 0:
        print("continued from iter step", start_step)
    else:
        print("resume failed", start_step, files)
    return start_step, ckpt_file


def main(args):
    father = Path('/mnt/')
    if father.exists(): # 是在服务器上
        data_dir = Path('/mnt/share/datasets/RE-ID/data')  # 服务器
        logs_dir = Path('/mnt/home/{}'.format(args.log_name))  # 服务器
    else: #本地
        data_dir = Path('/home/joselyn/workspace/ATM_SERIES/data')  # 本地跑用这个
        logs_dir = Path('/home/joselyn/workspace/ATM_SERIES/{}'.format(args.log_name))  # 本地跑用这个


    cudnn.benchmark = True
    cudnn.enabled = True
    save_path = os.path.join(logs_dir, args.dataset, args.exp_name, args.exp_order)  # 到编号位置.
    total_step = 100 // args.EF + 1
    sys.stdout = Logger(osp.join(save_path, 'log' + str(args.EF) + time.strftime(".%m_%d_%H:%M:%S") + '.txt'))
    dataf_file = open(osp.join(save_path, 'dataf.txt'), 'a')  # 保存性能数据.  #特征空间中的性能问题.
    data_file = open(osp.join(save_path, 'data.txt'), 'a')  # 保存性能数据.  #特征空间中的性能问题.
    kf_file = open(osp.join(save_path,'kf.txt'),'a')
    # 数据格式为 label_pre_r, select_pre_r,label_pre_t, select_pre_t  ,加上了了tagper的数据.
    tagper_path = osp.join(save_path,'tagper')  #tagper存储路径.
    if not Path(tagper_path).exists():
        os.mkdir(tagper_path)


    '''# 记录配置信息 和路径'''
    print('-'*20+'config_info'+'-'*20)
    config_file = open(osp.join(save_path, 'config.txt'), 'w')
    config_info = str(args).split('(')[1].strip(')').split(',')
    config_info.sort()
    for one in config_info:
        key,value=map(str,one.split('='))
        config_file.write(key.strip()+'='+value.strip('\'')+'\n')
        print(key.strip()+'='+value.strip('\''))
    config_file.write('save_path='+save_path)
    print('save_path='+save_path)
    print('-' * 20 + 'config_info' + '-' * 20)
    config_file.close()

    train_time_file = open(osp.join(save_path, 'time.txt'), 'a')  # 只记录训练所需要的时间.
    # 数据格式为 step_time total_time.
    total_time = 0

    # get all the labeled and unlabeled data for training

    dataset_all = datasets.create(args.dataset, osp.join(data_dir, args.dataset))
    num_all_examples = len(dataset_all.train)
    l_data, u_data = get_init_shot_in_cam1(dataset_all,
                                           load_path="./examples/{}_init_{}.pickle".format(dataset_all.name, args.init),
                                           init=args.init)

    resume_step, ckpt_file = -1, ''
    if args.resume:
        resume_step, ckpt_file = resume(save_path)

        # initial the EUG algorithm
    eug = EUG(batch_size=args.batch_size, num_classes=dataset_all.num_train_ids,
              dataset=dataset_all, l_data=l_data, u_data=u_data, save_path=save_path, max_frames=args.max_frames,
              embeding_fea_size=args.fea, momentum=args.momentum, lamda=args.lamda)
    tagper = EUG(batch_size=args.batch_size, num_classes=dataset_all.num_train_ids,
                 dataset=dataset_all, l_data=l_data, u_data=u_data, save_path=tagper_path,
                 max_frames=args.max_frames,
                 embeding_fea_size=args.fea, momentum=args.momentum, lamda=args.lamda)

    new_train_data = l_data
    unselected_data = u_data
    iter_mode = 2 #迭代模式,确定是否训练tagper
    for step in range(total_step):
        # for resume
        if step < resume_step:
            continue


        # pro12
        ratio = (step + 1) * args.EF / 100
        nums_to_select = int(len(u_data) * ratio)

        ratio_t = (step+1+args.t) * args.EF /100
        y1 = (1 - args.t) * (step + 1 - total_step) / (total_step - 1) + 1
        nums_to_select_tagper1 = int(nums_to_select * y1)
        nums_to_select_tagper2 = int(len(u_data) * ratio_t)
        nums_to_select_tagper =int( (nums_to_select_tagper1+nums_to_select_tagper2)/2 )
        print("nums_to_select_tagper1 = {} nums_to_select_tagper2 = {} nums_to_select_tagper = {} ".format(nums_to_select_tagper1,nums_to_select_tagper2,nums_to_select_tagper))
        if nums_to_select >= len(u_data):
            break

        # yml: ... Ritio {} ...
        print("Runing: EF={}%, step {}:\t Nums_to_be_select {} \t Ritio {}\t Logs-dir {}".format(
            args.EF, step, nums_to_select, ratio, save_path))

        # train the model or load ckpt
        start_time = time.time()
        print("training reid model")
        eug.train(new_train_data, unselected_data, step, loss=args.loss, epochs=args.epochs, step_size=args.step_size,
                  init_lr=0.1) if step != resume_step else eug.resume(ckpt_file, step)

        # 只对eug进行性能评估
        # mAP, rank1, rank5, rank10, rank20 = 0,0,0,0,0
        mAP, rank1, rank5, rank10, rank20 = eug.evaluate(dataset_all.query, dataset_all.gallery)
        # 把数据写到data文件里.
        data_file.write('{} {:.2%} {:.2%} {:.2%} {:.2%} {:.2%}\n'.format(step, mAP, rank1, rank5, rank10, rank20))

        pred_y, pred_score,label_pre = eug.estimate_label()
        selected_idx = eug.select_top_data(pred_score, min(nums_to_select_tagper,len(u_data)-50) if iter_mode==2 else min(nums_to_select,len(u_data)))   #直接翻两倍取数据. -50个样本,保证unselected_data数量不为0
        new_train_data, unselected_data, select_pre= eug.generate_new_train_data(selected_idx, pred_y,step,args.experiment) # yml:新增后两个参数
        raw_label_pre, raw_select_pre = label_pre,select_pre
        t_label_pre,t_select_pre = 0,0
        raw_select_pre_t = 0
        # label_pre_t,select_pre_t=0,0
        if iter_mode==2:
            raw_select_pre_t = raw_select_pre
            print("training tagper model")
            selected_idx = eug.select_top_data(pred_score, min(nums_to_select, len(u_data)))
            _, _, raw_select_pre = eug.generate_new_train_data(selected_idx, pred_y,step,args.experiment) # yml:新增后两个参数
            # kf_file.write('{} {:.2%} {:.2%}'.format(step, label_pre, select_pre))

            tagper.resume(osp.join(save_path,'step_{}.ckpt'.format(step)),step)
            tagper.train(new_train_data, unselected_data, step, loss=args.loss, epochs=args.epochs, step_size=args.step_size, init_lr=0.1)

            pred_y, pred_score, label_pre= tagper.estimate_label()
            selected_idx = tagper.select_top_data(pred_score,min(nums_to_select,len(u_data)))  # 采样目标数量
            new_train_data, unselected_data, select_pre= tagper.generate_new_train_data(selected_idx, pred_y,step,args.experiment) # yml:新增后两个参数
            t_label_pre,t_select_pre = label_pre,select_pre
            label_pre,select_pre = t_label_pre,t_select_pre
            if nums_to_select_tagper >=len(u_data):
                iter_mode=1 #切换模式
                print('tagper is stop')
        else: #mode = 1
            # raw_select_pre = raw_select_pre_t
            # raw_select_pre_t = 0
            label_pre,select_pre = raw_label_pre,raw_select_pre

        end_time = time.time()
        step_time = end_time - start_time
        total_time = step_time + total_time
        train_time_file.write('{} {:.6} {:.6}\n'.format(step, step_time, total_time))
        kf_file.write('{} {} {} {:.2%} {:.2%} {:.2%} {:.2%} {:.2%}\n'.format(step,nums_to_select,nums_to_select_tagper,raw_label_pre,raw_select_pre,raw_select_pre_t,t_label_pre,t_select_pre))
        dataf_file.write(
            '{} {:.2%} {:.2%}\n'.format(step, label_pre, select_pre))
    dataf_file.close()
    train_time_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progressive Learning for One-Example re-ID')
    parser.add_argument('-d', '--dataset', type=str, default='mars',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-f', '--fea', type=int, default=1024)
    parser.add_argument('--EF', type=int, default=10)
    parser.add_argument('--t', type=float, default=2) #tagper 采样的倍率
    parser.add_argument('--exp_order', type=str, default='0')
    parser.add_argument('--exp_name', type=str, default='atm')
    parser.add_argument('--exp_aim', type=str, default='for paper')
    parser.add_argument('--run_file',type=str,default='train.py')
    parser.add_argument('--log_name',type=str,default='pl_logs')

    parser.add_argument('--resume', type=str, default='Yes')
    parser.add_argument('--max_frames', type=int, default=900)
    parser.add_argument('--loss', type=str, default='ExLoss', choices=['CrossEntropyLoss', 'ExLoss'])
    parser.add_argument('--init', type=float, default=-1)
    parser.add_argument('-m', '--momentum', type=float, default=0.5)
    parser.add_argument('-e', '--epochs', type=int, default=70)
    parser.add_argument('-s', '--step_size', type=int, default=55)
    parser.add_argument('--lamda', type=float, default=0.5)
    # yml: FSR新增的参数
    parser.add_argument('--experiment', type=str, default='1')
    main(parser.parse_args())
