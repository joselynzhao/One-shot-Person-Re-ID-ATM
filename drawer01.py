import os
import codecs
import os.path as osp
import math
import matplotlib.pyplot as plt
import numpy as np
from  pathlib import  Path

class drawer01():
    def __init__(self,compare_path_list,save_name):
        father = Path('/mnt/')
        if father.exists():  # 是在服务器上
            print("正在服务器上运行")
            self.logs_dir = '/mnt/home/'  # 服务器
        else:  # 本地
            self.logs_dir = '/home/joselyn/workspace/ATM_SERIES/'  # 本地跑用这个
        self.compare_path_list = compare_path_list
        self.save_name = save_name
        self.names = [path.split('/')[-1] for path in self.compare_path_list]
        self.save_path = osp.join(self.logs_dir,'figure_out',self.save_name) # 这个是固定不变的
        if not Path(self.save_path).exists():
            print(self.save_path+'is not exist, creating now!')
            os.makedirs(self.save_path)  # 可生成多级目录
        desc_file = open(osp.join(self.save_path,'desc.txt'),'w')
        for path in self.compare_path_list:
            desc_file.write(path+'\n')
        desc_file.write('\nsave_name:'+self.save_name)
        desc_file.close()
        self.item_name_P = 'mAP,Rank-1,Rank-5,Rank-10,Rank-20'.split(',')
        self.item_name_F = 'label_pre,select_pre'.split(',')

        # self.draw_Perf()
        self.draw_Feat()

    def get_datas_F(self): # 获取特征相关的 label_pre和select_pre.
        # 正常情况下, 这两个数据是在dataf.txt 文件中的. 但是早期的训练.这个两个数据在data中.
        self.dataf = []
        for path in self.compare_path_list:
            path = osp.join(self.logs_dir,path)
            # file = ''
            # seg = []
            file_data = []
            try:
                file = open(osp.join(path,'dataf.txt'),'r')
                seg = [1,3]
            except FileNotFoundError:
                file = open(osp.join(path,'data.txt'),'r')
                seg = [6,8]
            infos = file.readlines()
            for info in infos:
                info_f = info.strip('\n').split(' ')[seg[0]:seg[1]]  # 不要第一个 step
                info_f = list(map(float, [dd.strip('%') for dd in info_f]))
                file_data.append(info_f)
            self.dataf.append(file_data)
        self.dataf = np.array(self.dataf)
        print(self.dataf)

    def get_datas_P(self): # 取性能数据.
        self.data = []
        for path in self.compare_path_list: # 对每一个实验去数据.
            path = osp.join(self.logs_dir, path)
            file_data = []
            file = open(osp.join(path,'data.txt'),'r')
            infos = file.readlines()
            for info in infos:
                info_f = info.strip('\n').split(' ')[1:6]#不要step
                info_f = list(map(float, [dd.strip('%') for dd in info_f]))
                file_data.append(info_f)
            self.data.append(file_data)
        self.data = np.array(self.data)
        print(self.data)

    def draw_Feat(self):
        self.get_datas_F()
        # 只有两个项目 label_pre 和 select_pre
        raw,col = 1,2
        self.__draw(self.dataf,self.item_name_F,raw,col,'Features')

    def draw_Perf(self): #只考虑perfermance的部分.
        self.get_datas_P()
        raw,col = 2,3
        self.__draw(self.data,self.item_name_P,raw,col,'Performance')

    def __draw(self,data,items,raw,col,fig_name,unit_size=4,hspace=0.3,wspace=0.3,dpi=300):
        plt.figure(figsize=(col*unit_size,raw*unit_size),dpi = dpi)
        plt.subplots_adjust(hspace=hspace,wspace=wspace)
        for idx, item_name in enumerate(items):
            print(idx,item_name)
            plt.subplot(raw,col,idx+1)
            print('self.data',len(data))
            for idx_info,info in enumerate(data): #遍历所有的
                print(info)
                print(len(info))
                max_point = np.argmax(info[:,idx])
                plt.annotate(str(info[max_point][idx]),xy=(max_point,info[max_point][idx]))
                x = np.linspace(1,len(info),len(info))
                plt.plot(x,info[:,idx],label=self.names[idx_info],marker='o')
            plt.xlabel('steps')
            plt.ylabel('value(%)')
            plt.title(item_name)
            plt.legend()
        plt.savefig(osp.join(self.save_path,fig_name),bbox_inches='tight')







if __name__ =='__main__':

    compare_path = [
        'pl_logs/DukeMTMC-VideoReID/atm/atm_vs_0'
    ]
    save_name='0_vs_atm'
    drawer = drawer01(compare_path,save_name)



