import os
import codecs
import os.path as osp
import math
import matplotlib.pyplot as plt
import numpy as np
from  pathlib import  Path

class drawer01():
    def __init__(self,compare_path_list,save_name):
        self.compare_path_list = compare_path_list
        self.save_name = save_name
        self.names = [path.split('/')[-1] for path in self.compare_path_list]
        self.save_path = osp.join('/mnt/home/figure_out',self.save_name) # 这个是固定不变的
        if not Path(self.save_path).exists():
            print(self.save_path+'is not exist, creating now!')
            os.mkdir(self.save_path)
        desc_file = open(osp.join(self.save_path,'desc.txt'),'w')
        for path in self.compare_path_list:
            desc_file.write(path+'\n')
        desc_file.close()
        self.item_name_P = 'mAP,Rank-1,Rank-5,Rank-10,Rank-20'.split(',')
        self.item_name_F = 'label_pre,select_pre'.split(',')

        self.draw_Perf()

    def get_datas_P(self): # 取性能数据.
        self.data = []
        for path in self.compare_path_list: # 对每一个实验去数据.
            file_data = []
            file = open(osp.join(path,'data.txt'),'r')
            infos = file.readlines()
            for info in infos:
                info_f = info.strip('\n').split(' ')[1:6]#不要step
                info_f = list(map(float,[dd.strip('%') for dd in info_f]))
                file_data.append(info_f)
            self.data.append(file_data)
        self.data = np.array(self.data)
        print(self.data)

    def draw_Perf(self): #只考虑perfermance的部分.
        self.get_datas_P()
        # item_num = len(self.item_name_P)
        raw,col = 2,3
        unit_size = 4
        plt.figure(figsize=(col*unit_size,raw*unit_size),dpi = 300)
        plt.subplots_adjust(hspace=0.3)
        for idx, item_name in enumerate(self.item_name_P):
            print(idx,item_name)
            plt.subplot(raw,col,idx+1)
            print('self.data',len(self.data))
            for idx_info,info in enumerate(self.data): #遍历所有的
                print(info)
                print(info[:,idx])
                print(len(info))
                max_point = np.argmax(info[:,idx])
                plt.annotate(str(info[max_point][idx]),xy=(max_point,info[max_point][idx]))
                x = np.linspace(1,len(info),len(info))
                plt.plot(x,info[:,idx],label=self.names[idx_info],marker='o')
            plt.xlabel('steps')
            plt.ylabel('value(%)')
            plt.title(item_name)
            plt.legend()
        plt.savefig(osp.join(self.save_path,'Performance'),bbox_inches='tight')




if __name__ =='__main__':
    compare_path = [
        '/mnt/home/pl_logs/DukeMTMC-VideoReID/atm/atm_vs_0',
        '/mnt/home/pl_logs/DukeMTMC-VideoReID/atm/0'
    ]
    save_name='0_vs_atm'
    drawer = drawer01(compare_path,save_name)



