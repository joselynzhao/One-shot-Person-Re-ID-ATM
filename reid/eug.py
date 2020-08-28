import torch
from torch import nn
from reid import models
from reid.trainers import Trainer
from reid.evaluators import extract_features, Evaluator
from reid.dist_metric import DistanceMetric
import numpy as np
from collections import OrderedDict
import os.path as osp
import pickle
from reid.utils.serialization import load_checkpoint
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from reid.exclusive_loss import ExLoss


class EUG():
    def __init__(self, batch_size, num_classes, dataset, l_data, u_data, save_path, embeding_fea_size=1024, dropout=0.5, max_frames=900, momentum=0.5, lamda=0.5):

        self.num_classes = num_classes
        self.save_path = save_path
        self.model_name = 'avg_pool'

        self.l_data = l_data
        self.u_data = u_data
        self.l_label = np.array([label for _,label,_,_ in l_data])
        self.u_label = np.array([label for _,label,_,_ in u_data])


        self.batch_size = batch_size
        self.data_height = 256
        self.data_width = 128
        self.data_workers = 6

        self.data_dir = dataset.images_dir
        self.is_video = dataset.is_video

        self.dropout = dropout
        self.max_frames = max_frames
        self.embeding_fea_size = embeding_fea_size
        self.train_momentum = momentum

        self.lamda = lamda

        if self.is_video:
            self.eval_bs = 1
            self.fixed_layer = True 
            self.frames_per_video = 16
        else:
            self.eval_bs = 256
            self.fixed_layer = False
            self.frames_per_video = 1

            
    def get_dataloader(self, dataset, training=False, is_ulabeled=False) :
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        if training:
            transformer = T.Compose([
                T.RandomSizedRectCrop(self.data_height, self.data_width),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalizer,
            ])
            batch_size = self.batch_size
        else:
            transformer = T.Compose([
                T.RectScale(self.data_height, self.data_width),
                T.ToTensor(),
                normalizer,
            ])
            batch_size = self.eval_bs

        data_loader = DataLoader(
            Preprocessor(dataset, root=self.data_dir, num_samples=self.frames_per_video,
                         transform=transformer, is_training=training, max_frames=self.max_frames),
            batch_size=batch_size, num_workers=self.data_workers,
            shuffle=training, pin_memory=True, drop_last=training)

        current_status = "Training" if training else "Test"
        print("create dataloader for {} with batch_size {}".format(current_status, batch_size))
        return data_loader



    def train(self, train_data, unselected_data, step, loss, epochs=70, step_size=55, init_lr=0.1, dropout=0.5):
        if loss in ["CrossEntropyLoss", 'ExLoss']:
            return self.softmax_train(train_data, unselected_data, step, epochs, step_size, init_lr, dropout, loss)
        else:
            print("{} loss not Found".format(loss))
            raise KeyError
            

    def softmax_train(self, train_data, unselected_data, step, epochs, step_size, init_lr, dropout, loss):

        """ create model and dataloader """
        model = models.create(self.model_name, dropout=self.dropout, num_classes=self.num_classes, 
                              embeding_fea_size=self.embeding_fea_size, classifier = loss, fixed_layer=self.fixed_layer)

        model = nn.DataParallel(model).cuda()

        # the base parameters for the backbone (e.g. ResNet50)
        base_param_ids = set(map(id, model.module.CNN.base.parameters())) 
        base_params_need_for_grad = filter(lambda p: p.requires_grad, model.module.CNN.base.parameters()) 
        new_params = [p for p in model.parameters() if id(p) not in base_param_ids]

        # set the learning rate for backbone to be 0.1 times
        param_groups = [
            {'params': base_params_need_for_grad, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]


        exclusive_criterion = ExLoss(self.embeding_fea_size, len(unselected_data) , t=10).cuda()

        optimizer = torch.optim.SGD(param_groups, lr=init_lr, momentum=self.train_momentum, weight_decay = 5e-4, nesterov=True)

        # change the learning rate by step
        def adjust_lr(epoch, step_size):
            
            use_unselcted_data = True
            lr = init_lr / (10 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)
            if epoch >= step_size:
                use_unselcted_data = False
                # print("Epoch {}, CE loss, current lr {}".format(epoch, lr))
            return use_unselcted_data


        s_dataloader = self.get_dataloader(train_data, training=True, is_ulabeled=False)
        u_dataloader = self.get_dataloader(unselected_data, training=True, is_ulabeled=True)


        """ main training process """
        trainer = Trainer(model, exclusive_criterion, fixed_layer=self.fixed_layer, lamda = self.lamda)
        for epoch in range(epochs):
            use_unselcted_data = adjust_lr(epoch, step_size)
            trainer.train(epoch, s_dataloader, u_dataloader, optimizer, use_unselcted_data, print_freq=len(s_dataloader)//2)

        ckpt_file = osp.join(self.save_path,  "step_{}.ckpt".format(step))
        torch.save(model.state_dict(), ckpt_file)
        self.model = model

    def get_feature(self, dataset):
        dataloader = self.get_dataloader(dataset, training=False)
        features,_ = extract_features(self.model, dataloader)
        features = np.array([logit.numpy() for logit in features.values()])
        return features

    def estimate_label(self):

        # extract feature 
        u_feas = self.get_feature(self.u_data)
        l_feas = self.get_feature(self.l_data)
        print("u_features", u_feas.shape, "l_features", l_feas.shape)

        u_size = u_feas.shape[0]

        dists_u_u = self.euclidean_dist(u_feas,u_feas)
        #dists_u_l = self.euclidean_dist(u_feas,l_feas)
        dists_l_u = self.euclidean_dist(l_feas,u_feas)
        num_correct_pred = 0

        l_u_index = np.argsort(dists_l_u,axis=1)
        u_u_index = np.argsort(dists_u_u,axis=1)


        dist_single = self.euclidean_dist(u_feas,l_feas)
        dist_kcross = np.zeros((u_feas.shape[0], l_feas.shape[0]), dtype=np.float32)

        print('************compute k_cross dists**************')
        for idx, u_fea in enumerate(tqdm(u_feas)):
            dist_k = self.cross_knearst_dist(dists_l_u,dists_u_u,l_u_index,u_u_index,idx)
            dist_kcross[idx,:] = dist_k

        #将两种距离规范化到[0,1]之间
        dist_single = (dist_single - dist_single.min())/(dist_single.max() - dist_single.min())
        dist_kcross = (dist_kcross - dist_kcross.min())/ (dist_kcross.max() - dist_kcross.min())

        dists = 0.5 * dist_single + dist_kcross * 0.5
        index_min = np.argmin(dists,axis=1)
        dists_min = np.min(dists,axis=1)

        scores = - dists_min
        labels = index_min
        print('*************generate labels ***************')
        for idx in tqdm(range(u_size)):
            if self.u_label[idx] == labels[idx]:
                num_correct_pred +=1

        print("Label predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
            num_correct_pred, u_feas.shape[0], num_correct_pred/u_feas.shape[0]))

        return labels, scores

    def cross_knearst_dist(self,dists_l_u,dists_u_u,l_u_index,u_u_index,idx,k=3):

        l_size = dists_l_u.shape[0]

        l_nearst_u = l_u_index[:,0:k-1]     #[751,k-1]
        u_nearst_u = u_u_index[idx][0:k] #u始终距离自己最近，所以需要去掉0号位  [k]
        #dists = np.zeros(l_nearst_u.shape[0],dtype=np.float32)

        #use time :0.00015s
        dist_l = np.sum(dists_l_u[:,u_nearst_u],axis=1)  #[751]
        dist_u = np.zeros(l_size,dtype=np.float32)
        #use time:0.0015s

        for u_u_label in u_nearst_u:
            dist_u += np.sum(dists_u_u[l_nearst_u,u_u_label],axis=1)
        dists = (dist_l + dist_u)/(k**2)

        return dists    #[751]

    def euclidean_dist(self,x,y):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        m, n = x.size(0),y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)

        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
        return dist.numpy()


    def select_top_data(self, pred_score, nums_to_select):
        v = np.zeros(len(pred_score))
        index = np.argsort(-pred_score)
        for i in range(nums_to_select):
            v[index[i]] = 1
        return v.astype('bool')



    def generate_new_train_data(self, sel_idx, pred_y,step=0,path=None):
        """ generate the next training data """

        seletcted_data = []
        unselected_data = []
        correct, total = 0, 0
        class_count = np.zeros(len(self.l_data),dtype=np.int)
        row_set = [[] for i in range(len(self.l_data))]
        for i, flag in enumerate(sel_idx):
            if flag: # if selected
                seletcted_data.append([self.u_data[i][0], int(pred_y[i]), self.u_data[i][2], self.u_data[i][3]])
                total += 1
                class_count[int(pred_y[i])] += 1
                row_set[int(pred_y[i])].append(i)
                if self.u_label[i] == int(pred_y[i]):
                    correct += 1
            else:
                unselected_data.append(self.u_data[i])
        acc = correct / total

        new_train_data = self.l_data + seletcted_data
        print("selected pseudo-labeled data: {} of {} is correct, accuracy: {:0.4f}  new train data: {}".format(
                correct, len(seletcted_data), acc, len(new_train_data)))
        print("Unselected Data:{}".format(len(unselected_data)))

        self.curve_generate(class_count,step,path)
        self.top_k_pesuo(class_count,row_set,step)

        return new_train_data, unselected_data

    def curve_generate(self, array, step, path):
        length = len(array)
        index = np.array([i for i in range(length)])
        value = array
        if not os.path.exists(path):
            os.makedirs(path)
        plt.plot(index, value)
        plt.savefig(os.path.join(path, 'step_{}.jpg'.format(step)))

        # 查看topk中选择的样本数和选择的准确率

    def top_k_pesuo(self, array, row_set, step):
        k = 10
        index_array = np.argsort(-array)[0:k]
        index_num = [array[i] for i in index_array]

        index_array_min = np.argsort(array)[0:k]
        index_num_min = [array[i] for i in index_array_min]
        print('**********************step_{}, top_{}***********************'.format(step, k))
        for i in range(k):
            correct = 0
            cla, num = index_array[i], index_num[i]
            for idx in range(num):
                if self.u_label[row_set[cla][idx]] == cla:
                    correct += 1
            print('top_{} is id {},selected {} sample,{} is corrected,accurary is {:.1%}'.format(i + 1, index_array[i],
                                                                                                 index_num[i], correct,
                                                                                                 correct / (
                                                                                                         num + 1e-12)))

        print('****************************top_{}***********************'.format(k))

        print('**********************step_{}, down_{}***********************'.format(step, k))
        for i in range(k):
            correct = 0
            cla, num = index_array_min[i], index_num_min[i]
            for idx in range(num):
                if self.u_label[row_set[cla][idx]] == cla:
                    correct += 1
            print('down_{} is id {},selected {} sample,{} is corrected,accurary is {:.1%}'.format(i + 1,
                                                                                                  index_array_min[i],
                                                                                                  index_num_min[i],
                                                                                                  correct, correct / (
                                                                                                          num + 1e-12)))

        print('****************************down_{}***********************'.format(k))

    def resume(self, ckpt_file, step):
        print("continued from step", step)
        model = models.create(self.model_name, dropout=self.dropout, num_classes=self.num_classes, is_output_feature = True)
        self.model = nn.DataParallel(model).cuda()
        self.model.load_state_dict(load_checkpoint(ckpt_file))

    def evaluate(self, query, gallery):
        test_loader = self.get_dataloader(list(set(query) | set(gallery)), training = False)
        param = self.model.state_dict()
        del self.model
        model = models.create(self.model_name, dropout=self.dropout, num_classes=self.num_classes, is_output_feature = True)
        self.model = nn.DataParallel(model).cuda()
        self.model.load_state_dict(param)
        evaluator = Evaluator(self.model)
        evaluator.evaluate(test_loader, query, gallery)



"""
    Get init split for the input dataset.
"""
def get_init_shot_in_cam1(dataset, load_path, init, seed=0):

    init_str = "one-shot" if init == -1 else "semi {}".format(init)

    np.random.seed(seed)
    random.seed(seed)

    # if previous split exists, load it and return
    if osp.exists(load_path):
        with open(load_path, "rb") as fp:
            dataset = pickle.load(fp)
            label_dataset = dataset["label set"]
            unlabel_dataset = dataset["unlabel set"]

        print("  labeled  |   N/A | {:8d}".format(len(label_dataset)))
        print("  unlabel  |   N/A | {:8d}".format(len(unlabel_dataset)))
        print("\nLoad one-shot split from", load_path)
        print(init_str + "\n")
        return label_dataset, unlabel_dataset

    label_dataset = []
    unlabel_dataset = []

    if  init_str == "one-shot":
        # dataset indexed by [pid, cam]
        dataset_in_pid_cam = [[[] for _ in range(dataset.num_cams)] for _ in range(dataset.num_train_ids) ]
        for index, (images, pid, camid, videoid) in enumerate(dataset.train):
            dataset_in_pid_cam[pid][camid].append([images, pid, camid, videoid])

        # generate the labeled dataset by randomly selecting a tracklet from the first camera for each identity
        for pid, cams_data  in enumerate(dataset_in_pid_cam):
            for camid, videos in enumerate(cams_data):
                if len(videos) != 0:
                    selected_video = random.choice(videos)
                    break
            label_dataset.append(selected_video)
        assert len(label_dataset) == dataset.num_train_ids
        labeled_videoIDs =[vid for _, (_,_,_, vid) in enumerate(label_dataset)]

    else:
        # dataset indexed by [pid]
        dataset_in_pid = [ [] for _ in range(dataset.num_train_ids) ]
        for index, (images, pid, camid, videoid) in enumerate(dataset.train):
            dataset_in_pid[pid].append([images, pid, camid, videoid])

        for pid, pid_data  in enumerate(dataset_in_pid):
            k = int(np.ceil(len(pid_data) * init))  # random sample ratio
            selected_video = random.sample(pid_data, k)
            label_dataset.extend(selected_video)
        
        labeled_videoIDs =[vid for _, (_,_,_, vid) in enumerate(label_dataset)]

    # generate unlabeled set
    for (imgs, pid, camid, videoid) in dataset.train:
        if videoid not in labeled_videoIDs:
            unlabel_dataset.append([imgs, pid, camid, videoid])

    with open(load_path, "wb") as fp:
        pickle.dump({"label set":label_dataset, "unlabel set":unlabel_dataset}, fp)


    print("  labeled    | N/A | {:8d}".format(len(label_dataset)))
    print("  unlabeled  | N/A | {:8d}".format(len(unlabel_dataset)))
    print("\nCreate new {} split and save it to {}".format(init_str, load_path))
    return label_dataset, unlabel_dataset
