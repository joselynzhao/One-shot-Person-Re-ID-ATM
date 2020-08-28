import torch
import torch.nn.functional as F
from torch import nn, autograd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def euclidean_dist(x, y):
    x = torch.unsqueeze(x, 1).to(device)
    y = torch.unsqueeze(y, 0).to(device)
    x_y = x - y
    dist = torch.pow(x_y, 2).sum(dim=-1) + 1e-12
    dist = dist.sqrt()
    return dist

def dist_loss(dists):
    #N = dists.size(0)
    loss = torch.tensor(0.).to(device)
    for idx,dist in enumerate(dists):
        dist_diff = (torch.pow(dist - dists, 2)+1e-12).sqrt().sum(dim=-1)
        loss +=dist_diff
    return loss

class DistRegularizeLoss(nn.Module):
    def __init__(self):
        super(DistRegularizeLoss, self).__init__()

    def forward(self,ide_feats,u_feats):
        dists = euclidean_dist(ide_feats,u_feats)
        dists = dists.mean(dim=-1)
        loss = dist_loss(dists)
        return loss