# from turtle import forward
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

class oneSide_weighted_rgcn(nn.Module):
    def __init__(self, opt):
        super(oneSide_weighted_rgcn, self).__init__()
        self.opt = opt
        self.weight_matrix_item = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        self.weight_matrix_user = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        
    def forward(self, user_feat, item_feat, adj):
        h = self.weight_matrix_item(item_feat)
        b = self.weight_matrix_user(user_feat)
        output = torch.mm(adj.to_dense(), h)
        finalOutput = output + b
        finalOutput = F.relu(finalOutput)
        return finalOutput
