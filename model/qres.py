# baseline implementation of QRes
# paper: Quadratic residual networks: A new class of neural networks for solving forward and inverse problems in physics involving pdes
# link: https://arxiv.org/abs/2101.08366
# code: https://github.com/jayroxis/qres

import torch
import torch.nn as nn

from util import get_clones


class QRes_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(QRes_block, self).__init__()
        self.H1 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.H2 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        x1 = self.H1(x)
        x2 = self.H2(x)
        return self.act(x1*x2 + x1)



class QRes(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(QRes, self).__init__()
        self.N = num_layer-1
        self.inlayer = QRes_block(in_dim, hidden_dim)
        self.layers = get_clones(QRes_block(hidden_dim, hidden_dim), num_layer-1)
        self.outlayer = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x, t):
        src = torch.cat((x,t), dim=-1)
        src = self.inlayer(src)
        for i in range(self.N):
            src = self.layers[i](src)
        src = self.outlayer(src)
        return src