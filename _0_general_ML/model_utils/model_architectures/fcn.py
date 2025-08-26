import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    
    def __init__(self, layer_sizes=[784, 50, 25, 10]):
        super().__init__()
        self.layer_sizes = layer_sizes
        w_s = []
        b_s = []
        for i in range(1, len(layer_sizes)):
            w_s.append(nn.Parameter(torch.randn((layer_sizes[i-1], layer_sizes[i]))))
            b_s.append(nn.Parameter(torch.zeros((layer_sizes[i]))))
        self.w_s = nn.ParameterList(w_s)
        self.b_s = nn.ParameterList(b_s)
        
    def relu(self, x):
        return torch.abs(x)+x
    
    def forward(self, x):
        # Switch from activation maps to vectors
        out = x.view(-1, self.layer_sizes[0])
        for i in range(len(self.w_s)-1):
            out = self.relu(torch.matmul(out, self.w_s[i])+self.b_s[i])
        out = torch.matmul(out, self.w_s[-1])+self.b_s[-1]
        return out
    
    