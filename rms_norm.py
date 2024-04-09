import torch 
import torch.nn as nn 
from model import ModelArgs

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float):
        super().__init__()
        self.eps = eps
        # gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x:torch.Tensor):
        # (B, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x:torch.Tensor):
        # (dim) * (B, Seq_len, dim) => (B, Seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)