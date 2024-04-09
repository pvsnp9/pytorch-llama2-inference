import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ModelArgs

class FeedForward(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        
        if args.ffn_dim_multiplier is not None: hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # round the hidden_dim to the nearest multiple of the multiple parameter
        # eg. 8 /6 => 8 + (6-1) // 6 = 2 * 6 = 12 first multiple i.e. bigger or eq to hidden
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of -1) // args.multiple_of) 
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        
    def forward(self, x:torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))