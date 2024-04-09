import torch
import torch.nn as nn
from model import ModelArgs
from attention import SelfAttention
from feed_forward import FeedForward
from rms_norm import RMSNorm

'''


params: {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}
'''

class EncoderBlock(nn.Module):
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        # 4096 //32 = 128
        self.head_dim = args.dim // args.n_heads 
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # norm before attention 
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # norm after attention
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex:torch.Tensor):
        # (B, seq_len, dim) + (B, seq_len, Dim) => (B, seq_len, dim)
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        
        out = h + self.feed_forward(self.ffn_norm(h))
        return out