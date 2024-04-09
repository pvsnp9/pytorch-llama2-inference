##!! code referecnce https://github.com/hkproj/pytorch-llama/tree/main
import math
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    # number of query heads 
    n_heads:int = 32 
    # number of heads for the Key and Value
    n_kv_heads: Optional[int] = None
    vocab_size:int = -1 #set when loading the tokenizer
    # hidden dimension of FFN
    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256
    
    norm_eps: float = 1e-5
    # For K V cache
    max_batch_size:int = 32
    max_seq_len:int = 2048
    
    device:str = None
    

#pre-computing the theta pos, #theta value is from paper
def precompute_theta_pos_freq(head_dim:int, seq_len:int, device:str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "dimension should be even number"
    # build the theta params
    # formula: theta_i = 10000 ^ (-2(i-1)/dim) for i = [1, 2, ....dim/2]
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # (head_dim/2)
    theta =  1.0 / (theta ** (theta_numerator/ head_dim)).to(device)
    
    # build the positions  (seq_len)
    m = torch.arange(seq_len, device=device)
    # mulitply each theta by each position using the outer product 
    # shape: (seq_len) outerproduct (head_dim/2) => (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()
    # compute the complex number in polar form c = R * exp(i*m*theta), R = 1 
    # (Seq_len, head_dim/2) -> (Seq_len, head_dim/2)
    freq_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freq_complex

def apply_rotarty_embeddings(x: torch.Tensor, freqs_complex:torch.Tensor, device:str):
    # (B, Seq_len, H, head_dim) -> (B, Seq_len, H, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))    
    # (Seq_len, head_dim/2) -> (1, Seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, H, head_dim/2) * (1, Seq_len, 1, head_dim/2) => (B, seq_len, H, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, H, head_dim/2) -> (B, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, head_dim/2, 2) -> (B, seq_len, H, head_dim/2)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

