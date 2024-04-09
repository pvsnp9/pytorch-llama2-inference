import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ModelArgs, apply_rotarty_embeddings

class SelfAttention(nn.Module):
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        # number of heads for the key and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # number ofheads for query
        self.n_heads_q = args.n_heads
        # how many times the heads of keys and queries should be repeated to match the head of queries 
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # dimension of each head 
        self.head_dim = args.dim // args.n_heads
        
        
        self.wq  = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk  = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv  = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo  = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        
    def forward(self, x:torch.Tensor, start_pos: int, freqs_complex:torch.Tensor):
        batch_size, seq_len, _ = x.shape # (B, 1, Dim)
        # (B, 1, Dim) => ((B, 1, H_Q, head_dim))
        xq = self.wq(x)
        # (B, 1, Dim) => ((B, 1, H_KV, head_dim))
        xk = self.wk(x)
        xv = self.wv(x)
        
        # (B, 1, H_Q * Head_Dim) => (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) => (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # rotary positional encodings   # (B, 1, H_Q * Head_Dim) => (B, 1, H_Q, Head_Dim)
        xq = apply_rotarty_embeddings(xq, freqs_complex, device=x.device)
        # (B, 1, H_KV * Head_Dim) => (B, 1, H_KV, Head_Dim)
        xk = apply_rotarty_embeddings(xk, freqs_complex, device=x.device)
        
        # caching
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv
        
        # extract all the cached keys and values at current pos
        # (B, seq_len_kv, h_kv, dim)
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        # (B, 1 (seq_len), h_q, head_dim) -> (B, h_q, 1, head_dim)
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        valeus = values.transpose(1,2)
        
        # (B, h_q, 1, head_dim) @ (B, h_q, head_dim, seq_len_kv) => (B, h_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        #(B, h_q, 1, seq_len_kv) @ (B, H_Q, seq_len_kv, head_dim) => (B, H_Q, 1, head_dim)
        out = torch.matmul(scores, valeus)
        # concatenate multihead (B, H_Q, 1, head_dim) -> (B, 1, H_Q, head_dim) => (B, 1, dim)
        out = (out.transpose(1,2).contiguous().view(batch_size, seq_len, -1))
        #(B, 1, dim) => (B, 1, dim)
        return self.wo(out)
        
        
        
        

def repeat_kv(x:torch.Tensor, n_rep:int):
    b, s, n_kv_heads, head_dim = x.shape
    if n_rep == 1: return x
    else:
        return(
            # (B, seq_len, n_kv_heads, 1, head_dim)
            x[:,:,:, None, :].expand(b, s, n_kv_heads, n_rep, head_dim).reshape(b,s,n_kv_heads *n_rep, head_dim)
        )