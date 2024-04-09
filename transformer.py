import torch
import torch.nn as nn
from model import ModelArgs, precompute_theta_pos_freq
from encoder import EncoderBlock
from rms_norm import RMSNorm

class Transformer(nn.Module):
    def __init__(self, args:ModelArgs ):
        super().__init__()
        
        # check the vocab size
        assert args.vocab_size != -1, "vocab size must be set"
        
        self.args = args
        self.vocab_size = args.vocab_size
        # number of lyers of the model. [emb-> RMS -> attn -> (emb + RMS) -> ffn (SwiGLU)] X 32 block
        self.n_layers = args.n_layers 
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, args.norm_eps)
        
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # precompute the frequencies of rotary positional encoding
        self.freqs_complex = precompute_theta_pos_freq(self.args.dim //self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)
        
        
    def forward(self, tokens:torch.Tensor, start_pos:int):
        # (B, Seq_len)
        btach_size, seq_len = tokens.shape
        assert seq_len == 1, "only one token is processed"
        
        #(B, seq_len) -> (B, seq_len, dim), dim=4096 for base model
        h = self.tok_embeddings(tokens)
        
        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]
        
        # apply encoders
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        output = self.output(h)
        return output
        