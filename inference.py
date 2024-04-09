from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from model import ModelArgs
from transformer import Transformer

class LlaMa:
    def __init__(self, model:Transformer, tokenizer:SentencePieceProcessor, model_args:ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
        
        
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path:str, load_model: bool, max_seq_len:int, max_batch_size:int, device:str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints found"
            checkpoint_path = checkpoints[0]
            print(f"loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print(f"loaded checkpoint in {time.time()- prev_time:.2f}s")
            prev_time = time.time()
            
        with open(Path(checkpoints_dir) / "params.json", "r")as f:
            params = json.loads(f.read())
            # construct model args
            model_args = ModelArgs(
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                device=device,
                **params
            )
            
            # load tokenizer
            tokenizer = SentencePieceProcessor()
            tokenizer.Load(tokenizer_path)
            model_args.vocab_size = tokenizer.vocab_size()
            
            if device == "cuda": torch.set_default_dtype(torch.float32)
            else: torch.set_default_dtype(torch.float32)
            
            model = Transformer(model_args).to(device)
            
            # load the model
            if load_model:
                # we have external var for freqs complex which is not default model. Hence, removing it.
                del checkpoint['rope.freqs']
                model.load_state_dict(checkpoint, strict=True)
                print(f"loaded the model state dict {(time.time() - prev_time):.2f}s")
            
            return  LlaMa(model, tokenizer, model_args)
        
    def text_completion(self, prompts: list[str], temperature:float = 0.6, top_p:float= 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None: max_gen_len = self.args.max_seq_len -1
        # convert each prompts into tokens
        promp_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # checking batch_size
        batch_size = len(promp_tokens)
        assert batch_size <= self.args.max_batch_size, "Prompts are lerger than the desired batch size, for this example"
        max_prompt_len = max(len(prompt) for prompt in promp_tokens)
        assert max_prompt_len <= self.args.max_seq_len, "Prompts are longer than desired sequence length"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        
        # create a list of tokens that will contain generated token along with initial prompt tokens 
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(promp_tokens):
            # populate the initial tokens with the prompt tokens
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_token_mask = tokens != pad_id
                
        for current_pos in tqdm(range(1, total_len), desc="Generating tokens"):
            with torch.no_grad():
                logits = self.model(tokens[:, current_pos-1:current_pos], current_pos)
                if temperature > 0:
                    # we apply temptrature before the softmaz
                    probs = torch.softmax(logits[:, -1]/temperature, dim=-1)
                    next_token = self._sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # replace only if its a pad token
            next_token = torch.where(prompt_token_mask[:, current_pos], tokens[:, current_pos], next_token)
            tokens[:, current_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_token_mask[:, current_pos]) & (next_token == self.tokenizer.eos_id)
            
            if all(eos_reached): break
        out_tokens = []
        out_text = []
        
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # pluck upto the eos token if present 
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        
        return(out_tokens, out_text)
        
    
    def _sample_top(self, probs, p):
        #(B, vocab_size)    
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        #(B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p 
        probs_sort[mask] = 0.0
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
            
            
##################################################### Inference the model #####################################################
'''
Techniques to predict next token:
- Greedy: Just select the tken with with highest probab (T=1). If the initial token gets wrong the rest of the token will also likely be wrong.
- Beamsearch: Search Top-K items. lets K=2. create a prompt with K new tokens and compute cumulative score. Repeat the process for all K. Performs better but slow.
- Temperature: (hyper-parameter) The idea is to scale logits before softmax. Low temp makes modle more confident and vice versa. The gap of logits in low temp is high.
- Random Sampling: Chance of picking nonsense token, even with low probs
- Top K: low probability can also be included into top k
- Top P: pick only with threshold probability. lets say, P >= 0.5
'''




if __name__ == "__main__":
    torch.manual_seed(9)
    allow_cuda = False
    
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"
    
    prompts = [
        "Simply put, the theory of relativity states that ",
    ]

    model = LlaMa.build(
        checkpoints_dir="llama-2-7b/",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )
    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 40)