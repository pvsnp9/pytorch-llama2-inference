### pytorch llama2 7B inference code

This code uses llama-2-7b pretrained model for generating text.

install the required packages

```
pip install torch sentencepiece tqdm
```
`senetencepiece` is tokenizer for llama2, similar to the BPE form OpenAI. 

Use the `download.sh` file to download the `llama2` model from meta. [Model Repo](https://github.com/meta-llama/llama)

```chmod +x download.sh```

```./download.sh``` This will donwload the modle, toekenizer into your current directory.

run `ineference.py` to generate text.

Based on your system hardware, allow cuda or use CPU instead if you have larger RAM. I used it locally with 32 GB RAM on 12th Gen Intel(R) Core(TM) i7-12700KF. I was not able to generate text as CPU quits as the RAM reaches limit. I could run it with dtype float16 but torch has not optimizer support for float16, i.e. `addmm_impl_cpu_` error. 

`TBD`
commandline generation/chat with llama2 