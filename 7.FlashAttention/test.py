import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['7.FlashAttention/main.cpp', '7.FlashAttention/flash.cu'], extra_cuda_cflags=['-O2'])

batch_size = 32
n_head = 12
seq_len = 1024
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

res1 = manual_attn(q, k, v)
res2 = minimal_attn.forward(q, k, v)

print(torch.allclose(res1, res2, rtol=0, atol=1e-05))