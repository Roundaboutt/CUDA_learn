import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="reduce_lib",
    sources=["2.reduce/test/reduce.cu"],
)

x = torch.tensor([[1.,2],
                  [3,4]]).cuda()
res1 = torch.sum(x)
res2 = lib.forward(x)

print(res1)
print(res2)