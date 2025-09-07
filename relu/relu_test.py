import torch
from torch.utils.cpp_extension import load

relu_extension = load(
    name="relu",
    sources=["./relu/relu_kernel.cu"]
)

x = torch.tensor([[1., 2, 3],
                  [-1, -3, 0]], device="cuda")
y = relu_extension.relu(x)
print(y)
print(torch.relu(y))