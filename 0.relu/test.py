import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="relu_lib",
    sources=["0.relu/relu.cu"],
)

x = torch.tensor([[1., 2, 3],
                  [-1, -3, 0]], device="cuda")
y = lib.forward(x)
print(y)
print(torch.relu(x))