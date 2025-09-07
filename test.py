import torch
from torch.utils.cpp_extension import load

sigmoid_extension = load(
    name="sigmoid",
    sources=["sigmoid.cu"]
)

x = torch.tensor([[1., 2, 3],
                  [4, 5, 6]], device="cuda")
y = sigmoid_extension.sigmoid(x)
print(y)
print(torch.sigmoid(x))