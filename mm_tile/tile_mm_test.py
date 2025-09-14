import torch
from torch.utils.cpp_extension import load

mm_extension = load(
    name="mm",
    sources=["./matrix_multiplication/mm_kernel.cu"],
    verbose=True
)

x = torch.tensor([[1., 2, 3],
                  [3, 1, 5],
                  [7, 3, 0]], device="cuda")
y = torch.tensor([[1., 7, 4],
                  [3, 5, 9],
                  [2, 1, 4]], device="cuda")

print(mm_extension.mm(x ,y))
print(x @ y)