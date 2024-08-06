import torch

## first part : Create tensor

x = torch.randn(3,3, requires_grad=True)
print("Initial Tensor: \n", x)

y = x + 2
z = y * y * 3
out = z.mean()
print("\nOutput\n", out)

## second part : calculate the gradient
out.backward()
print("\nGradient\n", x.grad)