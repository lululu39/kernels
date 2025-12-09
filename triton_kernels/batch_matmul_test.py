import torch
from batch_matmul import batched_matmul_naive
from batch_matmul import batched_matmul

# B, M, N, P = 4, 5, 6, 7
B, M, N, P = 32, 32, 128, 64
# B, M, N, P = 2, 2, 2, 2
x = torch.randn(B, M, N, requires_grad=True, device="cuda")
y = torch.randn(B, N, P, requires_grad=True, device="cuda")

dz = torch.ones(B, M, P)

# out1 = batched_matmul_naive(x, y)
out1 = batched_matmul(x, y)
loss1 = out1.sum()
loss1.backward()
dx1 = x.grad.clone()
dy1 = y.grad.clone()

x.grad.zero_()
y.grad.zero_()
out2 = torch.matmul(x, y)
loss2 = out2.sum()
loss2.backward()
dx2 = x.grad.clone()
dy2 = y.grad.clone()

print("Forward equal:", torch.allclose(out1, out2, atol=1e-5))

print("dx equal:", torch.allclose(dx1, dx2, atol=1e-5))
print("dy equal:", torch.allclose(dy1, dy2, atol=1e-5))


# training loop
x = torch.randn(B, M, N, requires_grad=True)
y = torch.randn(B, N, P, requires_grad=True)

label = torch.randn(B, M, P)

# for i in range(100):
#     # out = batched_matmul_naive(x, y)
#     out = batched_matmul(x, y)
#     loss = (out - label).pow(2).mean()
#     loss.backward()
#     print(f"Iter {i}: loss={loss.item():.4f}")
#     with torch.no_grad():
#         x -= 1. * x.grad
#         y -= 1. * y.grad
#         x.grad.zero_()
#         y.grad.zero_()