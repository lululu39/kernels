import torch
import torch.nn.functional as F
from activations import my_relu, my_sigmoid, my_swish, my_softplus, my_gelu
from fla.ops.utils import mean_pooling
from pooling import my_mean_pooling

def test_activation(fn_triton, fn_torch, name):
    x = torch.randn(1000, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32, requires_grad=True)
    y_triton = fn_triton(x)
    y_torch = fn_torch(x)

    forward_diff = (y_triton - y_torch).abs().max().item()
    print(f"{name} forward diff: {forward_diff}")
    assert forward_diff < 1e-5, f"{name} forward diff {forward_diff} exceeds 1e-5"

    # backward test
    grad = torch.randn_like(y_triton)
    y_triton.backward(grad, retain_graph=True)
    grad_triton = x.grad.clone()
    x.grad.zero_()
    y_torch.backward(grad, retain_graph=True)
    grad_torch = x.grad.clone()
    backward_diff = (grad_triton - grad_torch).abs().max().item()
    print(f"{name} backward diff: {backward_diff}")
    assert backward_diff < 1e-5, f"{name} backward diff {backward_diff} exceeds 1e-5"
    print(f"{name} test passed! forward_diff={forward_diff:.2e}, backward_diff={backward_diff:.2e}")

def test_pooling(batch=2, seq=64, heads=4, dim=32, chunk_size=8, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float32, requires_grad=True)

    # forward
    y_my = my_mean_pooling(x, chunk_size)
    y_fla = mean_pooling(x, chunk_size)

    fwd_diff = (y_my - y_fla).abs().max().item()
    print(f"Pooling forward diff: {fwd_diff}")
    assert fwd_diff < 1e-5, f"Pooling forward diff {fwd_diff} exceeds 1e-5"

    # backward
    grad = torch.randn_like(y_my)
    y_my.backward(grad, retain_graph=True)
    grad_my = x.grad.clone()
    x.grad.zero_()
    y_fla.backward(grad, retain_graph=True)
    grad_fla = x.grad.clone()
    bwd_diff = (grad_my - grad_fla).abs().max().item()
    print(f"Pooling backward diff: {bwd_diff}")
    assert bwd_diff < 1e-5, f"Pooling backward diff {bwd_diff} exceeds 1e-5"

    print(f"Pooling test passed! forward_diff={fwd_diff:.2e}, backward_diff={bwd_diff:.2e}")


def test_pooling_varlen(device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    seqs = [13, 7, 5]
    total = sum(seqs)
    batch = 1
    heads = 4
    dim = 32
    chunk_size = 8
    x = torch.randn(batch, total, heads, dim, device=device, dtype=torch.float32, requires_grad=True)
    cu_seqlens = torch.tensor([0, seqs[0], seqs[0] + seqs[1], total], dtype=torch.long, device=device)

    # forward
    y_my = my_mean_pooling(x, chunk_size, cu_seqlens)
    y_fla = mean_pooling(x, chunk_size, cu_seqlens)
    fwd_diff = (y_my - y_fla).abs().max().item()
    print(f"Pooling varlen forward diff: {fwd_diff}")
    assert fwd_diff < 1e-5, f"Pooling varlen forward diff {fwd_diff} exceeds 1e-5"

    # backward
    grad = torch.randn_like(y_my)
    y_my.backward(grad, retain_graph=True)
    grad_my = x.grad.clone()
    x.grad.zero_()
    y_fla.backward(grad, retain_graph=True)
    grad_fla = x.grad.clone()
    bwd_diff = (grad_my - grad_fla).abs().max().item()
    print(f"Pooling varlen backward diff: {bwd_diff}")
    assert bwd_diff < 1e-5, f"Pooling varlen backward diff {bwd_diff} exceeds 1e-5"

    print(f"Pooling varlen test passed! forward_diff={fwd_diff:.2e}, backward_diff={bwd_diff:.2e}")

if __name__ == "__main__":
    # test_activation(my_relu, torch.relu, "ReLU")
    # test_activation(my_sigmoid, torch.sigmoid, "Sigmoid")
    # test_activation(my_swish, lambda x: x * torch.sigmoid(x), "Swish")
    # test_activation(my_softplus, F.softplus, "Softplus")
    # test_activation(my_gelu, F.gelu, "GELU")
    # test_pooling()
    test_pooling_varlen()