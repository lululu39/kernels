import torch
import torch.nn.functional as F
from activations import my_relu, my_sigmoid, my_swish, my_softplus

def test_activation(fn_triton, fn_torch, name):
    x = torch.randn(1000, device='cuda', dtype=torch.float32, requires_grad=True)
    y_triton = fn_triton(x)
    y_torch = fn_torch(x)
    forward_diff = (y_triton - y_torch).abs().max().item()
    assert forward_diff < 1e-5, f"{name} forward diff {forward_diff} exceeds 1e-5"

    # backward test
    grad = torch.randn_like(y_triton)
    y_triton.backward(grad, retain_graph=True)
    grad_triton = x.grad.clone()
    x.grad.zero_()
    y_torch.backward(grad, retain_graph=True)
    grad_torch = x.grad.clone()
    backward_diff = (grad_triton - grad_torch).abs().max().item()
    assert backward_diff < 1e-5, f"{name} backward diff {backward_diff} exceeds 1e-5"
    print(f"{name} test passed! forward_diff={forward_diff:.2e}, backward_diff={backward_diff:.2e}")

if __name__ == "__main__":
    test_activation(my_relu, torch.relu, "ReLU")
    test_activation(my_sigmoid, torch.sigmoid, "Sigmoid")
    test_activation(my_swish, lambda x: x * torch.sigmoid(x), "Swish")
    test_activation(my_softplus, F.softplus, "Softplus")