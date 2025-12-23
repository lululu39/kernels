import triton.language as tl
import triton
import torch
from fla.utils import contiguous, autocast_custom_bwd, autocast_custom_fwd

# NOTE: for elementwise ops, we only need to flatten (view) the input tensors into 1d arrays

@triton.jit
def sigmoid_fwd_kernel(
    x_ptr,
    y_ptr,
    T,
    BB: tl.constexpr
):
    block_id_b = tl.program_id(0)

    off_b = block_id_b * BB + tl.arange(0, BB)
    mask_b = off_b < T

    x = tl.load(x_ptr + off_b, mask_b)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + off_b, y.to(y_ptr.dtype.element_ty), mask_b)

@triton.jit
def sigmoid_bwd_kernel(
    y_ptr,
    dx_ptr,
    dy_ptr,
    T,
    BB: tl.constexpr
):
    block_id_b = tl.program_id(0)
    off_b = block_id_b * BB + tl.arange(0, BB)
    mask_b = off_b < T
    y = tl.load(y_ptr + off_b, mask_b)
    dy = tl.load(dy_ptr + off_b, mask_b)
    dx = dy * y * (1.0 - y)
    tl.store(dx_ptr + off_b, dx.to(dx_ptr.dtype.element_ty), mask_b)

def sigmoid_fwd(x: torch.Tensor):
    T = x.numel()
    y = torch.zeros_like(x)
    BB = min(32, max(16, triton.next_power_of_2(T)))
    sigmoid_fwd_kernel[(triton.cdiv(T, BB),)](
        x_ptr=x,
        y_ptr=y,
        T=T,
        BB=BB
    )
    return y

def sigmoid_bwd(y: torch.Tensor, dy: torch.Tensor):
    T = y.numel()
    dx = torch.zeros_like(y)
    BB = min(32, max(16, triton.next_power_of_2(T)))
    sigmoid_bwd_kernel[(triton.cdiv(T, BB),)](
        dx_ptr=dx,
        dy_ptr=dy,
        y_ptr=y,
        T=T,
        BB=BB
    )
    return dx

class SigmoidFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = sigmoid_fwd(x)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, dy):
        y, = ctx.saved_tensors
        return sigmoid_bwd(y, dy)

my_sigmoid = SigmoidFunction.apply


@triton.jit
def swish_fwd_kernel(
    x_ptr,
    y_ptr,
    T,
    BB: tl.constexpr
):
    block_id = tl.program_id(0)

    off_b = tl.arange(0, BB) + block_id * BB
    mask_b = off_b < T

    x = tl.load(x_ptr + off_b, mask_b)
    y = x * (1.0 / (1.0 + tl.exp(-x)))
    tl.store(y_ptr + off_b, y.to(y_ptr.dtype.element_ty), mask_b)

@triton.jit
def swish_bwd_kernel(
    x_ptr,
    dx_ptr,
    dy_ptr,
    T,
    BB: tl.constexpr
):
    block_id = tl.program_id(0)

    off_b = tl.arange(0, BB) + block_id * BB
    mask_b = off_b < T

    x = tl.load(x_ptr + off_b, mask_b)
    dy = tl.load(dy_ptr + off_b, mask_b)
    sigmoid_x = (1.0 / (1.0 + tl.exp(-x)))
    dx = dy * ((1.0 - sigmoid_x) * sigmoid_x * x + sigmoid_x)
    tl.store(dx_ptr + off_b, dx.to(dx_ptr.dtype.element_ty), mask_b)

def swish_fwd(
    x: torch.Tensor,
):
    T = x.numel()
    BB = min(32, max(16, triton.next_power_of_2(T)))
    y = torch.empty_like(x)
    swish_fwd_kernel[(triton.cdiv(T, BB),)](
        x_ptr=x,
        y_ptr=y,
        T=T,
        BB=BB
    )
    return y

def swish_bwd(
    x: torch.Tensor,
    dy: torch.Tensor
):
    T = x.numel()
    BB = min(32, max(16, triton.next_power_of_2(T)))
    dx = torch.empty_like(x)
    swish_bwd_kernel[(triton.cdiv(T, BB),)](
        x_ptr=x,
        dx_ptr=dx,
        dy_ptr=dy,
        T=T,
        BB=BB
    )
    return dx

class SwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_fwd(x)

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        return swish_bwd(x, dy)

my_swish = SwishFunction.apply

@triton.jit
def relu_fwd_kernel(
    x,
    y,
    T: tl.constexpr,
    D: tl.constexpr
):
    i_t = tl.program_id(0)
    o = tl.arange(0, D) + i_t * D
    b_x = tl.load(x + o, o < T)
    b_y = tl.where((b_x > 0), b_x, 0)
    tl.store(y + o, b_y.to(y.dtype.element_ty), o < T)

@triton.jit
def relu_bwd_kernel(
    dy,
    dx,
    x,
    T: tl.constexpr,
    D: tl.constexpr
):
    i_t = tl.program_id(0)
    o = tl.arange(0, D) + i_t * D
    b_x = tl.load(x + o, o < T)
    b_dy = tl.load(dy + o, o < T)
    b_dx = tl.where((b_x > 0), b_dy, 0)
    tl.store(dx + o, b_dx.to(dx.dtype.element_ty), o < T)

def relu_fwd(
    x: torch.Tensor
):
    T = x.numel()
    D = min(32, max(16, triton.next_power_of_2(T)))
    y = torch.empty_like(x)
    relu_fwd_kernel[(triton.cdiv(T, D),)](
        x=x,
        y=y,
        T=T,
        D=D
    )
    return y

def relu_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
):
    T = x.numel()
    D = min(32, max(16, triton.next_power_of_2(T)))
    dx = torch.empty_like(x)
    relu_bwd_kernel[(triton.cdiv(T, D),)](
        x=x,
        dy=dy,
        dx=dx,
        T=T,
        D=D
    )
    return dx


class ReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return relu_fwd(x)

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        return relu_bwd(x, dy)

my_relu = ReLUFunction.apply