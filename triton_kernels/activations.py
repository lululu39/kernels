import triton.language as tl
import triton
import torch
from fla.utils import contiguous, autocast_custom_bwd, autocast_custom_fwd

@triton.jit
def sigmoid_fwd_kernel(
    x_ptr,
    y_ptr,
    B,
    T: tl.constexpr,
    D: tl.constexpr,
    BB: tl.constexpr
):
    block_id_b = tl.program_id(0)

    off_b = block_id_b * BB

    # NOTE: block shape elements must be of tl.constexpr type

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(B, T, D),
        strides=(T * D, D, 1),
        offsets=(off_b, 0, 0),
        block_shape=(BB, T, D),
        order=(0, 1, 2)
    )
    y_block_ptr = tl.make_block_ptr(
        y_ptr,
        shape=(B, T, D),
        strides=(T * D, D, 1),
        offsets=(off_b, 0, 0),
        block_shape=(BB, T, D),
        order=(0, 1, 2)
    )
    x = tl.load(x_block_ptr, boundary_check=(0, 1, 2))
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_block_ptr, y.to(y_ptr.dtype.element_ty), boundary_check=(0, 1, 2))

@triton.jit
def sigmoid_bwd_kernel(
    y_ptr,
    dx_ptr,
    dy_ptr,
    B,
    T: tl.constexpr,
    D: tl.constexpr,
    BB: tl.constexpr
):
    block_id_b = tl.program_id(0)
    off_b = block_id_b * BB
    y_block_ptr = tl.make_block_ptr(
        y_ptr,
        shape=(B, T, D),
        strides=(T * D, D, 1),
        offsets=(off_b, 0, 0),
        block_shape=(BB, T, D),
        order=(0, 1, 2)
    )
    dy_block_ptr = tl.make_block_ptr(
        dy_ptr,
        shape=(B, T, D),
        strides=(T * D, D, 1),
        offsets=(off_b, 0, 0),
        block_shape=(BB, T, D),
        order=(0, 1, 2)
    )
    dx_block_ptr = tl.make_block_ptr(
        dx_ptr,
        shape=(B, T, D),
        strides=(T * D, D, 1),
        offsets=(off_b, 0, 0),
        block_shape=(BB, T, D),
        order=(0, 1, 2)
    )
    y = tl.load(y_block_ptr, boundary_check=(0, 1, 2))
    dy = tl.load(dy_block_ptr, boundary_check=(0, 1, 2))
    dx = dy * y * (1.0 - y)
    tl.store(dx_block_ptr, dx.to(dx_ptr.dtype.element_ty), boundary_check=(0, 1, 2))

def sigmoid_fwd(x: torch.Tensor):
    B, T, D, = x.shape
    y = torch.zeros_like(x)
    BB = min(32, max(16, triton.next_power_of_2(B)))
    sigmoid_fwd_kernel[(triton.cdiv(B, BB),)](
        x_ptr=x,
        y_ptr=y,
        B=B,
        T=T,
        D=D,
        BB=BB
    )
    return y

def sigmoid_bwd(y: torch.Tensor, dy: torch.Tensor):
    B, T, D, = y.shape
    dx = torch.zeros_like(y)
    BB = min(32, max(16, triton.next_power_of_2(B)))
    sigmoid_bwd_kernel[(triton.cdiv(B, BB),)](
        dx_ptr=dx,
        dy_ptr=dy,
        y_ptr=y,
        B=B,
        T=T,
        D=D,
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



