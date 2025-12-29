import triton.language as tl
import triton
import torch
from fla.utils import contiguous, autocast_custom_bwd, autocast_custom_fwd
from typing import Optional

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BT']
)
@triton.jit(do_not_specialize=['T'])
def mean_pooling_fwd_kernel(
    x,
    y,
    cu_seqlens, # varlen
    chunk_indices, # varlen
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_d, i_nt, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    NT = tl.cdiv(T, BT)

    bos = i_b * T
    bos_n = i_b * NT

    p_x = tl.make_block_ptr(x + (bos * H + i_h) * D, (T, D), (H * D, 1), (i_nt * BT, i_d * BD), (BT, BD), (1, 0))
    p_y = tl.make_block_ptr(y + ((bos_n + i_nt) * H + i_h) * D, (D,), (1,), (i_d * BD,), (BD,), (0,))

    b_x = tl.load(p_x, boundary_check=(0,1)).to(tl.float32) # NOTE: cast here
    b_y = tl.sum(b_x, axis=0) / min(BT, T - i_nt * BT)

    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0,))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BT']
)
@triton.jit(do_not_specialize=['T'])
def mean_pooling_bwd_kernel(
    dy,
    dx,
    cu_seqlens, # varlen
    chunk_indices, # varlen
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_d, i_nt, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    NT = tl.cdiv(T, BT)

    bos = i_b * T
    bos_n = i_b * NT

    p_dx = tl.make_block_ptr(dx + (bos * H + i_h) * D, (T, D), (H * D, 1), (i_nt * BT, i_d * BD), (BT, BD), (1, 0))
    p_dy = tl.make_block_ptr(dy + ((bos_n + i_nt) * H + i_h) * D, (D,), (1,), (i_d * BD,), (BD,), (0,))

    b_dy = tl.load(p_dy, boundary_check=(0,))

    b_dx = b_dy / tl.full([BT], min(BT, T - i_nt * BT), dtype=tl.float32)[:, None] # [BT, BD]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0,1))

def mean_pooling_fwd(
    x: torch.Tensor,
    chunk_size: int,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, D = x.shape
    BT = chunk_size
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    chunk_indices = None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    y = torch.empty(B, NT, H, D, dtype=x.dtype, device=x.device)
    def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B * H) # NOTE: this function
    # NOTE: when using meta, no need to specify in function
    mean_pooling_fwd_kernel[grid](
        x=x,
        y=y,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        D=D,
        BT=BT,
    )
    return y

def mean_pooling_bwd(
    dy: torch.Tensor,
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    cu_seqlens: Optional[torch.LongTensor] = None,  
):
    B, T, H, D = batch_size, seq_len, *dy.shape[-2:]
    NT = dy.shape[1]
    BT = chunk_size
    chunk_indices = None
    dx = torch.empty(B, T, H, D, dtype=dy.dtype, device=dy.device)
    def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B * H) # NOTE: this function
    mean_pooling_bwd_kernel[grid](
        dy=dy,
        dx=dx,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        D=D,
        BT=BT,
    )
    return dx


class MeanPoolingFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, x, chunk_size, cu_seqlens):
        ctx.dtype = x.dtype
        ctx.batch_size = x.shape[0]
        ctx.seq_len = x.shape[1]
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        return mean_pooling_fwd(
            x=x,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens
        )

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, dy):
        batch_size = ctx.batch_size
        seq_len = ctx.seq_len
        chunk_size = ctx.chunk_size
        cu_seqlens = ctx.cu_seqlens
        return mean_pooling_bwd(
            dy=dy,
            batch_size=batch_size,
            seq_len=seq_len,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens
        ), None, None

def my_mean_pooling(
    x: torch.Tensor,
    chunk_size: int,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    # NOTE: format of tensor is always seq first
    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {x.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
    
    y = MeanPoolingFunction.apply(x, chunk_size, cu_seqlens)
    return y