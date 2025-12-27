import triton.language as tl
import triton
import torch
from fla.utils import contiguous, autocast_custom_bwd, autocast_custom_fwd

@triton.jit(do_not_specialize=['T'])
def mean_pooling_fwd_kernel(
    x,
    y,
    cu_seqlens, # varlen
    chunk_iundices, # varlen
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
    