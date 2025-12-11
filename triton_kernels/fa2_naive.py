import triton.language as tl
import triton
import torch


@triton.jit
def flash_attention_naive_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    B,
    H, # we assume no group of query are used in naive fa2
    T,
    K,
    V, # 
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # qk [B, T, H, K]
    # v [B, T, H, V]

    # parallize in three dimensions
    # 1. value hidden dimension V
    # 2. sequence dimension
    # 3. batch * head

    # NOTE: we are not supposed to paralleize over the K dimension, so BK is just for better triton loading,

    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t, 0), (BT, K))

