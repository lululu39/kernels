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
    BT: tl.constexpr, # for q partition
    BS: tl.constexpr, # for k and v partition
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # qk [B, H, T, K]
    # v [B, H, T, V]
    # o [B, H, T, V]
    # lse [B, H, T]

    # parallize in three dimensions
    # 1. batch * head
    # 2. sequence dimension
    # 3. hidden dimension V

    # NOTE: we are not supposed to paralleize over the K dimension, so BK is just for better triton loading,

    i_bh, i_t, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_bh * T
    p_q = tl.make_block_ptr(q + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + bos * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))

    # load Q, saved in sram the whole time
    b_q = tl.load(p_q, boundary_check=(0,1))

    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    b_m = tl.full([BT], float('-inf'), dtype=tl.float32) # max value

    b_acc = tl.zeros([BT], dtype=tl.float32) # accumulated sum of exp(x - m)

    # partition block along k and v for tiling
    # NOTE: causality here
    for i_s in tl.range(0, i_t * BT, BS):
        # since we use transpose of k, the order here represents the relative ordering of K and T w.r.t to their original undelying data layout
        # NOTE: remember this use of make_block_ptr
        p_k = tl.make_block_ptr(k + bos * K, (K, T), (1, K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + bos * V, (T, V), (V, 1), (i_s, 0), (BS, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0,1))
        b_v = tl.load(p_v, boundary_check=(0,1))
        # q @ k^T (tiled)
        b_s = tl.dot(b_q, b_k)