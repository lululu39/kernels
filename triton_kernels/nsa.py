import triton.language as tl
import triton
import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous

@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})

# NOTE: in nsa, the BT in fa2 is now from BT to G (query head group) at the same i_t
# so you could also say that BT=1, but we add on another dimension which is G into consideration

# NOTE: dimensions offsets that are not used in block ptr offsets are then used on the base pointer

@triton.jit
def nsa_compression_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    cu_seqlens, # varlen
    token_indices, # varlen
    chunk_offsets, # varlen
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr, # NOTE: groups of query sharing the same KV set, this is a must because NSA use GQA
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr, # NOTE: block step size along compressed k and v
    BS: tl.constexpr, # block size of compression
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr, # meta param
):
    # q [B, T, HQ, K]
    # k [B, TC, H, K] NOTE: already compressed k and v
    # v [B, TC, H, V]
    # o [B, T, HQ, V]
    # lse [B, T, HQ]

    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H # NOTE: why not HQ as in FA2? because we are dealing with all HQ query heads (they share the same KV heads) in a single block

    TC = tl.cdiv(T, BS) # total blocks in original sequence
    NC = (i_t + 1) // BS # how many valid blocks for a query group at position i_t

    bos = i_b * T, eos = (i_b + 1) * T
    boc = i_b * TC

    # use base to aceess last two dims block

    p_q = tl.make_block_ptr(p + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, K), (1, 0))

    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0,1)) # [G, BK]
    
    b_o = tl.zeros([G, BV], dtype=tl.float32) # [G, BV]

    b_m = tl.full([G], float('-inf'), dtype=tl.float32)

    b_acc = tl.zeros([G], dtype=tl.float32) # lse = log(acc) + m

    for i_c in tl.range(0, NC, BC):
        
        o_c = tl.arange(0, BC) + i_c
        # we need k^T and v

        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1))

        p_v = tl.make_block_ptr(v + (boc * H + i_h) * V, (TC, V), (H * V, 1), (i_c, i_v * BV), (BC, BV), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0,1)) # [BK, BC]

        b_v = tl.load(p_v, boundary_check=(0,1)) # [BC, BV]

        b_s = tl.dot(b_q, b_k) * scale # [G, BC]

        # instead of two loops in FA2, we just use 1 loop with uniform
        b_s = tl.where((o_c < NC)[None, :], b_s, float('-inf'))

        b_m, b_mp = tl.maximum(tl.max(b_s, 1), b_m), b_m # [G]

        b_r = tl.exp(b_mp - b_m) # [G]

        b_p = tl.exp(b_s - b_m[:, None]) # [G, BC]

        b_acc = b_acc * b_r + tl.sum(b_p, 1)

        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v) # remember casting
    
    if NC == 0:
        b_lse = tl.zeros([G], dtype=tl.float32)
    else:
        b_o = b_o / b_acc[:, None]
        b_lse = b_m + tl.log(b_acc)
    
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,1))

    if i_v == 0:
        # we only store once the lse
        tl.store(lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G), b_lse.to(lse.dtype.element_ty))


@triton.jit
def nsa_compression_bwd_kernel_dq(
    q,
    k,
    v,
    do,
    dq,
    lse,
    delta,
    scale,
    cu_seqlens, # varlen
    token_indices, # varlen
    chunk_offsets, # varlen
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr, # NOTE: groups of query sharing the same KV set, this is a must because NSA use GQA
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr, # NOTE: block step size along compressed k and v
    BS: tl.constexpr, # block size of compression
    BK: tl.constexpr,
    BV: tl.constexpr, # NOTE: does not to be 1 because the dq now is NV * (*q.shape), so we can do partiton along V
    IS_VARLEN: tl.constexpr, # meta param
):
    # q [B, T, HQ, K]
    # k [B, TC, H, K] NOTE: already compressed k and v
    # v [B, TC, H, V]
    # dq [NV, B, T, HQ, K]
    # do [B, T, HQ, V]
    # lse [B, T, HQ]
    # delta [B, T, HQ]


    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H # NOTE: why not HQ as in FA2? because we are dealing with all HQ query heads (they share the same KV heads) in a single block

    TC = tl.cdiv(T, BS) # total blocks in original sequence
    NC = (i_t + 1) // BS # how many valid blocks for a query group at position i_t

    bos = i_b * T, eos = (i_b + 1) * T
    boc = i_b * TC

    # precompute the base pointer

    q += (bos + i_t) * HQ * K
    do += (bos + i_t) * HQ * V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * B * T + bos + i_t) * HQ * K

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse, (HQ,), (1,), (i_h * G,), (G,), (0,))
    p_delta = tl.make_block_ptr(delta, (HQ,), (q,), (i_h * G,), (G,), (0,))

    b_do = tl.load(p_do, boundary_check=(0,1)) # [G, BV]
    b_q = tl.load(p_q, boundary_check=(0,1))
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    b_dq = tl.zeros([G, BK], dtype=tl.float32)

    for i_c in tl.range(0, NC, BC):

        o_c = tl.arange(0, BC) + i_c

        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1))

        p_v = tl.make_block_ptr(v + (boc * H + i_h) * V, (V, TC), (1, H * V), (i_v * BV, i_c), (BV, BC), (0, 1)) # v^T

        b_k = tl.load(p_k, boundary_check=(0,1)) # [BK, BC]

        b_v = tl.load(p_v, boundary_check=(0,1)) # [BV, BC]

        b_s = tl.dot(b_q, b_k) # [G, BC]

        b_p = tl.exp(b_s - b_lse[:, None]) # [G, BC]

        b_p = tl.where((o_c < NC)[None, :], b_p, 0)

        b_dp = tl.dot(b_do, b_v) # [G, BC]

        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None]) # [G, BC] NOTE: remember to cast when elementwise 

        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))

    b_dq *= scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0,1))


@triton.jit
def nsa_compression_bwd_kernel_dkv(
    q,
    k,
    v,
    do,
    dk,
    dv,
    lse,
    delta,
    scale,
    offsets, # varlen
    token_indices, # varlen
    chunk_offsets, # varlen
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr, # NOTE: groups of query sharing the same KV set, this is a must because NSA use GQA
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr, # NOTE: block step size along compressed k and v
    BS: tl.constexpr, # block size of compression
    BK: tl.constexpr,
    BV: tl.constexpr, # NOTE: does not to be 1 because the dq now is NV * (*q.shape), so we can do partiton along V
    USE_OFFSETS: tl.constexpr, # meta param
):
    # q [B, T, HQ, K]
    # k [B, TC, H, K] NOTE: already compressed k and v
    # v [B, TC, H, V]
    # dk [NV, B, TC, H, K]
    # dv [B, TC, H, V]
    # do [B, T, HQ, V]
    # lse [B, T, HQ]
    # delta [B, T, HQ]

    # why dk and dq need NV but dv dont? because former two need dp, which is partial if you partition along V dimension


    i_c, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2) # NOTE: i_t to i_c, offsets
    i_b, i_h = i_bh // H, i_bh % H # NOTE: why not HQ as in FA2? because we are dealing with all HQ query heads (they share the same KV heads) in a single block

    TC = tl.cdiv(T, BS) # total blocks in original sequence
    NC = (i_t + 1) // BS # how many valid blocks for a query group at position i_t

    bos = i_b * T, eos = (i_b + 1) * T
    boc = i_b * TC

    k += (boc * H + i_h) * K
    v += (boc * H + i_h) * V
    dk += (i_v * B * TC * H + boc * H + i_h) * K
    dv += (boc * H + i_h) * V

    p_k = tl.make_block_ptr(k, (TC, K), (H*K, 1), (i_c * BC, 0), (BC, BK), (1,0))
    p_v = tl.make_block_ptr(v, (TC, V), (H*V, 1), (i_c * BC, i_v * BV), (BC, BV), (1,0))

    p_dk = tl.make_block_ptr(dk, (TC, K), (H*K, 1), (i_c * BC, 0), (BC, BK), (1,0))

    p_dv = tl.make_block_ptr(dv, (TC, V), (H * V, 1), (i_c * BC, i_v * BV), (BC, BV), (1,0))

    b_k = tl.load(p_k, boundary_check=(0,1))
    b_v = tl.load(p_v, boundary_check=(0,1))

    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)

    for i_t in tl.range(i_c * BC * BS, T):

        # for q, we deal with one row at a time, but with G heads
        o_c = tl.arange(0, BC) + i_c * BC

        p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1,0))
        p_lse = tl.make_block_ptr(lse + (bos + i_h) * HQ, (HQ,), (1,), (i_h * G,), (G,), (0,))
        p_delta = tl.make_block_ptr(delta + (bos + i_t) * HQ, (HQ,), (1,), (i_h * G,), (G,), (0,))

        b_q = tl.load(p_q, boundary_check=(0,1)) # [G, BK]
        b_do = tl.load(p_do, boundary_check=(0,1)) # [G, BV]
        b_lse = tl.load(p_lse, boundary_check=(0,)) # [G,]
        b_delta = tl.load(p_delta, boundary_check=(0,)) # [G,] 

        b_s = tl.dot(b_k, tl.trans(b_q)) * scale # [BC, G]
        b_p = tl.exp(b_s.to(tl.float32) - b_lse[None, :]) # [BC, G]

        # we do masking, translate to o_c < NC in previous kernel
        NC = (i_t + 1) // BS

        # NOTE: o_c is [BC] so we broadcast the second dimension
        b_p = tl.where((o_c < NC)[:, None], b_p, 0)

        b_dv += tl.dot(b_p.to(b_do.dtype), b_do) # [BC, BV]

        b_dp = tl.dot(b_v, tl.trans(b_do)) # [BC, G]
        b_ds = b_p * (b_dp - b_delta[None, :]) # [BC, G] this is a transpose

        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q) # [BC, BK]

    b_dk *= scale

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0,1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0,1))

    