import triton.language as tl
import triton
import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets, prepare_token_indices, prepare_lens

from .pooling import my_mean_pooling
from .sort import _bitonic_merge, _compare_and_swap # NOTE: we must import rather than directly copy to this file to avoid compilation error
from fla.ops.attn.parallel import parallel_attn_bwd_preprocess as preprocess

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK'],
)
@triton.jit
def moba_topk_kernel(
    q,
    k, # compressed
    scale,
    block_indices,
    cu_seqlens,
    token_indices, # varlen
    chunk_offsets, # cu seqlens for kv
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr, # how many blocks to choose per query token
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    # NOTE: in moba, we do not share query groups for selection, so 
    # q [B, T, HQ, K]
    # k [B, TC, H, K]
    # block indieces shape [B, T, HQ, S]
    
    # NOTE: maybe we can parallize the query, but i think it's hard using this kind of sorting top-k

    i_t, i_bhq = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bhq // HQ, i_bhq % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_hq, 0), (1, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0,1)) # [1, BK]

    TC = tl.cdiv(T, BS)
    NC = (i_t + 1) // BS # complete blocks

    # calculate lse
    b_m = tl.full([1], float('-inf'), dtype=tl.float32)

    b_acc = tl.zeros([1], dtype=tl.float32)

    for i_c in tl.range(0, NC, BC):

        o_c = i_c + tl.arange(0, BC)

        # k transpose
        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c * BC), (BK, BC), (0, 1))

        b_k = tl.load(p_k, boundary_check=(0,1)) # [BK, BC]

        b_s = tl.dot(b_q, b_k) * scale # [1, BC]
        b_s = tl.where((o_c < NC)[None, :], b_s, float('-inf'))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m

        b_r = tl.exp(b_mp - b_m) # [1]

        b_p = tl.exp(b_s - b_m[None, :]) # [1, BC]

        b_acc = b_acc * b_r + tl.sum(b_p, 1)
    
    if NC == 0:
        b_lse = tl.zeros([1], dtype=tl.float32)
    else:
        b_lse = b_m + log(b_acc)
    
    # calculate the topk blocks

    b_i = tl.full([BC], -1, dtype=tl.float32)
    o_i = tl.zeros([BC], dtype=tl.int32)
    m_i = tl.arange(0, BC) < BC // 2

    IC = i_t // BS

    for i_c in tl.range(0, tl.cdiv(i_t + 1, BS), BC):
        o_c = i_c + tl.arange(0, BC)

        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c * BC), (BK, BC), (0, 1))

        b_k = tl.load(p_k, boundary_check=(0,1)) # [BK, BC]

        b_s = tl.dot(b_q, b_k) * scale # [1, BC]

        b_s = tl.where((o_c < NC)[None, :], b_s, float('-inf'))

        # must select current block
        b_p = tl.where((o_c == IC), 1., tl.exp(b_s - b_lse[:, None]))

        b_i, b_ip = tl.sum(b_p, 0), b_i # [BC], here sum is to reduce dimension

        o_i, o_ip = tl.where(o_c <= IC, o_c, -1), o_i

        n_dims: tl.constexpr = tl.standard._log2(b_i.shape[0])
        for i in tl.static_range(1, n_dims):
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), i, 2, n_dims)

        if i_c != 0:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, False, n_dims)
            b_i_new = b_ip * m_i + b_i * (1 - m_i)
            o_i_new = o_ip * m_i + o_i * (1 - m_i)
            b_i, o_i = _bitonic_merge(b_i_new, o_i_new.to(tl.int32), n_dims, True, n_dims)
        else:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, True, n_dims)
    
    m_top = tl.arange(0, BC//S) == 0
    b_top = tl.sum(m_top[:, None] * tl.reshape(o_i, [BC//S, S]), 0) # [S,]

    # this HQ * S is to avoid shape manipulation for b_top
    p_b = tl.make_block_ptr(block_indices + (bos + i_t) * HQ * S, (HQ * S,), (1,), (i_hq * S,), (S,), (0,))
    tl.store(p_b, b_top.to(p_b.dtype.element_ty))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit
def moba_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr, # varlen
    USE_BLOCK_COUNTS: tl.constexpr,
):
    
    # q [B, T, HQ, K]
    # k [B, T, H, K] NOTE: non-compressed k and v
    # v [B, T, H, V]
    # o [B, T, HQ, V]
    # lse [B, T, HQ]
    # block_indices [B, T, HQ, S]


    i_t, i_v, i_bhq = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b, i_hq = i_bhq // HQ, i_bhq % HQ

    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    
    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_hq, 0), (1, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0,1))

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    block_indices += (bos + i_t) * HQ * S + i_hq * S

    if USE_BLOCK_COUNTS:
        # [B, T, HQ]
        NS = tl.load(block_counts + (bos + i_t) * HQ + i_hq)
    else:
        NS = S 
    
    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_hq, i_v * BV), (1, BV), (1,0))

    p_lse = tl.make_block_ptr(lse + (bos + i_t) * HQ, (HQ,), (1,), (i_hq,), (1,), (0,))

    b_o = tl.zeros([1, BV], dtype=tl.float32)

    b_m = tl.full([1], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([1], dtype=tl.float32)

    for i in tl.range(NS):

        i_s = tl.load(block_indices + i).to(tl.int32) * BS # offset for kv blocks

        if i_s <= i_t and i_s >= 0:
            o_s = i_s + tl.arange(0, BS) # a block offset

            p_k = tl.make_block_ptr(K, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))

            p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1,0))

            b_k = tl.load(p_k, boundary_check=(0,1)) # [BK, BS]

            b_v = tl.load(p_v, boundary_check=(0,1)) # [BS, BV]

            b_s = tl.dot(b_q, b_k) * scale # [1, BS]

            b_s = tl.where((o_s < i_t)[None, :], b_s, float('-inf'))

            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m

            b_r = tl.exp(b_mp - b_m) # [1,]

            b_p = tl.exp(b_s - b_m[:, None])

            b_acc = b_acc * b_r + tl.sum(b_p, 1)

            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v) # [1, BV]

    b_o = b_o / b_acc[:, None] # broadcast
    b_m += tl.log(b_acc) # lse

    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,1))
