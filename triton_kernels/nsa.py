import triton.language as tl
import triton
import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets, prepare_token_indices

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_func = None

from .fa2 import flash_attention_2_bwd_preprocess as preprocess

from .pooling import my_mean_pooling
from .sort import _bitonic_merge, _compare_and_swap # NOTE: we must import rather than directly copy to this file to avoid compilation error
# NOTE: in nsa, the BT in fa2 is now from BT to G (query head group) at the same i_t
# so you could also say that BT=1, but we add on another dimension which is G into consideration

# NOTE: dimensions offsets that are not used in block ptr offsets are then used on the base pointer


# NOTE: this configuration is important! I copied from fla
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
)
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


    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32) # how many chunks before
    else:
        bos, eos = i_b * T, (i_b + 1) * T
        boc = i_b * tl.cdiv(T, BS)


    TC = tl.cdiv(T, BS) # total blocks in original sequence
    NC = (i_t + 1) // BS # how many valid blocks for a query group at position i_t

    # use base to aceess last two dims block

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

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

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
)
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

    all = B * T # for vasrlen's sake, need to place before T is updated
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H # NOTE: why not HQ as in FA2? because we are dealing with all HQ query heads (they share the same KV heads) in a single block

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    # precompute the base pointer

    TC = tl.cdiv(T, BS) # total blocks in original sequence
    NC = (i_t + 1) // BS # how many valid blocks for a query group at position i_t

    q += (bos + i_t) * HQ * K
    do += (bos + i_t) * HQ * V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * all + bos + i_t) * HQ * K

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse, (HQ,), (1,), (i_h * G,), (G,), (0,))
    p_delta = tl.make_block_ptr(delta, (HQ,), (1,), (i_h * G,), (G,), (0,))

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

        b_s = tl.dot(b_q, b_k) * scale # [G, BC]

        b_p = tl.exp(b_s - b_lse[:, None]) # [G, BC]

        b_p = tl.where((o_c < NC)[None, :], b_p, 0)

        b_dp = tl.dot(b_do, b_v) # [G, BC]

        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None]) # [G, BC] NOTE: remember to cast when elementwise 

        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))

    b_dq *= scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0,1))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
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
    cu_seqlens, # varlen
    chunk_indices, # varlen
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
    # dk [NV, B, TC, H, K]
    # dv [B, TC, H, V]
    # do [B, T, HQ, V]
    # lse [B, T, HQ]
    # delta [B, T, HQ]

    # one element of a kv head means a group of G query heads

    # why dk and dq need NV but dv dont? because former two need dp, which is partial if you partition along V dimension


    i_c, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2) # NOTE: i_t to i_c, offsets
    i_b, i_h = i_bh // H, i_bh % H # NOTE: why not HQ as in FA2? because we are dealing with all HQ query heads (they share the same KV heads) in a single block

    all = B * tl.cdiv(T, BS)

    if IS_VARLEN:
        # NOTE: problems here!
        i_n, i_c = tl.load(chunk_indices + i_c * 2).to(tl.int32), tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)
    
    TC = tl.cdiv(T, BS) # total blocks in original sequence

    k += (boc * H + i_h) * K
    v += (boc * H + i_h) * V
    dk += (i_v * all * H + boc * H + i_h) * K
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
        p_lse = tl.make_block_ptr(lse + (bos + i_t) * HQ, (HQ,), (1,), (i_h * G,), (G,), (0,)) # NOTE: previously bug here
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

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK'],
)
@triton.jit
def nsa_topk_kernel(
    q,
    k,
    lse,
    scale,
    block_indices, # indices of selected block
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr, # NOTE: groups of query sharing the same KV set, this is a must because NSA use GQA
    K: tl.constexpr,
    S: tl.constexpr, # number of seletced block
    BC: tl.constexpr, # NOTE: block step size along compressed k and v
    BS: tl.constexpr, # block size of compression
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr, # meta param
):
    
    # does not involve v so only two-dim grid
    # we deal with G query heads in a block, so paralleize in q
    # lse [B, T, H]
    # block indices [B, T, H, S]
    # S <= BC // 2 because we will discard BS//2 history max in bitonic merge, so we do not want to lose potetial top-S max

    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    i_b, i_h = i_bh // H, i_bh % H

    TC = tl.cdiv(T, BS)
    # NOTE: when (i_t + 1) % BS == 0, NC != IC
    NC = (i_t + 1) // BS # this is a number
    IC = i_t // BS # this is a offset

    bos, eos = i_b * T, (i_b + 1) * T
    boc = i_b * TC

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1,0))
    p_lse = tl.make_block_ptr(lse + (bos + i_t) * HQ, (HQ,), (1,), (i_h * G,), (G,), (0,))

    b_q = tl.load(p_q, boundary_check=(0,1))
    b_lse = tl.load(p_lse, boundary_check=(0,)) # NOTE: we assume the lse is returned in compression, so we do not recompute

    # NOTE: the first half of b_i always descends and second half always ascends if we are updating
    # NOTE: set to -1 to indicate invalid index
    b_i = tl.full([BC], -1, dtype=tl.float32) # [BC], where BC >= 2 * S because we are going to use bitonic merge to sort top-k
    o_i = tl.zeros([BC], dtype=tl.int32) # NOTE: use int
    m_i = tl.arange(0, BC) < (BC // 2)

    # NOTE: why cdiv? because incomplete block should also be considered.
    # This is differnt from compression branch!
    for i_c in tl.range(0, tl.cdiv(i_t + 1, BS), BC):

        # NOTE: here I use a difference iteration strategy
        # NOTE: offset <= offset, while offset < number, vice versa
        # NOTE: when can be equal, use offset, when can not, use number

        o_c = tl.arange(0, BC) + i_c

        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0,1))

        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0,1))

        b_s = tl.dot(b_q, b_k) * scale # [G, BC]

        b_s = tl.where((o_c < NC)[None, :], b_s, float('-inf'))

        # always select 1st and last two blocks
        # NOTE: why, though

        b_p = tl.where(((o_c == 0) | (o_c == IC - 1) | (o_c == IC) ), 1., tl.exp(b_s - b_lse[:, None]))

        # accumulate scores across all G heads
        b_i, b_ip = tl.sum(b_p, 0), b_i # [BC]

        # NOTE: difference
        # discard invalid block offsets
        # NOTE: this step is required!!!!
        # NOTE: the incomplete block where current query token is located should be selected
        o_i, o_ip = tl.where(o_c <= IC, o_c, -1), o_i

        n_dims: tl.constexpr = tl.standard._log2(BC)

        # NOTE: it seems that we should first use order=2 when sorting smaller sequences, 
        # then use order=1/0 when sorting the whole sequence when our sequence is already bitonic

        for i in tl.static_range(1, n_dims):
            # we do bitnonic merge
            # NOTE: cast
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), i, 2, n_dims)

        if i_c == 0:
            # descending order
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, True, n_dims)
        else:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, False, n_dims) # ascending
            b_i = b_ip * m_i + b_i * (1 - m_i)
            o_i = o_ip * m_i + o_i * (1 - m_i)
            # then we make the bitonic sequence fully sorted: descending order
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, True, n_dims)
    
    # NOTE: pretty weird code
    m_top = tl.arange(0, BC//S) == 0
    b_top = tl.sum(m_top[:, None] * tl.reshape(o_i, [BC//S, S]), 0)

    # keep the block shape same to our data shape, which is S,

    p_b = tl.make_block_ptr(block_indices + (bos + i_t) * H * S, (H * S,), (1,), (i_h * S,), (S,), (0,))

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
def nsa_selection_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    block_indices, # indices of selected blocks
    block_counts, # how many blocks each query chooses, can be less than S
    cu_seqlens, # varlen
    token_indices, # varlen
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr, # NOTE: groups of query sharing the same KV set, this is a must because NSA use GQA
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr, # number of seletced block (max value)
    BS: tl.constexpr, # block size of compression
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr, # meta param
    USE_BLOCK_COUNTS: tl.constexpr,
):
    
    # q [B, T, HQ, K]
    # k [B, T, H, K] NOTE: non-compressed k and v
    # v [B, T, H, V]
    # o [B, T, HQ, V]
    # lse [B, T, HQ]
    # block_indices [B, T, H, S]

    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b, i_h = i_bh // H, i_bh % H

    bos = i_b * T

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1,0))

    b_q = tl.load(p_q, boundary_check=(0,1))

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S

    NS = S # number of selected blocks, currently default to S (the maximum value)

    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0)) # [G, BV]

    p_lse = tl.make_block_ptr(lse + (bos + i_t) * HQ, (HQ,), (1,), (i_h * G,), (G,), (0,)) # [G]

    b_o = tl.zeros([G, BV], dtype=tl.float32)

    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)

    for i in tl.range(NS):

        i_s = tl.load(block_indices + i).to(tl.int32) * BS # the true offsets for selected k and v block

        if i_s <= i_t and i_s >= 0: # NOTE: use and, no &
            # ensure causality
            o_s = i_s + tl.arange(0, BS) # dont know if this is requireed, since we do not select incploete block
            # NOTE: but one scnatios may be that S is larger than num of valid blocks, so we inevetbly choose some invalid block

            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))

            p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1,0))

            b_k = tl.load(p_k, boundary_check=(0,1)) # [BK, BS]

            b_v = tl.load(p_v, boundary_check=(0,1)) # [BS, BV]

            b_s = tl.dot(b_q, b_k) * scale # [G, BS]

            # NOTE: still needs some masking to avoid corner cases
            # NOTE: anyway you just do it there's no harm
            b_s = tl.where((o_s <= i_t)[None, :], b_s, float('-inf'))

            b_m, b_mp = tl.maximum(tl.max(b_s, 1), b_m), b_m # [G]

            b_r = tl.exp(b_mp - b_m) # [G]

            b_p = tl.exp(b_s - b_m[:, None])

            b_acc = b_acc * b_r + tl.sum(b_p, 1)

            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v) # [G, BV] NOTE: o need to be rescaled as well
    
    b_o = b_o / b_acc[:, None]
    b_m += tl.log(b_acc)
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor)
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def nsa_selection_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    block_indices, # indices of selected blocks
    block_counts, # how many blocks each query chooses, can be less than S
    cu_seqlens, # varlen
    token_indices, # varlen
    T,
    B: tl.constexpr, # NOTE
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr, # NOTE: groups of query sharing the same KV set, this is a must because NSA use GQA
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr, # number of seletced block (max value)
    BS: tl.constexpr, # block size of compression
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr, # meta param
    USE_BLOCK_COUNTS: tl.constexpr,
):
    # dq: [NV, B, T, H, K]
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b, i_h = i_bh // H, i_bh % H

    bos = i_b * T

    block_indices += ((bos + i_t) * H + i_h) * S

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1,0))

    p_dq = tl.make_block_ptr(dq + (i_v * B * T + bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1,0))

    p_do = tl.make_block_ptr(do + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1,0))

    p_lse = tl.make_block_ptr(lse + (bos + i_t) * HQ, (HQ,), (1,), (i_h * G,), (G,), (0,))

    p_delta = tl.make_block_ptr(delta + (bos + i_t) * HQ, (HQ,), (1,), (i_h * G,), (G,), (0,))

    b_q = tl.load(p_q, boundary_check=(0,1))

    b_do = tl.load(p_do, boundary_check=(0,1))

    b_lse = tl.load(p_lse, boundary_check=(0,))

    b_delta = tl.load(p_delta, boundary_check=(0,))

    b_dq = tl.zeros([G, BK], dtype=tl.float32)

    NS = S # number of selected blocks, currently default to S (the maximum value)


    for i in tl.range(NS):

        i_s = tl.load(block_indices + i).to(tl.int32) * BS # the true offsets for selected k and v block

        # NOTE: i_s is suppoedd to be always less than 
        if i_s >= 0 and i_s <= i_t:

            o_s = tl.arange(0, BS) + i_s

            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_s), (BK, BS), (0,1))
            p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0,1))

            b_k = tl.load(p_k, boundary_check=(0,1))
            b_v = tl.load(p_v, boundary_check=(0,1)) # [BV, BS]

            b_s = tl.dot(b_q, b_k) * scale # [G, BS] NOTE: this is only partial, so we need NV dim for dq!

            b_p = tl.exp(b_s - b_lse[:, None]) # [G, BS]

            b_p = tl.where((o_s <= i_t)[None, :], b_p, 0)

            b_dp = tl.dot(b_do, b_v) # [G, BS]
            
            # NOTE: cast here
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None]) # [G, BS]

            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    
    b_dq *= scale # wherether you scale q or not, for dq we always need to scale in the end. why? see the formula

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0,1))


@triton.heuristics({
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor)
})
@triton.jit(do_not_specialize=['T'])
def nsa_selection_kernel_block_mask(
    block_indices,
    block_counts,
    block_mask,
    H: tl.constexpr,
    S: tl.constexpr,
    T, # NOTE: do not use tl.contextptr fotr T, otherwise compilation error
    BS: tl.constexpr,
    NS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr # not used currently
):
    # not sure why NS = tl.cdiv(T, BS) is used, because block_indices shoulw not contain indices larger than NS

    # NOTE: imo, NS shoule be T // BS, no cdiv since we want complete blocks
    # block_indices [B, T, H, S]
    i_b, i_t, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_h, i_s = i_hs // S, i_hs % S

    bos = i_b * T

    b_i = tl.load(block_indices + ((bos + i_t) * H + i_h) * S + i_s) # NOTE: use scalar load!

    b_m = ((b_i <= i_t // BS) and (b_i >= 0)) # again, first part is not required since we did this in top-k kernel and set invalid indices to -1
    # b_m = (b_i >= 0) # NOTE: this works as well!


    if b_i < NS and b_i >= 0:
        tl.store(block_mask + ((bos + i_t) * H + i_h) * NS + b_i, b_m.to(block_mask.dtype.element_ty))
    
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def nsa_selection_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    scale,
    block_mask, # use this instaed of online computing a block is valid or not
    cu_seqlens, # varlen
    chunk_indices, # varlen
    T,
    B: tl.constexpr, # NOTE
    H: tl.constexpr,
    HQ: tl.constexpr,
    M: tl.constexpr, # NOTE: besically 
    G: tl.constexpr, # NOTE: groups of query sharing the same KV set, this is a must because NSA use GQA
    K: tl.constexpr,
    V: tl.constexpr,
    BS: tl.constexpr, # block size of compression
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr, # meta param
):
    # bk and bv should be [BS, BK] and [BV, BS]

    # NOTE: we use NS = cdiv(T, BS) to iterate all blocks (including incomplete ones)
    # because dk and dv is torch.empty, so we nned to store zero into the incomplete block (eve if we do not calculate it)
    # as for block_mask kernel, i don't think cdiv is required, since block_mask is initially zero
    i_s, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b, i_h = i_bh // H, i_bh % H

    all = B * T

    bos = i_b * T

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1,0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1,0))

    # NOTE: k needs an extra NV dimension
    p_dk = tl.make_block_ptr(dk + ((i_v * all + bos) * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1,0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1,0))

    b_k = tl.load(p_k, boundary_check=(0,1)) # [BS, BK]
    b_v = tl.load(p_v, boundary_check=(0,1)) # [BS, BV]

    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    o_s = tl.arange(0, BS) + i_s * BS

    for i in tl.range(i_s * BS, T):
        b_m = tl.load(block_mask + (bos + i) * H * M + i_h * M + i_s)

        if b_m:
            # this k/v block is valid at query position i
            p_q = tl.make_block_ptr(q + (bos + i) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1,0))
            p_lse = tl.make_block_ptr(lse + (bos + i) * HQ, (HQ,), (1,), (i_h * G,), (G,), (0,))
            p_delta = tl.make_block_ptr(delta + (bos + i) * HQ, (HQ,), (1,), (i_h * G,), (G,), (0,))
            p_do = tl.make_block_ptr(do + (bos + i) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))

            b_q = tl.load(p_q, boundary_check=(0,1)) # [G, BK]
            b_lse = tl.load(p_lse, boundary_check=(0,)) # [G,]
            b_delta = tl.load(p_delta, boundary_check=(0,)) # [G,]
            b_do = tl.load(p_do, boundary_check=(0,1)) # [G, BV]

            b_s = tl.dot(b_k, tl.trans(b_q)) * scale # [BS, G]
            b_p = tl.exp(b_s - b_lse[None, :])
            b_p = tl.where((o_s <= i)[:, None], b_p, 0)
            b_dp = tl.dot(b_v, tl.trans(b_do)) # [BS, G]
            b_ds = b_p * (b_dp - b_delta[None, :]) # [BS, G]

            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q) # [BS, BK]
    
    b_dk *= scale
    
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0,1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0,1))


def nsa_compression_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None, # used for varlen
):
    B, T, HQ, K, V, H = *q.shape, v.shape[-1], k.shape[-2]
    G = HQ // H
    BC = BS = block_size

    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    # NOTE: cumulative chunk number
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BS) if cu_seqlens is not None else None

    grid = (T, NV, B * H)

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device) # NOTE: the dtype and q device
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    nsa_compression_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BC=BC,
        BS=BS,
        BK=BK,
        BV=BV,
    )

    return o, lse


def nsa_compression_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    block_size: int,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None,
):
    
    B, T, HQ, K, V, H = *q.shape, v.shape[-1], k.shape[-2]
    G = HQ // H

    BC = BS = block_size

    BK = max(triton.next_power_of_2(K), 16) # BK less than K
    BV = min(128, max(triton.next_power_of_2(v.shape[-1]), 16))

    NV = triton.cdiv(V, BV)

    if cu_seqlens is not None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        chunk_indices = prepare_chunk_indices(chunk_offsets, BC) # true chunk indices for compressed k and v
        NC = len(chunk_indices) # total number of kv chunks (NOTE: chunk upon already compressed kv, so we use chunk_offsets to obtain chunk_indices for dkv kernel because we weant to process BC nuumber of kv (compressed!) tokens in one kernel)
    else:
        chunk_indices, chunk_offsets = None, None
        NC = triton.cdiv(triton.cdiv(T, BS), BC)

    delta = preprocess(o, do)

    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)

    grid = (T, NV, B * H)

    nsa_compression_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        do=do,
        dq=dq,
        lse=lse,
        delta=delta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BC=BC,
        BS=BS,
        BK=BK,
        BV=BV,
    )

    dq = dq.sum(0) # we accumulate results along NV

    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(*v.shape, dtype=v.dtype, device=q.device)

    grid = (NC, NV, B * H)

    nsa_compression_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        do=do,
        dk=dk,
        dv=dv,
        lse=lse,
        delta=delta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BC=BC,
        BS=BS,
        BK=BK,
        BV=BV,
    )

    dk = dk.sum(0)

    return dq, dk, dv

class NSACompressionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_size, scale, cu_seqlens):
        ctx.dtype = q.dtype

        # 2-d sequence indices denoting the cu_seqlens of tokens in each sequence
        # for example, if the passed `cu_seqlens` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        token_indices = prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None

        o, lse = nsa_compression_fwd(
            q=q,
            k=k,
            v=v,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.cu_seqlens = cu_seqlens
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.scale = scale

        # NOTE: the cast here
        return o.to(q), lse

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, *args):
        # NOTE: the args here is because we return lse as well
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = nsa_compression_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            do=do,
            lse=lse,
            scale=ctx.scale,
            block_size=ctx.block_size,
            cu_seqlens=ctx.cu_seqlens,
            token_indices=ctx.token_indices,
        )

        return dq.to(q), dk.to(k), dv.to(v), None, None, None


def nsa_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    if scale is None:
        scale = k.shape[-1] ** -0.5

    return NSACompressionFunction.apply(
        q,
        k,
        v,
        block_size,
        scale,
        cu_seqlens
    )
    

def nsa_selection_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor, # obtained with top-k
    block_counts: torch.LongTensor | int,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None, # used for varlen
):
    
    B, T, HQ, K, H, V, S = *q.shape, k.shape[-2], v.shape[-1], block_indices.shape[-1]
    G = HQ // H
    BS = block_size

    BK = min(128, triton.next_power_of_2(K)) # generally we do not split K
    BV = min(128, triton.next_power_of_2(V))

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    assert NK == 1, "The key dimension can not be larger than 256"

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device) # NOTE: use float, and is of HQ!

    grid = (T, NV, B * H) # why T, because only one query position is processed in each block
    nsa_selection_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        scale=scale,
        block_indices=block_indices,
        block_counts=block_counts,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV,
    )

    return o, lse

def nsa_selection_block_mask(
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    cu_seqlens: torch.LongTensor,
    block_size: int,
):
    B, T, H, S = block_indices.shape
    BS = block_size
    NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)

    nsa_selection_kernel_block_mask[(B, T, H * S)](
        block_indices=block_indices,
        block_counts=block_counts,
        block_mask=block_mask,
        H=H,
        S=S,
        T=T,
        BS=BS,
        NS=NS,
    )
    return block_mask

def nsa_selection_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    block_indices: torch.LongTensor, # obtained with top-k
    block_counts: torch.LongTensor | int,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None, # used for varlen 
):
    B, T, HQ, K, H, V, S = *q.shape, k.shape[-2], v.shape[-1], block_indices.shape[-1]
    G = HQ // H
    BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    BV = min(128, max(triton.next_power_of_2(v.shape[-1]), 16)) # again, split v, not k

    NV = triton.cdiv(V, BV)
    
    NS = triton.cdiv(T, BS) # used for dkv kernel

    from .fa2 import flash_attention_2_bwd_preprocess as preprocess

    delta = preprocess(o, do)

    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    grid = (T, NV, B * H)
    nsa_selection_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        block_indices=block_indices,
        block_counts=block_counts,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV,
    )

    chunk_indices = None
    dq = dq.sum(0)

    grid = (NS, NV, B * H)

    block_mask = nsa_selection_block_mask(block_indices=block_indices, block_counts=block_counts, cu_seqlens=cu_seqlens, block_size=block_size)

    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    nsa_selection_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        block_mask=block_mask,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        M=block_mask.shape[-1],
        BS=BS,
        BK=BK,
        BV=BV,
    )

    dk = dk.sum(0)
    return dq, dk, dv


@torch.compile
class NSASelectionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_indices, block_counts, block_size, scale, cu_seqlens):
        ctx.dtype = q.dtype

        # token_indices = prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
        token_indices = None

        o, lse = nsa_selection_fwd(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.block_indices = block_indices
        ctx.block_counts = block_counts
        ctx.cu_seqlens = cu_seqlens
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.scale = scale

        return o.to(q)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = nsa_selection_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            block_indices=ctx.block_indices,
            block_counts=ctx.block_counts,
            block_size=ctx.block_size,
            scale=ctx.scale,
            cu_seqlens=ctx.cu_seqlens,
            token_indices=ctx.token_indices,
        )
        return dq.to(q), dk.to(k), dv.to(v), None, None, None, None, None

def nsa_selection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    block_size: int,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    if scale is None:
        scale = k.shape[-1] ** -0.5
    
    return NSASelectionFunction.apply(
        q,
        k,
        v,
        block_indices,
        block_counts,
        block_size,
        scale,
        cu_seqlens
    )


def nsa_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    block_counts: torch.LongTensor | int,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None
):
    B, T, HQ, K, H = *q.shape, k.shape[-2]
    G = HQ // H
    S = block_counts if isinstance(block_counts, int) else block_counts.max().item()
    S = triton.next_power_of_2(S) # NOTE: triton requires us to do so
    BC = BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    assert BC >= 2 * S, f"BC ({BC}) must be greater than or equal to 2 * S ({S})"

    block_indices = torch.zeros(B, T, H, S, dtype=torch.int32, device=q.device)

    # token_indices = prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
    # chunk_offsets = prepare_chunk_offsets(cu_seqlens, BS) if cu_seqlens is not None else None

    token_indices = None
    chunk_offsets = None
    grid = (T, B * H)
    nsa_topk_kernel[grid](
        q=q,
        k=k,
        lse=lse,
        scale=scale,
        block_indices=block_indices,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        S=S,
        BC=BC,
        BS=BS,
        BK=BK,
    )
    return block_indices


    

def native_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor | None,
    g_slc: torch.Tensor | None,
    g_swa: torch.Tensor | None,
    block_indices: torch.LongTensor | None,
    block_counts: torch.LongTensor | int = 16,
    block_size: int = 16,
    window_size: int = 0,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g_cmp (torch.Tensor):
            Gate score for compressed attention of shape `[B, T, HQ]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, T, HQ]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, T, HQ]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]`.
            `S` is the number of selected blocks for each query token, which is set to 16 in the paper.
            If `g_cmp` is provided, the passed `block_indices` will be ignored.
        block_counts (Optional[Union[torch.LongTensor, int]]):
            Number of selected blocks for each query.
            If a tensor is provided, with shape `[B, T, H]`,
            each query can select the same number of blocks.
            If not provided, it will default to 16.
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
    """

    assert block_counts is not None, "block counts must be provided for selection"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    # NOTE: note this! 
    assert q.shape[2] % (k.shape[2] * 16) == 0, "Group size must be a multiple of 16 in NSA"

    k_cmp, v_cmp = my_mean_pooling(k, block_size, cu_seqlens), my_mean_pooling(v, block_size, cu_seqlens)

    o_cmp, lse_cmp = None, None

    if g_cmp is not None:
        o_cmp, lse_cmp = nsa_compression(
            q=q,
            k=k_cmp,
            v=v_cmp, # NOTE!!!!!!!!!!!!!!
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens
        )
        if block_indices is not None:
            import warnings
            warnings.warn("`block_indices` will be ignored when `g_cmp` is provided")
        
        block_indices = nsa_topk(
            q=q,
            k=k_cmp,
            lse=lse_cmp,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens
        )
    
    o = o_slc = nsa_selection(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        scale=scale,
        cu_seqlens=cu_seqlens
    )

    if g_slc is not None:
        o = o_slc * g_slc.unsqueeze(-1)
    
    if g_cmp is not None:
        o = torch.addcmul(o, o_cmp, g_cmp.unsqueeze(-1))
    
    if window_size > 0:
        if cu_seqlens is not None:
            max_seqlen = q.shape[1]
            o_swa = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(window_size-1, 0),
            ).unsqueeze(0)
        else:
            o_swa = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(window_size-1, 0),
            )
        o = torch.addcmul(o, o_swa, g_swa.unsqueeze(-1))
    return o