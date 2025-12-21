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
def _compare_and_swap(
    x,
    ids,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


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

    bos, eos = i_b * T, (i_b + 1) * T
    boc = i_b * TC

    # use base to aceess last two dims block

    p_q = tl.make_block_ptr(p + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

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

    bos, eos = i_b * T, (i_b + 1) * T
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
    # dk [NV, B, TC, H, K]
    # dv [B, TC, H, V]
    # do [B, T, HQ, V]
    # lse [B, T, HQ]
    # delta [B, T, HQ]

    # one element of a kv head means a group of G query heads

    # why dk and dq need NV but dv dont? because former two need dp, which is partial if you partition along V dimension


    i_c, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2) # NOTE: i_t to i_c, offsets
    i_b, i_h = i_bh // H, i_bh % H # NOTE: why not HQ as in FA2? because we are dealing with all HQ query heads (they share the same KV heads) in a single block

    TC = tl.cdiv(T, BS) # total blocks in original sequence
    NC = (i_t + 1) // BS # how many valid blocks for a query group at position i_t

    bos, eos = i_b * T, (i_b + 1) * T
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


def nsa_topk_kernel(
    q,
    k,
    v,
    lse,
    scale,
    block_indices, # indices of selected block
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
    B: tl.constexpr,
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
    p_lse = tl.make_block_ptr(lse + (bos + i_t) * HQ, (HQ,), (1,), (i_h * G), (G,), (0,))

    b_q = tl.load(p_q, boundary_check=(0,1))
    b_lse = tl.load(p_lse, boundary_check=(0,)) # NOTE: we assume the lse is returned in compression, so we do not recompute

    # NOTE: the first half of b_i always descends and second half always ascends if we are updating
    b_i = tl.zeros([BC], dtype=tl.float32) # [BC], where BC >= 2 * S because we are going to use bitonic merge to sort top-k
    o_i = tl.zeros([BC], dtype=tl.float32)
    m_i = tl.arange(0, BC) < (BC // 2)

    for i_c in tl.range(0, NC, BC):

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
        o_i, o_ip = tl.where(o_c < NC, o_c, -1), o_i

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
    chunk_offsets, # varlen
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

    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b, i_h = i_bh // H, i_bh % H

    bos = i_b * T

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1,0))

    b_q = tl.load(p_q, boundary_check=(0,1))

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S

    NS = S # number of selected blocks, currently default to S (the maximum value)

    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0)) # [G, BV]

    p_lse = tl.make_block_ptr(o + (bos + i_t) * HQ, (HQ,), (1,), (i_h * G,), (G,), (0,)) # [G]

    b_o = tl.zeros([G, BV], dtype=tl.float32)

    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)

    for i in tl.range(NS):

        i_s = tl.load(block_indices + i).to(tl.int32) * BS # the true offsets for selected k and v block

        if i_s <= i_t & i_s >= 0:
            # ensure causality
            o_s = i_s + tl.arange(0, BS) # dont know if this is requireed, since we do not select incploete block
            # NOTE: but one scnatios may be that S is larger than num of valid blocks, so we inevetbly choose some invalid block

            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BV, BS), (0, 1))

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

            b_o += tl.dot(b_p.to(b_v.dtype), b_v) # [G, BV]
    
    b_o = b_o / b_acc[:, None]
    b_m += tl.log(b_acc)
    tl.store(lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,1))


@triton.jit
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
    chunk_offsets, # varlen
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

    b_do = tl.load(b_do, boundary_check=(0,1))

    b_lse = tl.load(p_lse, boundary_check=(0,))

    b_delta = tl.load(p_delta, boundary_check=(0,))

    b_dq = tl.zeros([G, BK], dtype=tl.float32)

    NS = S # number of selected blocks, currently default to S (the maximum value)


    for i in tl.range(NS):

        i_s = tl.load(block_indices + i).to(tl.int32) * BS # the true offsets for selected k and v block

        # NOTE: i_s is suppoedd to be always less than 
        if i_s >= 0 & i_s <= i_t:

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
