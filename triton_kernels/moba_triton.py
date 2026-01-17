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

def moba_topk(
    q: torch.Tensor,
    k: torch.Tensor,
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

    block_indices = torch.zeros(B, T, HQ, S, dtype=torch.int32, device=q.device)

    token_indices = prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BS) if cu_seqlens is not None else None

    grid = (T, B * HQ)
    moba_topk_kernel[grid](
        q=q,
        k=k,
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
def moba_bwd_kernel_dq(
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
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr, 
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr, 
    BS: tl.constexpr, 
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr, 
    USE_BLOCK_COUNTS: tl.constexpr,
):
    
    # dq: [NV, B, T, HQ, K]

    i_t, i_v, i_bhq = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b, i_hq = i_bhq // HQ, i_bhq % HQ

    i_h = i_hq // G

    all = B * T

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    
    block_indices += ((bos + i_t) * HQ + i_hq) * S


    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_hq, 0), (1, BK), (1,0))

    p_dq = tl.make_block_ptr(dq + (i_v * all + bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_hq, 0), (1, BK), (1,0))

    p_do = tl.make_block_ptr(do + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_hq, i_v * BV), (1, BV), (1,0))

    p_lse = tl.make_block_ptr(lse + (bos + i_t) * HQ, (HQ,), (1,), (i_hq,), (1,), (0,))

    p_delta = tl.make_block_ptr(delta + (bos + i_t) * HQ, (HQ,), (1,), (i_hq,), (1,), (0,))

    b_q = tl.load(p_q, boundary_check=(0,1))

    b_do = tl.load(p_do, boundary_check=(0,1))

    b_lse = tl.load(p_lse, boundary_check=(0,))

    b_delta = tl.load(p_delta, boundary_check=(0,))

    b_dq = tl.zeros([1, BK], dtype=tl.float32)

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * HQ + i_hq)
    else:
        NS = S
    
    for i in tl.range(NS):

        i_s = tl.load(block_indices + i).to(tl.int32) * BS

        if i_s >= 0 and i_s <= i_t:

            o_s = tl.arange(0, BS) + i_s

            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_s), (BK, BS), (0,1))
            p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0,1))

            b_k = tl.load(p_k, boundary_check=(0,1)) # [BK, BS]
            b_v = tl.load(p_v, boundary_check=(0,1)) # [BV, BS]

            b_s = tl.dot(b_q, b_k) * scale

            b_p = tl.exp(b_s - b_lse[:, None]) # [1, BS]

            b_p = tl.where((o_s <= i_t)[None, :], b_p, 0)

            b_dp = tl.dot(b_do, b_v)

            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None]) # [1, BS]

            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k)) # [1, BK]
        
    
    b_dq *= scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0,1))


@triton.heuristics({
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor)
})
@triton.jit(do_not_specialize=['T'])
def moba_kernel_block_mask(
    block_indices,
    block_counts,
    block_mask,
    HQ: tl.constexpr,
    S: tl.constexpr,
    T,
    BS: tl.constexpr,
    NS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr # not used currently
):  
    # NOTE: basically the same as NSA, except for the head processing
    
    # block_indices [B, T, HQ, S]
    # NOTE: for dkv kernel, we add additional G dimension and then aggregate along this dimension, to support GQA

    i_b, i_t, i_hqs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    i_hq, i_s = i_hqs // S, i_hqs % S

    bos = i_b * T

    b_i = tl.load(block_indices + ((bos + i_t) * HQ + i_hq) * S + i_s) # NOTE: use scalar load!
    
    if USE_BLOCK_COUNTS:
        # causality and less than predefined maximum blocks
        b_m = b_i * BS <= i_t and i_s < tl.load(block_counts + i_b * T * HQ + i_t * HQ + i_hq)
    else:
        b_m = ((b_i <= i_t // BS) and (b_i >= 0))

    if b_i < NS and b_i >= 0:
        tl.store(block_mask + ((bos + i_t) * HQ + i_hq) * NS + b_i, b_m.to(block_mask.dtype.element_ty))


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
def moba_bwd_kernel_dkv(
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
    B: tl.constexpr, 
    H: tl.constexpr,
    HQ: tl.constexpr,
    M: tl.constexpr, # NOTE: besically cdiv(T, BS)
    G: tl.constexpr, # NOTE: groups of query sharing the same KV set, this is a must because we maybe using use GQA
    K: tl.constexpr,
    V: tl.constexpr,
    BS: tl.constexpr, # block size of compression
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr, # meta param
):
    # bk and bv should be [BS, BK] and [BV, BS]

    # NOTE: for moba, we do not share the same selectin in a query group, so must add one dimension G to aggregate all the gradients
    # dk: [NV * G, T, H, K]
    # dv: [G, T, H, V] because v does not need to aggregate along NV

    # NOTE: we use NS = cdiv(T, BS) to iterate all blocks (including incomplete ones)
    # because dk and dv is torch.empty, so we nned to store zero into the incomplete block (eve if we do not calculate it)
    # as for block_mask kernel, i don't think cdiv is required, since block_mask is initially zero
    i_s, i_vg, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b, i_h = i_bh // H, i_bh % H
    i_v, i_g = i_vg // G, i_vg % G

    all = B * T

    if IS_VARLEN:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(chunk_indices + i_s * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1,0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1,0))

    # NOTE: k needs an extra NV dimension
    p_dk = tl.make_block_ptr(dk + ((i_vg * all + bos) * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1,0))
    p_dv = tl.make_block_ptr(dv + ((i_g * all + bos) * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1,0))

    b_k = tl.load(p_k, boundary_check=(0,1)) # [BS, BK]
    b_v = tl.load(p_v, boundary_check=(0,1)) # [BS, BV]

    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    o_s = tl.arange(0, BS) + i_s * BS

    for i in tl.range(i_s * BS, T):
        b_m = tl.load(block_mask + (bos + i) * HQ * M + (i_h * G + i_g) * M + i_s)

        if b_m:
            # this k/v block is valid at query position i
            p_q = tl.make_block_ptr(q + (bos + i) * HQ * K, (HQ, K), (K, 1), (i_h * G + i_g, 0), (1, BK), (1,0))
            p_lse = tl.make_block_ptr(lse + (bos + i) * HQ, (HQ,), (1,), (i_h * G + i_g,), (1,), (0,))
            p_delta = tl.make_block_ptr(delta + (bos + i) * HQ, (HQ,), (1,), (i_h * G + i_g,), (1,), (0,))
            p_do = tl.make_block_ptr(do + (bos + i) * HQ*V, (HQ, V), (V, 1), (i_h * G + i_g, i_v * BV), (1, BV), (1, 0))

            b_q = tl.load(p_q, boundary_check=(0,1)) # [1, BK]
            b_lse = tl.load(p_lse, boundary_check=(0,)) # [1,]
            b_delta = tl.load(p_delta, boundary_check=(0,)) # [1,]
            b_do = tl.load(p_do, boundary_check=(0,1)) # [1, BV]

            b_s = tl.dot(b_k, tl.trans(b_q)) * scale # [BS, 1]
            b_p = tl.exp(b_s - b_lse[None, :])
            b_p = tl.where((o_s <= i)[:, None], b_p, 0)
            b_dp = tl.dot(b_v, tl.trans(b_do)) # [BS, 1]
            b_ds = b_p * (b_dp - b_delta[None, :]) # [BS, 1]

            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q) # [BS, BK]
    
    b_dk *= scale
    
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0,1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0,1))


def moba_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None, 
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

    grid = (T, NV, B * HQ) 
    moba_fwd_kernel[grid](
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

def moba_block_mask(
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    cu_seqlens: torch.LongTensor,
    block_size: int,
):
    B, T, HQ, S = block_indices.shape
    BS = block_size
    if cu_seqlens is not None:
        NS = triton.cdiv(prepare_lens(cu_seqlens).max().item(), BS)
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, HQ, NS, dtype=torch.bool, device=block_indices.device)

    moba_kernel_block_mask[(B, T, HQ * S)](
        block_indices=block_indices,
        block_counts=block_counts,
        block_mask=block_mask,
        HQ=HQ,
        S=S,
        T=T,
        BS=BS,
        NS=NS,
    )
    return block_mask


def moba_bwd(
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
    
    delta = preprocess(o, do)

    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    grid = (T, NV, B * HQ)
    moba_bwd_kernel_dq[grid](
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

    dq = dq.sum(0)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BS)
        NS = len(chunk_indices)
    else:
        chunk_indices = None
        NS = triton.cdiv(T, BS)

    grid = (NS, NV * G, B * H)

    block_mask = moba_block_mask(block_indices=block_indices, block_counts=block_counts, cu_seqlens=cu_seqlens, block_size=block_size)

    dk = torch.empty(NV * G, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(G, *v.shape, dtype=v.dtype, device=q.device)

    moba_bwd_kernel_dkv[grid](
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
    dv = dv.sum(0)

    return dq, dk, dv

@torch.compile
class MoBAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_indices, block_counts, block_size, scale, cu_seqlens):
        ctx.dtype = q.dtype

        token_indices = prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None

        o, lse = moba_fwd(
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
        dq, dk, dv = moba_bwd(
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


def mixture_of_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_counts: torch.LongTensor | int,
    block_size: int,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    if scale is None:
        scale = k.shape[-1] ** -0.5
    
    k_cmp = my_mean_pooling(k, chunk_size=block_size, cu_seqlens=cu_seqlens)

    block_indices = moba_topk(
        q=q,
        k=k_cmp,
        block_counts=block_counts,
        block_size=block_size,
        scale=scale,
        cu_seqlens=cu_seqlens
    )
    
    return MoBAFunction.apply(
        q,
        k,
        v,
        block_indices,
        block_counts,
        block_size,
        scale,
        cu_seqlens
    )