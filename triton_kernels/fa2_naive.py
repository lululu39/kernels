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
    scale,
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

    RCP_LN2: tl.constexpr = 1.4426950216 # use this instead of directly tl.exp for numerical reasons
    # exp = exp2(x * RCP_LN2)

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
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2 # [BT, BS]
        # save previous maximum
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, axis=1)), b_m

        b_p = tl.exp2(b_s - b_m[:, None]) # [BT, BS]
        b_r = tl.exp2(b_mp - b_m) # rescale factor [BT,]

        b_acc = b_r * b_acc + tl.sum(b_p, 1)

        # here we use broadcast for b_r
        # NOTE: we always cast to original data dtype when we are calculating the direct target
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v) # partial sum of final o, [BT, BV]

    o_q = tl.arange(0, BT) + i_t * BT # offset for q block

    # NOTE: now we need to deal with a particular block which is the i_t-th block, note the condition of i_s/k < T
    for i_s in tl.range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + bos * K, (K, T), (1, K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + bos * V, (T, V), (V, 1), (i_s, 0), (BS, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0,1))
        b_v = tl.load(p_v, boundary_check=(0,1))

        o_k = tl.arange(0, BS) + i_s
        m_k = o_k < T

        b_s = tl.dot(b_q, b_k) * RCP_LN2 # [BT, BS]

        b_s = tl.where((o_q[:, None] >= o_k[None, :] & m_k[None, :]), b_s, float('-inf'))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, axis=1)), b_m

        b_p = tl.exp2(b_s - b_m[:, None]) # [BT, BS]
        b_r = tl.exp2(b_mp - b_m) # rescale factor [BT,]

        b_acc = b_r * b_acc + tl.sum(b_p, 1)

        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v) # partial sum of final o, [BT, BV]
    
    # we rescale o by b_acc in the end, rather in every itertion
    b_o = b_o / b_acc[:, None]
    b_m += tl.log2(b_acc) # lse
    # NOTE: dunno why cast, but should be important to remember
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def flash_attention_naive_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    do,
    dq,
    delta, # D in fa2 paper (page 7)
    scale,
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
    # separating the dq and dkv kernels, for dq we paralleize along row block of q because each q is independent to each other
    # we do this to avoid atmoic addition on dq, the same applies to dkv kernel as well.

    # qk [B, H, T, K]
    # v [B, H, T, V]
    # o [B, H, T, V]
    # do [B, H, T, V]
    # delta [B, H, T]
    # lse [B, H, T]

    # parallize in three dimensions
    # 1. batch * head
    # 2. sequence dimension
    # 3. hidden dimension V

    RCP_LN2: tl.constexpr = 1.4426950216

    i_bh, i_t, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_bh * T

    p_q = tl.make_block_ptr(q + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1,0))
    p_do = tl.make_block_ptr(do + bos * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_delta = tl.make_block_ptr(delta + bos, (T,), (1,), (i_t * BT,), (BT,), (0))
    p_lse = tl.make_block_ptr(lse + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))

    b_q = tl.load(p_q, boundary_check=(0,1)) # [BT, BK]

    b_dq = tl.zeros([BT, BK], dtype=tl.float32) # [BT, BK]

    b_lse = tl.load(p_lse, boundary_check=(0,)) # [BT, ]

    b_do = tl.load(p_do, boundary_check=(0,1)) # [BT, BV]

    b_delta = tl.load(p_delta, boundary_check=(0,)) # [BT,]


    for i_s in tl.range(0, i_t * BT, BS):

        # calculate along columns blocks of K

        p_k = tl.make_block_ptr(k + bos * K, (K, T), (1, K), (0, i_s), (BK, BS), (0, 1)) # k^T
        p_v = tl.make_block_ptr(v + bos * V, (V, T), (1, V), (0, i_s), (BV, BS), (0, 1)) # NOTE: v^T

        b_k = tl.load(p_k, boundary_check=(0,1))
        b_v = tl.load(p_v, boundary_check=(0,1))

        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        
        b_p = tl.exp2(b_s - b_lse[:, None]) # exp(s - m - log(acc)) = exp(s - m) / acc
        b_dp = tl.dot(b_do.to(tl.float32), b_v) # dp = do @ v^T [BT, BS] NOTE: why cast?
        b_ds = b_p * (b_dp - b_delta[:, None]) # [BT, BS]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k)) # [BT, BK] NOTE: why cast?
    
    o_q = tl.arange(0, BT) + i_t * BT

    for i_s in tl.range(i_t * BT, min((i_t + 1) * BT, T), BS):

        p_k = tl.make_block_ptr(k + bos * K, (K, T), (1, K), (0, i_s), (BK, BS), (0, 1)) # k^T
        p_v = tl.make_block_ptr(v + bos * V, (V, T), (1, V), (0, i_s), (BV, BS), (0, 1)) # NOTE: v^T

        b_k = tl.load(p_k, boundary_check=(0,1))
        b_v = tl.load(p_v, boundary_check=(0,1))

        o_k = tl.arange(0, BS) + i_s
        m_k = o_k < T # NOTE: for tl.where masking, important

        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

        b_p = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], tl.exp2(b_s - b_lse[:, None]), 0)
        b_dp = tl.dot(b_do.to(tl.float32), b_v) # dp = do @ v^T [BT, BS] NOTE: why cast?
        b_ds = b_p * (b_dp - b_delta[:, None]) # [BT, BS]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k)) # [BT, BK] NOTE: why cast?
    
    b_dq *= scale # remember this
    tl.store(p_dq, b_dq, boundary_check=(0,1))


@triton.jit
def flash_attention_naive_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    do,
    dk,
    dv,
    delta, # D in fa2 paper (page 7)
    scale,
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
    # separating q and kv because of GQA
    # for dk and dv we paralleize along columns block of k and row block of v because each each k and v is independent of each other
    # we do this to avoid atmoic addition on dk and dv.

    # qk [B, H, T, K]
    # v [B, H, T, V]
    # o [B, H, T, V]
    # do [B, H, T, V]
    # delta [B, H, T]
    # lse [B, H, T]

    # parallize in three dimensions
    # 1. batch * head
    # 2. sequence dimension
    # 3. hidden dimension V

    RCP_LN2: tl.constexpr = 1.4426950216

    i_bh, i_t, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_bh * T

    p_k = tl.make_block_ptr(k + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + bos * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + bos * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0,1))
    b_v = tl.load(p_v, boundary_check=(0,1))

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    o_k = tl.arange(0, BT) + i_t * BT
    m_k = o_k < T

    # NOTE: for p, dp, ds, we are directly calculating their transposes here!

    for i_s in tl.range(i_t * BT, min((i_t + 1) * BT, T), BS):
        # traverse along rows of q with step size of BS, this is only for one block
        # so now i_s is the row of q, o, do, lse and delta
        p_q = tl.make_block_ptr(q + bos * K, (T, K), (K, 1), (i_s, 0), (BS, BK), (1,0))
        p_do = tl.make_block_ptr(do + bos * V, (T, V), (V, 1), (i_s, 0), (BS, BV), (1,0))
        p_lse = tl.make_block_ptr(lse + bos, (T,), (1,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos, (T,), (1,), (i_s,), (BS,), (0,))

        o_q = tl.arange(0, BS) + i_s
        m_q = o_q < T

        b_q = tl.load(p_q, boundary_check=(0,1))
        b_do = tl.load(p_do, boundary_check=(0,1))
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))

        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2 # [BT, BS]
        # NOTE: we have to change the lte brodcast order because we are dealing with tranposes.
        b_p = tl.where((o_q[None, :] >= o_k[:, None]) & m_q[None, :], tl.exp2(b_s - b_lse[None, :]), 0) # [BT, BS]

        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)

        b_dp = tl.dot(b_v, tl.trans(b_do)) # [BT, BS]
        b_ds = b_p * (b_dp - b_delta[None, :]) # [BT, BS]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q) # [BT, BK]

    for i_s in tl.range((i_t + 1) * BT, triton.cdiv(T, BS) * BS, BS):
        # the rest of q rows to be computed
        p_q = tl.make_block_ptr(q + bos * K, (T, K), (K, 1), (i_s, 0), (BS, BK), (1,0))
        p_do = tl.make_block_ptr(do + bos * V, (T, V), (V, 1), (i_s, 0), (BS, BV), (1,0))
        p_lse = tl.make_block_ptr(lse + bos, (T,), (1,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos, (T,), (1,), (i_s,), (BS,), (0,))

        m_q = o_q < T

        b_q = tl.load(p_q, boundary_check=(0,1))
        b_do = tl.load(p_do, boundary_check=(0,1))
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))

        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2 # [BT, BS]
        # NOTE: we have to change the lte brodcast order because we are dealing with tranposes.
        b_p = tl.where(m_q[None, :], tl.exp2(b_s - b_lse[None, :]), 0) # [BT, BS]

        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)

        b_dp = tl.dot(b_v, tl.trans(b_do)) # [BT, BS]
        b_ds = b_p * (b_dp - b_delta[None, :]) # [BT, BS]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q) # [BT, BK]

    b_dk *= scale  # same as dq
    # NOTE: casting
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0,1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0,1))
        


