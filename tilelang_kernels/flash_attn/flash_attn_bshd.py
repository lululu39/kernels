import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as tl
import argparse
import itertools
from functools import partial

# get all posible configs
def get_configs():
    iter_params = dict(BT=[64], BS=[64], num_stages=[1], threads=[128])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[3, 4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flash_attn_mha_fwd_bshd(
    B,
    H,
    T,
    D,
    causal=True,
    BT=64,
    BS=64,
    num_stages=1,
    threads=128,
):
    scale = (1.0 / D) ** 0.5 * 1.4426950216 # log2(e)
    shape = [B, T, H, D]
    dtype = tl.float16
    accum_dtype = tl.float32

    NT = tl.ceildiv(T, BT)
    NS = tl.ceildiv(T, BS) # num iters for kv

    @tl.prim_func
    def main(
        q: tl.Tensor(shape, dtype), # type: ignore
        k: tl.Tensor(shape, dtype), # type: ignore
        v: tl.Tensor(shape, dtype), # type: ignore
        o: tl.Tensor(shape, dtype), # type: ignore
        lse: tl.Tensor((B, T, H), accum_dtype), # type: ignore
    ):
        
        with tl.Kernel(NT, H, B, threads=threads) as (i_t, i_h, i_b):
            
            # Only read: shared
            # Will write: register

            b_q = tl.alloc_shared([BT, D], dtype)
            b_k = tl.alloc_shared([BS, D], dtype)
            b_v = tl.alloc_shared([BS, D], dtype)

            # b_o need writing, so we distinct the memory
            # NOTE: FIXME: tilelang tl.copy may only accept two neigboring memory layout
            b_o_shared = tl.alloc_shared([BT, D], dtype)
            b_o = tl.alloc_fragment([BT, D], accum_dtype)

            b_lse = tl.alloc_shared([BT], accum_dtype)
            b_m = tl.alloc_fragment([BT], accum_dtype)
            b_acc = tl.alloc_fragment([BT], accum_dtype)
            b_mp = tl.alloc_fragment([BT], accum_dtype)
            b_s = tl.alloc_fragment([BT, BS], accum_dtype)
            b_p_cast = tl.alloc_fragment([BT, BS], dtype)
            b_r = tl.alloc_fragment([BT], accum_dtype) # the running scaling factor
            b_lse = tl.alloc_fragment([BT], accum_dtype)
            b_p_sum = tl.alloc_fragment([BT], accum_dtype)

            # NOTE: no explicit boundary check like in triton
            tl.copy(q[i_b, i_t * BT : (i_t + 1) * BT, i_h, :], b_q)
            tl.fill(b_o, 0)
            tl.fill(b_acc, 0)
            tl.fill(b_m, -tl.infinity(accum_dtype)) # -inf

            loop_range = (
                tl.min(NS, tl.ceildiv((i_t + 1) * BT, BS)) if causal else NS
            )

            for k in tl.Pipelined(loop_range, num_stages=num_stages):
                tl.copy(k[i_b, k * BS: (k + 1) * BS, i_h, :], b_k)
                tl.copy(v[i_b, k * BS: (k + 1) * BS, i_h, :], b_v)

                # tl.gemm requires A @ B = C, where the output is added to C rather than assigned.
                # NOTE: in tilelang, we use tl.parallel for element-wise function

                if causal:
                    for i, j in tl.Parallel(BT, BS):
                        b_s[i, j] = tl.if_then_else(i_t * BT + i >= k * BS + k, 0, -tl.infinity(b_s.dtype))
                else:
                    for i, j in tl.Parallel(BT, BS):
                        b_s[i, j] = tl.if_then_else(k * BS + k <= T, 0, -tl.infinity(b_s.dtype))
                
                # the warps are parallized across rows
                tl.gemm(b_q, b_k, b_s, transpose_B=True, policy=tl.GemmWarpPolicy.FullRow)

                tl.copy(b_m, b_mp)

                tl.reduce_max(b_s, b_m, dim=1, clear=False)

                for i in tl.Parallel(BT):
                    b_m[i] = tl.max(b_m[i], b_mp[i])
                
                # NOTE: scale is a scalar and thus moved to elementwise function

                for i in tl.Parallel(BT):
                    b_r[i] = tl.exp2(b_mp[i] * scale - b_m[i] * scale)

                # for b_q, we reuse b_s buffer
                for i, j in tl.Parallel(BT, BS):
                    b_s[i, j] = tl.exp2(b_s[i, j] * scale - b_m[i] * scale)

                
                tl.reduce_sum(b_s, b_p_sum, dim=1)

                for i in tl.Parallel(BT):
                    # update old acc and add new one
                    b_acc[i] = b_acc[i] * b_r[i] + b_p_sum[i]
                
                # cast b_p to b_q dtype
                tl.copy(b_s, b_p_cast) # [BT, BS]

                for i, j in tl.Parallel(BT, BS):
                    b_o[i, j] *= b_r[i]
                
                tl.gemm(b_p_cast, b_v, b_o, policy=T.GemmWarpPolicy.FullRow) # [BT, D]
            
            for i, j in tl.Parallel(BT, D):
                b_o[i, j] /= b_acc[i]
            
            for i in tl.Parallel(BT):
                b_m[i] += tl.log2(b_acc[i])
            
            # register to shared mem
            tl.copy(b_m, b_lse)
            tl.copy(b_o, b_o_shared)

            # shared mem to HBM
            tl.copy(b_o_shared, o[i_b, i_t * BT : (i_t + 1) * BT, i_h, :])
            tl.copy(b_lse, lse[i_b, i_t * BT : (i_t + 1) * BT, i_h])
    

    return main



                
                




