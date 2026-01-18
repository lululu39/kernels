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
    out_idx=[3],
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
    ):
        
        with tl.Kernel(NT, H, B, threads=threads) as (i_t, i_h, i_b):
            
            # Only read: shared
            # Will write: register

            b_q = tl.alloc_shared([BT, D], dtype)
            b_k = tl.alloc_shared([BS, D], dtype)
            b_v = tl.alloc_shared([BS, D], dtype)

            # b_o need writing, so we distinct the memory
            b_o_shared = tl.alloc_shared([BT, D], dtype)
            b_o = tl.alloc_fragment([BT, D], accum_dtype)

            b_m = tl.alloc_fragment([BT], accum_dtype)
            b_acc = tl.alloc_fragment([BT], accum_dtype)
            b_mp = tl.alloc_fragment([BT], accum_dtype)
            b_s = tl.alloc_fragment([BT, BS], accum_dtype)
            b_s_cast = tl.alloc_fragment([BT, BS], dtype)
            b_r = tl.alloc_fragment([BT], accum_dtype) # the running scaling factor
            b_lse = tl.alloc_fragment([BT], accum_dtype)
            b_s_sum = tl.alloc_fragment([BT], accum_dtype)

            tl.copy(q[i_b, i_t * BT : (i_t + 1) * BT, i_h, :], b_q)
            tl.fill(b_o, 0)
            tl.fill(b_acc, 0)
            tl.fill(b_m, -tl.infinity(accum_dtype)) # -inf

            loop_range = (
                tl.min(NS, tl.ceildiv((i_t + 1) * BT, BS)) if causal else NS
            )

            for k in tl.Pipelined(loop_range, num_stages=num_stages):
                tl.copy(k[i_b, k * BS: (k + 1) * BS, i_h, :], b_k)

                




