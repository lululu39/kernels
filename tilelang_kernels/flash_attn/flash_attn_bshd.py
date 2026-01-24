import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as tl
import argparse
import itertools
from functools import partial
from flash_attn.flash_attn_interface import _flash_attn_forward

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
def flash_attn_fwd_bshd(
    B,
    HQ,
    H,
    G,
    T,
    D,
    causal=True,
    BT=64,
    BS=64,
    num_stages=1,
    threads=128,
):
    scale = (1.0 / D) ** 0.5 * 1.4426950216 # log2(e)
    # scale = (1.0 / D) ** 0.5 
    shape_qo = [B, T, HQ, D]
    shape_kv = [B, T, H, D]
    dtype = tl.float16
    accum_dtype = tl.float32

    NT = tl.ceildiv(T, BT)
    NS = tl.ceildiv(T, BS) # num iters for kv

    @tl.prim_func
    def main(
        q: tl.Tensor(shape_qo, dtype), # type: ignore
        k: tl.Tensor(shape_kv, dtype), # type: ignore
        v: tl.Tensor(shape_kv, dtype), # type: ignore
        o: tl.Tensor(shape_qo, dtype), # type: ignore
        lse: tl.Tensor((B, T, HQ), accum_dtype), # type: ignore
    ):
        
        with tl.Kernel(NT, HQ, B, threads=threads) as (i_t, i_hq, i_b):
            
            # Only read and large: shared
            # Will write: register

            i_h = i_hq // G

            b_q = tl.alloc_shared([BT, D], dtype)
            b_k = tl.alloc_shared([BS, D], dtype)
            b_v = tl.alloc_shared([BS, D], dtype)

            b_o = tl.alloc_fragment([BT, D], accum_dtype)
            b_m = tl.alloc_fragment([BT], accum_dtype)
            b_acc = tl.alloc_fragment([BT], accum_dtype)
            b_mp = tl.alloc_fragment([BT], accum_dtype)
            b_s = tl.alloc_fragment([BT, BS], accum_dtype)
            b_p_cast = tl.alloc_fragment([BT, BS], dtype)
            b_r = tl.alloc_fragment([BT], accum_dtype) # the running scaling factor
            b_p_sum = tl.alloc_fragment([BT], accum_dtype)

            # NOTE: no explicit boundary check like in triton, but automatically generated
            tl.copy(q[i_b, i_t * BT : (i_t + 1) * BT, i_hq, :], b_q)

            # NOTE: scale is a elementwise op, so we use tl.Parallel
            for i, j in tl.Parallel(BT, D):
                b_q[i, j] *= scale

            tl.fill(b_o, 0)
            tl.fill(b_acc, 0)
            tl.fill(b_m, -tl.infinity(accum_dtype)) # -inf

            loop_range = (
                tl.min(NS, tl.ceildiv((i_t + 1) * BT, BS)) if causal else NS
            )

            for i_s in tl.Pipelined(loop_range, num_stages=num_stages):
                tl.copy(k[i_b, i_s * BS: (i_s + 1) * BS, i_h, :], b_k)
                tl.copy(v[i_b, i_s * BS: (i_s + 1) * BS, i_h, :], b_v)

                # NOTE: we can also pre-mask b_s as below to save a tl.clear op

                # if causal:
                #     for i, j in tl.Parallel(BT, BS):
                #         b_s[i, j] = tl.if_then_else(i_t * BT + i >= i_s * BS + j, 0, -tl.infinity(b_s.dtype))
                # else:
                #     for i, j in tl.Parallel(BT, BS):
                #         b_s[i, j] = tl.if_then_else(i_s * BS + j < T, 0, -tl.infinity(b_s.dtype))
                
                # the warps are parallized across rows

                tl.clear(b_s) # NOTE: not used if we pre-mask

                # NOTE: tl.gemm requires A @ B = C, where the output is added to C rather than assigned.

                # NOTE: tl.gemm or triton's tl.dot WILL output float output, so in tilelang we explicity set the precision 
                # of output buffer to float.

                tl.gemm(b_q, b_k, b_s, transpose_B=True, policy=tl.GemmWarpPolicy.FullRow)

                # NOTE: in tilelang, we use tl.parallel for element-wise function
                if causal:
                    for i, j in tl.Parallel(BT, BS):
                        b_s[i, j] = tl.if_then_else(i_t * BT + i >= i_s * BS + j, b_s[i, j], -tl.infinity(b_s.dtype))
                else:
                    for i, j in tl.Parallel(BT, BS):
                        b_s[i, j] = tl.if_then_else(i_s * BS + j < T, b_s[i, j], -tl.infinity(b_s.dtype))

                tl.copy(b_m, b_mp)

                tl.reduce_max(b_s, b_m, dim=1, clear=False)

                for i in tl.Parallel(BT):
                    b_m[i] = tl.max(b_m[i], b_mp[i])

                for i in tl.Parallel(BT):
                    b_r[i] = tl.exp2(b_mp[i] - b_m[i])

                # for b_p, we reuse b_s buffer
                for i, j in tl.Parallel(BT, BS):
                    b_s[i, j] = tl.exp2(b_s[i, j] - b_m[i])

                
                tl.reduce_sum(b_s, b_p_sum, dim=1)

                for i in tl.Parallel(BT):
                    # update old acc and add new one
                    b_acc[i] = b_acc[i] * b_r[i] + b_p_sum[i]
                
                # cast b_p to b_q dtype
                # NOTE: Create a new buffer to cast layout (Accum -> Operand) and dtype (fp32 -> fp16) for the next GEMM.
                # NOTE: because GEMM output layout is different from inout layout, and we cannot use the same fragment because the layout mismatch
                tl.copy(b_s, b_p_cast) # [BT, BS]

                for i, j in tl.Parallel(BT, BS):
                    b_o[i, j] *= b_r[i]
                
                tl.gemm(b_p_cast, b_v, b_o, policy=tl.GemmWarpPolicy.FullRow) # [BT, D]
            
            for i, j in tl.Parallel(BT, D):
                b_o[i, j] /= b_acc[i]
            
            for i in tl.Parallel(BT):
                # no matter in log2 or loge, we all use this calculation
                b_m[i] = b_m[i] + tl.log2(b_acc[i])
            
            # register to HBM
            # NOTE: tilelang will automatically inject boundary check based on the index and the shape
            tl.copy(b_o, o[i_b, i_t * BT : (i_t + 1) * BT, i_hq, :])
            tl.copy(b_m, lse[i_b, i_t * BT : (i_t + 1) * BT, i_hq])
    
    return main

@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flash_attn_bwd_bshd_preprocess(
    B,
    HQ,
    T,
    D,
    BTD,
):
    dtype = tl.float16
    accum_dtype = tl.float32
    shape = [B, T, HQ, D]

    @tl.prim_func
    def main(
        o: tl.Tensor(shape, dtype), # type: ignore
        do: tl.Tensor(shape, dtype), # type: ignore
        delta: tl.Tensor([B, T, HQ], accum_dtype), # type: ignore
    ):
        
        with tl.Kernel(B, tl.ceildiv(T, BTD), HQ) as (i_b, i_t, i_hq):

            # b_o = tl.alloc_shared([BTD, BTD], dtype)
            # b_do = tl.alloc_shared([BTD, BTD], dtype)
            # NOTE: FIXME: i think it is best to put all vars in register, but 
            # in fwd we are constrained by the register size
            # now, the BTD is small so we directly use register mem

            b_o = tl.alloc_fragment([BTD, BTD], dtype)
            b_do = tl.alloc_fragment([BTD, BTD], dtype)
            b_acc = tl.alloc_fragment([BTD, BTD], accum_dtype)
            b_delta = tl.alloc_fragment([BTD], accum_dtype)

            tl.clear(b_acc)

            for k in range(tl.ceildiv(D, BTD)):
                tl.copy(o[i_b, i_t * BTD : (i_t + 1) * BTD, i_hq, k * BTD : (k + 1) * BTD], b_o)
                tl.copy(do[i_b, i_t * BTD : (i_t + 1) * BTD, i_hq, k * BTD : (k + 1) * BTD], b_do)
                
                for i, j in tl.Parallel(BTD, BTD):
                    b_acc[i, j] += b_o[i, j] * b_do[i, j]
            
            tl.reduce_sum(b_acc, b_delta, dim=1) # [BTD,]

            tl.copy(b_delta, delta[i_b, i_t * BTD: (i_t + 1) * BTD, i_hq])

    return main

        
    
def flash_attn_bwd_bshd_dq(
    B,
    HQ,
    H,
    G,
    T,
    D,
    causal=True,
    BT=64,
    BS=64,
    num_stages=1,
    threads=128,
):
    
    scale = (1.0 / D) ** 0.5 * 1.4426950216 # log2(e)
    # scale = (1.0 / D) ** 0.5 
    shape_qo = [B, T, HQ, D]
    shape_kv = [B, T, H, D]
    dtype = tl.float16
    accum_dtype = tl.float32

    NT = tl.ceildiv(T, BT)
    NS = tl.ceildiv(T, BS) # num iters for kv

    # NOTE: read param first, write param later

    @tl.prim_func
    def main(
        q: tl.Tensor(shape_qo, dtype), # type: ignore
        k: tl.Tensor(shape_kv, dtype), # type: ignore
        v: tl.Tensor(shape_kv, dtype), # type: ignore
        do: tl.Tensor(shape_qo, dtype), # type: ignore
        lse: tl.Tensor((B, T, HQ), accum_dtype), # type: ignore
        delta: tl.Tensor((B, T, HQ), accum_dtype), # type: ignore
        dq: tl.Tensor(shape_qo, dtype), # type: ignore
    ):
        
        with tl.Kernel(NT, HQ, B, threads=threads) as (i_t, i_hq, i_b):

            # Only read: shared
            # Will write: register

            i_h = i_hq // G

            b_q = tl.alloc_shared([BT, D], dtype)
            b_k = tl.alloc_shared([BS, D], dtype)
            b_v = tl.alloc_shared([BS, D], dtype)
            b_do = tl.alloc_shared([BT, D], dtype)
            b_lse = tl.alloc_shared([BT], accum_dtype)

            b_dq = tl.alloc_fragment([BT, D], dtype)

            b_s = tl.alloc_fragment([BT, BS], accum_dtype)

            # copy q and merge scale into q
            tl.copy(q[i_b, i_t * BT : (i_t + 1) * BT, i_hq, :], b_q)

            for i, j in tl.Parallel(BT, D):
                b_q[i, j] *= scale

            loop_range = (
                tl.min(NS, tl.ceildiv((i_t + 1) * BT, BS)) if causal else NS
            )

            for i_s in tl.Pipelined(loop_range, num_stages=num_stages):
                tl.copy(k[i_b, i_s * BS: (i_s + 1) * BS, i_h, :], b_k)
                tl.copy(v[i_b, i_s * BS: (i_s + 1) * BS, i_h, :], b_v)


    
    

def ref_attn_fwd_bshd(Q, K, V, causal):
    dim = Q.size(-1)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)
    lse = torch.logsumexp(scores, dim=-1) * 1.4426950216 # log2(exp sum)
    lse = lse.permute(0, 2, 1).to(torch.float32) # (B, H, T) -> (B, T, H)
    return output, lse

def test_fwd(
    batch: int = 8,
    heads_q: int = 32,
    heads: int = 32,
    seq_len: int = 4096,
    dim: int = 128,
    causal: bool = False,
    tune: bool = False,
):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5

    if not tune:
        kernel = flash_attn_fwd_bshd(batch, heads_q, heads, heads_q // heads, seq_len, dim, causal, BT=128, BS=128, num_stages=1, threads=128)
        ref_program_processed = partial(ref_attn_fwd_bshd, causal=causal)
        profiler = kernel.get_profiler()
        profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
        print("All checks pass.")
        latency = profiler.do_bench(ref_program_processed, warmup=500)
        print("Ref: {:.2f} ms".format(latency))
        print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        latency = profiler.do_bench(warmup=500)
        print("Tile-lang: {:.2f} ms".format(latency))
        print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    else:
        best_result = flash_attn_fwd_bshd(batch, heads_q, heads, heads_q // heads, seq_len, dim, causal)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2, help="batch size")
    parser.add_argument("--heads", type=int, default=16, help="heads")
    parser.add_argument("--groups", type=int, default=1, help="groups")
    parser.add_argument("--seq_len", type=int, default=4096, help="sequence length")
    parser.add_argument("--dim", type=int, default=32, help="dim")
    parser.add_argument("--causal", action="store_true", help="causal")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    args = parser.parse_args()
    test_fwd(args.batch, args.heads * args.groups, args.heads, args.seq_len, args.dim, args.causal, args.tune)