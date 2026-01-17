import tilelang

import tilelang.language as tl

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, BM, BN, BK, dtype=tl.float16, accum_type=tl.float32):
    @tl.prim_func
    def gemm(
        A: tl.Tensor((M, K), dtype), # type: ignore
        B: tl.Tensor((K, N), dtype), # type: ignore
        C: tl.Tensor((M, N), dtype), # type: ignore
    ):
        # A, B, C are in HBM global memory
        # define a grid
        with tl.Kernel(tl.ceildiv(N, BN), tl.ceildiv(M, BM), threads=128) as (i_n, i_m):
            
            # preallocate memory for a and b (tiles of A and B to used in SRAM)
            a = tl.alloc_shared((BM, BK), dtype)
            b = tl.alloc_shared((BK, BN), dtype)

            # the accumulation is allocated to register
            c = tl.alloc_fragment((BM, BN), accum_type)

            tl.clear(c) # make it all zero

            # loop over K to calculate, using num_stage=3 pipeline

            for k in tl.Pipelined(0, tl.ceildiv(K, BK), num_stages=3):

                tl.copy(A[i_m * BM, k * BK], a) # [BM, BK]
                tl.copy(B[k * BK, i_n * BN], b) # [BK, BN]

                tl.gemm(a, b, c)
            
            tl.copy(c, C[i_m * BM, i_n * BN])
    
    return gemm

def main():

    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

    c = kernel(a, b)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # # Get CUDA Source
    # print("CUDA Source:")
    # print(kernel.get_kernel_source())

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    # latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")

if __name__ == "__main__":
    main()