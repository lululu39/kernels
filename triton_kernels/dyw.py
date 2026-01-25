import torch
import triton
import triton.language as tl

from typing import Any, Tuple
import math
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous


import torch.nn as nn

@torch.compile
def torch_dyw(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    r: torch.Tensor,
    K: int,
    R: int
) -> torch.Tensor:
    
    w = torch.matmul(x, r)  # [B, T, K]
    h = torch.einsum('bsd,kdr->kbsr', x, a)
    y = torch.einsum('kbsr,krd->kbsd', h, b)
    y = torch.einsum('bsk,kbsd->bsd', w.softmax(dim=-1), y)

    return y

@triton.jit
def triton_dyw_fwd_kernel(
    x,
    y,
    a,
    b,
    r,
    B: tl.constexpr,
    S: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    BS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    
    # x: [B, S, M]
    # r: [M, K]
    # a: [K, M, R]
    # b: [K, R, N]
    # y: [B, S, N]

    i_b, i_s, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # x @ r -> s [BS, M] [M, K] -> [BS, K]

    b_s = tl.zeros([BS, K], dtype=tl.float32)

    for i_m in range(0, M, BM):
        p_x = tl.make_block_ptr(x + i_b * S * M, (S, M), (M, 1), (i_s * BS, i_m * BM), (BS, BM), (1,0))
        b_x = tl.load(p_x, boundary_check=(0,1)) # [BS, BM]

        p_r = tl.make_block_ptr(r, (M, K), (K, 1), (i_m * BM, 0), (BM, K), (1, 0))
        b_r = tl.load(p_r, boundary_check=(0,1))

        b_s += tl.dot(b_x, b_r) # partial sum [BS, K]

    # softmax to get actual score
    b_m = tl.max(b_s, 1) # [BS]
    b_s = tl.exp(b_s - b_m[:, None])
    b_sum = tl.sum(b_s, 1)
    b_s = b_s / b_sum[:, None] # [BS, K]

    # split into k path to calculate x @ a -> h -> h @ b = y
    # [BS, M] [M, R] -> [BS, R] -> [BS, N]

    p_y = tl.make_block_ptr(y + i_b * S * N, (S, N), (N, 1), (i_s * BS, i_n * BN), (BS, BN), (1,0))

    b_y = tl.zeros([BS, BN], dtype=tl.float32)

    for i_k in range(0, K):

        b_h = tl.zeros([BS, R], dtype=tl.float32)
        # NOTE: we for loop k, since we do not want to do atmoic add

        # x @ a -> h

        for i_m in range(0, M, BM):
            p_x = tl.make_block_ptr(x + i_b * S * M, (S, M), (M, 1), (i_s * BS, i_m * BM), (BS, BM), (1, 0))
            b_x = tl.load(p_x, boundary_check=(0,1)) # [BS, BM]

            p_a = tl.make_block_ptr(a + i_k * M * R, (M, R), (R, 1), (i_m * BM, 0), (BM, R), (1,0))
            b_a = tl.load(p_a, boundary_check=(0,1))

            b_h += tl.dot(b_x, b_a) # [BM, R]

        p_b = tl.make_block_ptr(b + i_k * R * N, (R, N), (N, 1), (0, i_n * BN), (R, BN), (1, 0))

        b_b = tl.load(p_b, boundary_check=(0,1))
        
        # score[k] * h @ b

        b_s_k = tl.sum(b_s * (tl.arange(0, K) == i_k)[None, :], axis=1)

        b_y += b_s_k[:, None] * tl.dot(b_h, b_b) # [BS, BN]
    
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0,1))


def triton_dyw_fwd(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    r: torch.Tensor,
    K: int,
    R: int  
):
    B, S, M, N = *x.shape, b.shape[-1]

    BS = 128

    BM = 64

    BN = 64

    y = torch.empty(B, S, N, dtype=x.dtype, device=x.device)

    grid = (B, triton.cdiv(S, BS), triton.cdiv(N, BN))

    triton_dyw_fwd_kernel[grid](
        x=x,
        y=y,
        a=a,
        b=b,
        r=r,
        B=B,
        S=S,
        M=M,
        N=N,
        K=K,
        R=R,
        BS=BS,
        BM=BM,
        BN=BN,
    )

    return y

backend2impl = {
    # "triton": triton,
    # "tilelang": triton.language,
    "torch": torch_dyw,
}

class DyWLinear(nn.Module):
    def __init__(self, dim, r=4, in_s=1, out_s=1, backend="torch"):
        super().__init__()
        
        k = int((in_s * out_s * dim) / ((in_s + out_s) * r))

        # k < sqrt(ab * dim / (a + b))
        # k < r

        self.in_dim = dim * in_s
        self.out_dim = dim * out_s
        self.K = k
        self.R = r

        self.r = nn.Parameter(torch.randn(self.in_dim, k))
        self.a = nn.Parameter(torch.randn(k, self.in_dim, r))

        self.b = nn.Parameter(torch.randn(k, r, self.out_dim))

        self.backend = backend

        torch.nn.init.normal_(self.r, mean=0, std=0.01)

        std_a = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.uniform_(self.a, -std_a, std_a)

        std_b = 1.0 / math.sqrt(self.R)
        torch.nn.init.uniform_(self.b, -std_b, std_b)

    # @torch.compile
    def forward(self, x):

        impl = backend2impl[self.backend]
        return impl(x, self.a, self.b, self.r, self.K, self.R)