import torch
import triton
import triton.language as tl

from typing import Any, Tuple
import math
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous

@torch.compile
def torch_dyw(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    r: torch.Tensor,
    K: int,
    R: int
) -> torch.Tensor:
    """
    x: torch.Tensor [B, T, D]
    """
    a = a.view(K, -1, R)  # [K, D, R]
    b = b.view(R, K, -1)  # [R, K, D]
    w = torch.matmul(x, r)  # [B, T, K]
    h = torch.einsum('bsd,kdr->kbsr', x, a)
    y = torch.einsum('kbsr,rkd->kbsd', h, b)
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
    D: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    BS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    
    # x: [B, S, M]
    # r: [M, K]
    # a: [KMD, R] [M, R]
    # b: [R, KND] [R, N]
    # y: [B, S, N]

    i_b, i_s, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # x @ r -> s [BS, M] [M, K] -> [BS, K]

    b_s = tl.zeros([BS, K], dtype=tl.float32)

    for i_m in range(0, M, BM):
        p_x = tl.make_block_ptr(x + i_b * S * M, (S, M), (M, 1), (i_s * BS, i_m * BM), (BS, BM), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0,1)) # [BS, BM]

        p_r = tl.make_block_ptr(r, (M, K), (K, 1), (i_m * BM, 0), (BM, K), (1, 0))
        b_r = tl.load(p_r, boundary_check=(0,1))

        b_s += tl.dot(b_x, b_r) # partial sum [BS, K]

    # softmax to get actual score
    b_m = tl.max(b_s, 1) # [BS]
    b_s = tl.exp(b_s - b_m[:, None])
    b_sum = tl.sum(b_s, 1)
    b_s = b_s / b_sum[:, None]
        



backend2impl = {
    # "triton": triton,
    # "tilelang": triton.language,
    "torch": torch_dyw,
}

torch.nn.Linear

class DyWLinear(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        in_scale: int,
        out_scale: int,
        K: int,
        R: int,
        backend: str = "triton",
        bias: bool = False,
        device: Any | None = None,
        dtype: Any | None = None
    ):
        
        super().__init__()

        assert backend in ["triton", "tilelang", "torch"], f"Unsupported backend: {backend}"
        assert bias is False, "Bias is not supported in DyWLinear."
        assert R >= 16, "Rank R must be at least 16 for DyWLinear."

        assert (K * R) <= ((in_scale * out_scale) / (in_scale + out_scale)), (
            f"Product of k and r must be less than or equal to "
            f"(in_scale * out_scale) / (in_scale + out_scale). "
            f"Got k * r = {K * R}, "
            f"(in_scale * out_scale) / (in_scale + out_scale) = "
            f"{(in_scale * out_scale) / (in_scale + out_scale)}."
        )

        self.in_scale = in_scale # m in formula, M = m * D
        self.out_scale = out_scale # n in formula
        self.in_dim = in_scale * dim
        self.out_dim = out_scale * dim
        self.dim = dim
        self.K = K
        self.R = R
        self.backend = backend

        self.r = torch.nn.Parameter(
            torch.randn(self.in_dim, self.K)
        )

        self.a = torch.nn.Parameter(
            torch.randn(K * self.in_dim, R)
        )
        self.b = torch.nn.Parameter(
            torch.randn(R, self.out_dim * K)
        )

        self.init_weights()
    
    def init_weights(self) -> None:

        torch.nn.init.normal_(self.r, mean=0, std=0.01)

        std_a = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.uniform_(self.a, -std_a, std_a)

        std_b = 1.0 / math.sqrt(self.R)
        torch.nn.init.uniform_(self.b, -std_b, std_b)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        impl = backend2impl[self.backend]

        return impl(x, self.a, self.b, self.r, self.K, self.R)
