import triton.language as tl
import triton
import torch
from fla.utils import contiguous, autocast_custom_bwd, autocast_custom_fwd

@triton.jit
def batched_matmul_fwd_kernel_naive(
    x_ptr, # [B x M x N]
    y_ptr, # [B x N x P]
    z_ptr, # [B x M x P]
    B,
    M,
    N,
    P,
    BB: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BP: tl.constexpr,
):
    # z = x @ y

    block_id_b = tl.program_id(0) # batch
    block_id_m = tl.program_id(1) # rows of x
    block_id_p = tl.program_id(2) # columns of y

    # offsets
    off_b = block_id_b * BB + tl.arange(0, BB)
    off_m = block_id_m * BM + tl.arange(0, BM)
    off_p = block_id_p * BP + tl.arange(0, BP)

    # masks
    mask_b = off_b < B
    mask_m = off_m < M
    mask_p = off_p < P

    z = tl.zeros([BB, BM, BP], dtype=tl.float32) 

    range_z = off_b[:, None, None] * M * P + off_m[None, :, None] * P + off_p[None, None, :]
    mask_z = mask_b[:, None, None] & mask_m[None, :, None] & mask_p[None, None, :]

    # block nmatmul loop
    for n in tl.range(0, N, BN):
        off_n = n + tl.arange(0, BN)
        mask_n = off_n < N

        range_x = off_b[:, None, None] * M * N + off_m[None, :, None] * N + off_n[None, None, :]
        mask_x = mask_b[:, None, None] & mask_m[None, :, None] & mask_n[None, None, :]

        range_y = off_b[:, None, None] * N * P + off_n[None, :, None] * P + off_p[None, None, :]
        mask_y = mask_b[:, None, None] & mask_n[None, :, None] & mask_p[None, None, :]

        # load chunk of each row and column
        x = tl.load(x_ptr + range_x, mask_x)
        y = tl.load(y_ptr + range_y, mask_y)

        z += tl.dot(x, y)
    
    tl.store(z_ptr + range_z, z, mask_z)

    return

@triton.jit
def batched_matmul_bwd_kernel_dx_naive(
    y_ptr,
    dx_ptr,
    dz_ptr,
    B,
    M,
    N,
    P,
    BB: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BP: tl.constexpr,
):
    # dx = dz @ y^T
    # [M x P] [P x N]

    block_id_b = tl.program_id(0) # batch
    block_id_m = tl.program_id(1) # rows of x
    block_id_n = tl.program_id(2) # columns of x

    # offsets
    off_b = block_id_b * BB + tl.arange(0, BB)
    off_m = block_id_m * BM + tl.arange(0, BM)
    off_n = block_id_n * BN + tl.arange(0, BN)

    # masks
    mask_b = off_b < B
    mask_m = off_m < M
    mask_n = off_n < N

    dx = tl.zeros([BB, BM, BN], dtype=tl.float32)

    range_dx = off_b[:, None, None] * M * N + off_m[None, :, None] * N + off_n[None, None, :]
    mask_dx = mask_b[:, None, None] & mask_m[None, :, None] & mask_n[None, None, :]

    for p in tl.range(0, P, BP):

        off_p = tl.arange(0, BP) + p
        mask_p = off_p < P

        range_dz = off_b[:, None, None] * M * P + off_m[None, :, None] * P + off_p[None, None, :]
        mask_dz = mask_b[:, None, None] & mask_m[None, :, None] & mask_p[None, None, :]

        range_y = off_b[:, None, None] * N * P + off_n[None, :, None] * P + off_p[None, None, :]
        mask_y = mask_b[:, None, None] & mask_n[None, :, None] & mask_p[None, None, :]

        dz = tl.load(dz_ptr + range_dz, mask_dz)
        y = tl.load(y_ptr + range_y, mask_y) 

        dx += tl.dot(dz, tl.trans(y, 0, 2, 1))

    tl.store(dx_ptr + range_dx, dx, mask_dx)

    return


@triton.jit
def batched_matmul_bwd_kernel_dy_naive(
    x_ptr,
    dy_ptr,
    dz_ptr,
    B,
    M,
    N,
    P,
    BB: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BP: tl.constexpr,
):
    # dy = x^T @ dz
    # [N x M] [M x P] -> [N x P]

    block_id_b = tl.program_id(0) # batch
    block_id_n = tl.program_id(1) # rows of y
    block_id_p = tl.program_id(2) # columns of y

    # offsets
    off_b = block_id_b * BB + tl.arange(0, BB)
    off_n = block_id_n * BN + tl.arange(0, BN)
    off_p = block_id_p * BP + tl.arange(0, BP)

    # masks
    mask_b = off_b < B
    mask_n = off_n < N
    mask_p = off_p < P

    dy = tl.zeros([BB, BN, BP], dtype=tl.float32)

    range_dy = off_b[:, None, None] * N * P + off_n[None, :, None] * P + off_p[None, None, :]
    mask_dy = mask_b[:, None, None] & mask_n[None, :, None] & mask_p[None, None, :]

    for m in tl.range(0, M, BM):
        off_m = m + tl.arange(0, BM)
        mask_m = off_m < M

        range_x = off_b[:, None, None] * M * N + off_m[None, :, None] * N + off_n[None, None, :]
        mask_x = mask_b[:, None, None] & mask_m[None, :, None] & mask_n[None, None, :]

        range_dz = off_b[:, None, None] * M * P + off_m[None, :, None] * P + off_p[None, None, :]
        mask_dz = mask_b[:, None, None] & mask_m[None, :, None] & mask_p[None, None, :]

        x = tl.load(x_ptr + range_x, mask_x) # [BB, BM, BN]
        dz = tl.load(dz_ptr + range_dz, mask_dz)

        assert len(x.shape) == 3, f"x.shape is {x.shape}, expected 3 dims"
        
        dy += tl.dot(tl.trans(x, 0, 2, 1), dz)
    
    tl.store(dy_ptr + range_dy, dy, mask_dy)
    return

def batched_matmul_fwd_naive(
    x: torch.Tensor,
    y: torch.Tensor,
):
    # wrapper function for the launch of kernels
    B, M, N, P = *x.shape, y.shape[-1]
    z = torch.zeros(B, M, P, dtype=x.dtype, device=x.device)

    # NOTE: still dunno how to set these hyperparameters, save them for later
    BB = min(32, max(16, triton.next_power_of_2(B)))
    BM = min(32, max(16, triton.next_power_of_2(M)))
    BN = min(32, max(16, triton.next_power_of_2(N)))
    BP = min(32, max(16, triton.next_power_of_2(P)))

    NB = triton.cdiv(B, BB)
    NM = triton.cdiv(M, BM)
    NP = triton.cdiv(P, BP)

    grid = (NB, NM, NP)

    batched_matmul_fwd_kernel_naive[grid](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        B=B,
        M=M,
        N=N,
        P=P,
        BB=BB,
        BM=BM,
        BN=BN,
        BP=BP
    )

    return z


def batched_matmul_bwd_naive(
    x: torch.Tensor,
    y: torch.Tensor,
    dz: torch.Tensor,
):
    
    # for launching bwd kernels

    B, M, N, P = *x.shape, y.shape[-1]
    dx = torch.empty(B, M, N, dtype=x.dtype, device=x.device)
    dy = torch.empty(B, N, P, dtype=y.dtype, device=y.device)

    # NOTE: still dunno how to set these hyperparameters, save them for later
    BB = min(32, max(16, triton.next_power_of_2(B)))
    BM = min(32, max(16, triton.next_power_of_2(M)))
    BN = min(32, max(16, triton.next_power_of_2(N)))
    BP = min(32, max(16, triton.next_power_of_2(P)))

    NB = triton.cdiv(B, BB)
    NM = triton.cdiv(M, BM)
    NN = triton.cdiv(N, BN)
    NP = triton.cdiv(P, BP)

    grid = (NB, NM, NN)

    batched_matmul_bwd_kernel_dx_naive[grid](
        y_ptr=y,
        dx_ptr=dx,
        dz_ptr=dz,
        B=B,
        M=M,
        N=N,
        P=P,
        BB=BB,
        BM=BM,
        BN=BN,
        BP=BP
    )

    grid = (NB, NN, NP)

    batched_matmul_bwd_kernel_dy_naive[grid](
        x_ptr=x,
        dy_ptr=dy,
        dz_ptr=dz,
        B=B,
        M=M,
        N=N,
        P=P,
        BB=BB,
        BM=BM,
        BN=BN,
        BP=BP
    )

    return dx, dy

@torch.compile
class BatchedMatmulFunctionNaive(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, x, y):
        ctx.dtype = x.dtype

        z = batched_matmul_fwd_naive(x, y)

        ctx.save_for_backward(x, y)

        return z.to(x.dtype)
    
    @staticmethod
    @contiguous
    @autocast_custom_bwd    
    def backward(ctx, dz):
        x, y = ctx.saved_tensors

        dx, dy = batched_matmul_bwd_naive(x, y, dz)

        return dx, dy

def batched_matmul_naive(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return BatchedMatmulFunctionNaive.apply(x, y)

@triton.jit
def batched_matmul_fwd_kernel(
    x_ptr, # [B x M x N]
    y_ptr, # [B x N x P]
    z_ptr, # [B x M x P]
    B,
    M,
    N,
    P,
    BB: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BP: tl.constexpr,
):

    block_id_b = tl.program_id(0) # batch
    block_id_m = tl.program_id(1) # rows of x
    block_id_p = tl.program_id(2) # columns of y

    # offsets
    off_b = block_id_b * BB
    off_m = block_id_m * BM 
    off_p = block_id_p * BP

    z = tl.zeros([BB, BM, BP], dtype=tl.float32) 

    z_block_ptr = tl.make_block_ptr(
        z_ptr,
        shape=(B, M, P),
        strides=(M * P, P, 1),
        offsets=(off_b, off_m, off_p),
        block_shape=(BB, BM, BP),
        order=(0, 1, 2)
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(B, M, N),
        strides=(M * N, N, 1),
        offsets=(off_b, off_m, 0),  
        block_shape=(BB, BM, BN),
        order=(0, 1, 2)
    )

    y_block_ptr = tl.make_block_ptr(
        y_ptr,
        shape=(B, N, P),
        strides=(N * P, P, 1),
        offsets=(off_b, 0, off_p),  
        block_shape=(BB, BN, BP),
        order=(0, 1, 2)
    )

    # block nmatmul loop
    for _ in tl.range(0, N, BN):

        # load chunk of each row and column
        x = tl.load(x_block_ptr, boundary_check=(0, 1, 2))
        y = tl.load(y_block_ptr, boundary_check=(0, 1, 2))

        z += tl.dot(x, y)

        x_block_ptr = tl.advance(x_block_ptr, (0, 0, BN)) 
        y_block_ptr = tl.advance(y_block_ptr, (0, BN, 0))
    
    tl.store(z_block_ptr, z, boundary_check=(0, 1, 2))

    return

@triton.jit
def batched_matmul_bwd_kernel_dx(
    y_ptr,
    dx_ptr,
    dz_ptr,
    B,
    M,
    N,
    P,
    BB: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BP: tl.constexpr,
):
    # dx = dz @ y^T
    # [M x P] [P x N]

    block_id_b = tl.program_id(0) # batch
    block_id_m = tl.program_id(1) # rows of x
    block_id_n = tl.program_id(2) # columns of x

    # offsets
    off_b = block_id_b * BB
    off_m = block_id_m * BM
    off_n = block_id_n * BN

    dx = tl.zeros([BB, BM, BN], dtype=tl.float32)

    dx_block_ptr = tl.make_block_ptr(
        dx_ptr,
        shape=(B, M, N),
        strides=(M * N, N, 1),
        offsets=(off_b, off_m, off_n),
        block_shape=(BB, BM, BN),
        order=(0, 1, 2)
    )

    dz_block_ptr = tl.make_block_ptr(
        dz_ptr,
        shape=(B, M, P),
        strides=(M * P, P, 1),
        offsets=(off_b, off_m, 0),  
        block_shape=(BB, BM, BP),
        order=(0, 1, 2)
    )

    y_block_ptr = tl.make_block_ptr(
        y_ptr,
        shape=(B, N, P),
        strides=(N * P, P, 1),
        offsets=(off_b, off_n, 0),  
        block_shape=(BB, BN, BP),
        order=(0, 1, 2)
    )

    for _ in tl.range(0, P, BP):

        dz = tl.load(dz_block_ptr, boundary_check=(0, 1, 2))
        y = tl.load(y_block_ptr, boundary_check=(0, 1, 2))

        dx += tl.dot(dz, tl.trans(y, 0, 2, 1))

        dz_block_ptr = tl.advance(dz_block_ptr, (0, 0, BP))
        y_block_ptr = tl.advance(y_block_ptr, (0, 0, BP))

    tl.store(dx_block_ptr, dx, boundary_check=(0, 1, 2))

    return


@triton.jit
def batched_matmul_bwd_kernel_dy(
    x_ptr,
    dy_ptr,
    dz_ptr,
    B,
    M,
    N,
    P,
    BB: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BP: tl.constexpr,
):
    # dy = x^T @ dz
    # [N x M] [M x P] -> [N x P]

    block_id_b = tl.program_id(0) # batch
    block_id_n = tl.program_id(1) # rows of y
    block_id_p = tl.program_id(2) # columns of y

    # offsets
    off_b = block_id_b * BB
    off_n = block_id_n * BN
    off_p = block_id_p * BP

    dy = tl.zeros([BB, BN, BP], dtype=tl.float32)

    dy_block_ptr = tl.make_block_ptr(
        dy_ptr,
        shape=(B, N, P),
        strides=(N * P, P, 1),
        offsets=(off_b, off_n, off_p),
        block_shape=(BB, BN, BP),
        order=(0, 1, 2)
    )

    dz_block_ptr = tl.make_block_ptr(
        dz_ptr,
        shape=(B, M, P),
        strides=(M * P, P, 1),
        offsets=(off_b, 0, off_p),  
        block_shape=(BB, BM, BP),
        order=(0, 1, 2)
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(B, M, N),
        strides=(M * N, N, 1),
        offsets=(off_b, 0, off_n),  
        block_shape=(BB, BM, BN),
        order=(0, 1, 2)
    )

    for _ in tl.range(0, M, BM):
        
        x = tl.load(x_block_ptr, boundary_check=(0, 1, 2))
        dz = tl.load(dz_block_ptr, boundary_check=(0, 1, 2))
        
        dy += tl.dot(tl.trans(x, 0, 2, 1), dz)

        x_block_ptr = tl.advance(x_block_ptr, (0, BM, 0))
        dz_block_ptr = tl.advance(dz_block_ptr, (0, BM, 0))
    
    tl.store(dy_block_ptr, dy, boundary_check=(0, 1, 2))
    return

def batched_matmul_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
):
    # wrapper function for the launch of kernels
    B, M, N, P = *x.shape, y.shape[-1]
    z = torch.zeros(B, M, P, dtype=x.dtype, device=x.device)

    # NOTE: still dunno how to set these hyperparameters, save them for later
    BB = min(32, max(16, triton.next_power_of_2(B)))
    BM = min(32, max(16, triton.next_power_of_2(M)))
    BN = min(32, max(16, triton.next_power_of_2(N)))
    BP = min(32, max(16, triton.next_power_of_2(P)))

    NB = triton.cdiv(B, BB)
    NM = triton.cdiv(M, BM)
    NP = triton.cdiv(P, BP)

    grid = (NB, NM, NP)

    batched_matmul_fwd_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        B=B,
        M=M,
        N=N,
        P=P,
        BB=BB,
        BM=BM,
        BN=BN,
        BP=BP
    )

    return z


def batched_matmul_bwd(
    x: torch.Tensor,
    y: torch.Tensor,
    dz: torch.Tensor,
):
    
    # for launching bwd kernels

    B, M, N, P = *x.shape, y.shape[-1]
    dx = torch.empty(B, M, N, dtype=x.dtype, device=x.device)
    dy = torch.empty(B, N, P, dtype=y.dtype, device=y.device)

    # NOTE: still dunno how to set these hyperparameters, save them for later
    BB = min(32, max(16, triton.next_power_of_2(B)))
    BM = min(32, max(16, triton.next_power_of_2(M)))
    BN = min(32, max(16, triton.next_power_of_2(N)))
    BP = min(32, max(16, triton.next_power_of_2(P)))

    NB = triton.cdiv(B, BB)
    NM = triton.cdiv(M, BM)
    NN = triton.cdiv(N, BN)
    NP = triton.cdiv(P, BP)

    grid = (NB, NM, NN)

    batched_matmul_bwd_kernel_dx[grid](
        y_ptr=y,
        dx_ptr=dx,
        dz_ptr=dz,
        B=B,
        M=M,
        N=N,
        P=P,
        BB=BB,
        BM=BM,
        BN=BN,
        BP=BP
    )

    grid = (NB, NN, NP)

    batched_matmul_bwd_kernel_dy[grid](
        x_ptr=x,
        dy_ptr=dy,
        dz_ptr=dz,
        B=B,
        M=M,
        N=N,
        P=P,
        BB=BB,
        BM=BM,
        BN=BN,
        BP=BP
    )

    return dx, dy

@torch.compile
class BatchedMatmulFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, x, y):
        ctx.dtype = x.dtype

        z = batched_matmul_fwd(x, y)

        ctx.save_for_backward(x, y)

        return z.to(x.dtype)
    
    @staticmethod
    @contiguous
    @autocast_custom_bwd    
    def backward(ctx, dz):
        x, y = ctx.saved_tensors

        dx, dy = batched_matmul_bwd(x, y, dz)

        return dx, dy

def batched_matmul(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return BatchedMatmulFunction.apply(x, y)