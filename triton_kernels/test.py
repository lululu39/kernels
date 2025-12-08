import triton 
import triton.language as tl
import torch

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
    for n in tl.range(0, N, BN):

        # load chunk of each row and column
        x = tl.load(x_block_ptr, boundary_check=(0, 1, 2))
        y = tl.load(y_block_ptr, boundary_check=(0, 1, 2))

        z += tl.dot(x, y)

        x_block_ptr = tl.advance(x_block_ptr, (0, 0, BN)) 
        y_block_ptr = tl.advance(y_block_ptr, (0, BN, 0))
    
    tl.store(z_block_ptr, z, boundary_check=(0, 1, 2))

    return

# test function

B, M, N, P = 32, 32, 32, 32
BB, BM, BN, BP = 16, 16, 16, 16
x = torch.randn(B, M, N, device="cuda", dtype=torch.float32)
y = torch.randn(B, N, P, device="cuda", dtype=torch.float32)
z = torch.empty(B, M, P, device="cuda", dtype=torch.float32)
NB, NM, NN, NP = triton.cdiv(B, BB), triton.cdiv(M, BM), triton.cdiv(N, BN), triton.cdiv(P, BP)

batched_matmul_fwd_kernel[(NB, NM, NP)](
    x,
    y,
    z,
    B,
    M,
    N,
    P,
    BB=BB,
    BM=BM,
    BN=BN,
    BP=BP,
)

print("3D load kernel executed successfully.")
