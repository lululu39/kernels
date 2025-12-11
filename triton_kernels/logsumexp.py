import triton.language as tl
import triton
import torch

@triton.jit
def logsumexp_fwd_kernel(
    x_ptr,
    z_ptr,
    N,
    D: tl.constexpr,
    BD: tl.constexpr
):
    block_id_n, block_id_d = tl.program_id(0), tl.program_id(1)
    off_d = block_id_d * BD + tl.arange(0, BD)
    mask_d = off_d < D

    x = tl.load(x_ptr + block_id_n * D + off_d, mask_d, other=-float('inf'))
    max = tl.max(x, axis=0)
    z = tl.log(tl.sum(tl.exp(x - max), keep_dims=False)) + max
    tl.store(z_ptr + block_id_n * tl.cdiv(D, BD) + block_id_d, z)

def logsumexp_fwd(
    x: torch.Tensor,
):
    shape = x.shape
    # flatten x
    x = x.view(-1, shape[-1])

    N, D, = x.shape
    # NOTE: copied from fla
    BD = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, BD)
    z = torch.empty(N, ND, dtype=torch.float)
    logsumexp_fwd_kernel[(N, ND)](
        x_ptr=x,
        z_ptr=z,
        N=N,
        D=D,
        BD=BD
    )
    z = z.logsumexp(-1).view(*shape[:-1]) # discard last dim
    return z


    