import os

import pytest
import torch
import triton
from triton_kernels.nsa import nsa_compression as tri_nsa
from triton_kernels.nsa import nsa_topk as tri_topk
from triton_kernels.pooling import my_mean_pooling

try:
    from fla.ops.nsa.compression import parallel_nsa_compression as fla_nsa
    from fla.ops.nsa.parallel import parallel_nsa_topk as fla_topk
    HAS_FLA = True
except Exception:
    HAS_FLA = False

from fla.ops.utils import prepare_lens
from fla.utils import assert_close, check_shared_mem, device

# parametrize B (1..2), T (512,1024,1317), block_counts (8,16) => 2*3*2 = 12 cases
@pytest.mark.parametrize(
    ("B", "T", "block_counts"),
    [
        pytest.param(B, T, bc, id=f"B{B}-T{T}-block_counts{bc}")
        for B in [1, 2]
        for T in [512, 1024, 1317, 2345]
        for bc in [8]
    ],
)
def test_nsa_compression_selection_equivalence(B: int, T: int, block_counts: int):
    if not HAS_FLA:
        pytest.skip(reason="Skipping test because reference fla.ops.nsa is not available")
    # some triton kernels require larger shared mem on newer GPUs for big K/V
    K = 64
    if not check_shared_mem('hopper') and K > 128:
        pytest.skip(reason="Skip test, do not have enough shared mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    # fixed model dims for tests
    H = 2
    G = 16
    V = 64
    HQ = H * G
    scale = K ** -0.5

    # create inputs
    q = torch.randn((B, T, HQ, K), dtype=torch.float16, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, K), dtype=torch.float16, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, V), dtype=torch.float16, device=device).requires_grad_(True)

    k = my_mean_pooling(k, 64)
    v = my_mean_pooling(v, 64)
    
    do = torch.randn((B, T, HQ, V), dtype=torch.float16, device=device)


    q1, k1, v1 = q.clone().detach().requires_grad_(True), \
             k.clone().detach().requires_grad_(True), \
             v.clone().detach().requires_grad_(True)

    tri = tri_nsa(q1, k1, v1)
    tri[0].backward(do)
    tri_lse = tri[1]
    tri_dq, tri_dk, tri_dv = q1.grad, k1.grad, v1.grad

    indices1 = tri_topk(
        q=q1, k=k1,
        lse=tri_lse,
        block_counts=16,
        block_size=64,
        scale=scale
    )

    q2, k2, v2 = q.clone().detach().requires_grad_(True), \
                k.clone().detach().requires_grad_(True), \
                v.clone().detach().requires_grad_(True)

    ref = fla_nsa(q2, k2, v2)
    ref[0].backward(do)
    ref_lse = ref[1]
    indices2 = fla_topk(
        q=q2, k=k2, 
        lse=ref_lse,
        block_counts=16,
        block_size=64,
        scale=scale
    )
    ref_dq, ref_dk, ref_dv = q2.grad, k2.grad, v2.grad

    assert_close("o", ref[0], tri[0], 0.005)
    assert_close("lse", ref_lse, tri_lse, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)

    assert_close("dv", ref_dv, tri_dv, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)

    # print differnce in indices
    diff = (indices1 - indices2).abs().sum().item()
    assert diff == 0, f"TopK indices differ by {diff} elements"