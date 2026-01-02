import os

import pytest
import torch

from triton_kernels.nsa import native_sparse_attention as tri_nsa

try:
    from fla.ops.nsa import parallel_nsa as fla_nsa
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
        for T in [512, 1024, 1317]
        # for T in [512]
        for bc in [8, 16]
        # for bc in [8]
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

    # create inputs
    q = torch.randn((B, T, HQ, K), dtype=torch.float16, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, K), dtype=torch.float16, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, V), dtype=torch.float16, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, V), dtype=torch.float16, device=device)

    # gating masks to exercise compression and selection branches
    g_cmp = torch.randn((B, T, HQ), dtype=torch.float16, device=device)
    g_slc = torch.randn((B, T, HQ), dtype=torch.float16, device=device)
    # swa left as None as requested
    g_swa = None

    # call triton implementation
    tri = tri_nsa(
        q=q, k=k, v=v,
        g_cmp=g_cmp, g_slc=g_slc, g_swa=g_swa,
        block_indices=None, block_counts=block_counts,
        block_size=64, window_size=0, scale=None
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    ref = fla_nsa(
        q=q, k=k, v=v,
        g_cmp=g_cmp, g_slc=g_slc, g_swa=g_swa,
        block_indices=None, block_counts=block_counts,
        block_size=64, window_size=0, scale=None
    )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    assert_close("o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)