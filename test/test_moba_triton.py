import os

import pytest
import torch

from triton_kernels.moba_triton import mixture_of_block_attention_v1 as tri_moba

from triton_kernels.moba_fa import mixture_of_block_attention as fa_moba

from fla.utils import assert_close, device

@pytest.mark.parametrize(
    ("B", "T", "block_counts"),
    [
        pytest.param(B, T, bc, id=f"B{B}-T{T}-block_counts{bc}")
        for B in [1, 2]
        for T in [1024, 2048]
        for bc in [4]
    ],
)
def test_moba_triton_vs_fa(B: int, T: int, block_counts: int):
    torch.manual_seed(42)
    device_ = device

    H = 2
    G = 4
    HQ = H * G
    K = 32
    V = K
    block_size = 16

    # create inputs
    q = torch.randn((B, T, HQ, K), dtype=torch.float32, device=device_).requires_grad_(True)
    k = torch.randn((B, T, H, K), dtype=torch.float32, device=device_).requires_grad_(True)
    v = torch.randn((B, T, H, V), dtype=torch.float32, device=device_).requires_grad_(True)

    do = torch.randn((B, T, HQ, V), dtype=torch.float32, device=device_)

    cu_seqlens = None

    # call triton implementation
    tri_out = tri_moba(
        q=q, k=k, v=v,
        block_counts=block_counts,
        block_size=block_size,
        scale=None,
        cu_seqlens=cu_seqlens,
    )
    tri_out.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    # test which has nan
    assert not torch.isnan(tri_out).any(), "Triton output has NaN"
    assert not torch.isnan(tri_dq).any(), "Triton dq has NaN"
    assert not torch.isnan(tri_dk).any(), "Triton dk has NaN"
    assert not torch.isnan(tri_dv).any(), "Triton dv has NaN"

    # # prepare inputs for fa (flatten batches into varlen sequence)
    # q_fa = q.view(B * T, HQ, K).detach().requires_grad_(True)
    # k_fa = k.view(B * T, H, K).detach().requires_grad_(True)
    # v_fa = v.view(B * T, H, V).detach().requires_grad_(True)

    # # call reference fa implementation
    # fa_out = fa_moba(
    #     q=q_fa, k=k_fa, v=v_fa,
    #     cu_seqlens=cu_seqlens,
    #     max_seqlen=T,
    #     chunk_size=block_size,
    #     topk=block_counts,
    # )
    # # fa_out shape [B*T, HQ, V] -> reshape to [B, T, HQ, V]
    # fa_out_batched = fa_out.view(B, T, HQ, V)
    # fa_out_batched.backward(do)
    # ref_dq = q_fa.grad.view(B, T, HQ, K).clone(); q_fa.grad = None
    # ref_dk = k_fa.grad.view(B, T, H, K).clone(); k_fa.grad = None
    # ref_dv = v_fa.grad.view(B, T, H, V).clone(); v_fa.grad = None

    # # compare outputs and grads
    # assert_close("o", fa_out_batched, tri_out, rtol=1e-3, atol=1e-3)
    # assert_close("dq", ref_dq, tri_dq, rtol=1e-3, atol=1e-3)
    # assert_close("dk", ref_dk, tri_dk, rtol=1e-3, atol=1e-3)
    # assert_close("dv", ref_dv, tri_dv, rtol=1e-3, atol=1e-3)