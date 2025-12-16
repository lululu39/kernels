import os

import pytest
import torch

from triton_kernels.fa2 import flash_attention_2
from fla.ops.utils import prepare_lens
from fla.utils import assert_close, check_shared_mem, device

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except Exception:
    HAS_FLASH = False


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'causal'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-causal{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1.0, True),
            (3, 111, 2, 100, 1.0, True),
            (3, 1024, 2, 60, 0.1, True),
            (3, 1024, 2, 128, 0.1, True),
            (4, 2048, 2, 64, 0.1, True),
            (1, 63, 1, 64, 1.0, False),
            (3, 111, 2, 100, 1.0, False),
            (3, 1024, 2, 60, 0.1, False),
            (3, 1024, 2, 128, 0.1, False),
            (4, 2048, 2, 64, 0.1, False),
        ]
    ],
)
def test_parallel(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    causal: bool,
):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shard mem")
    if not HAS_FLASH:
        pytest.skip(reason="Skipping test because flash-attn is not installed")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    q = torch.randn((B, H, T, D), dtype=torch.float16, device=device).requires_grad_(True)
    k = torch.randn((B, H, T, D), dtype=torch.float16, device=device).requires_grad_(True)
    v = torch.randn((B, H, T, D), dtype=torch.float16, device=device).requires_grad_(True)
    do = torch.randn((B, H, T, D), dtype=torch.float16, device=device)

    ref = flash_attn_func(q=q.transpose(1, 2), k=k.transpose(1, 2), v=v.transpose(1, 2), softmax_scale=scale, causal=causal)
    ref.backward(do.transpose(1, 2))
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # transpoe back to B, H, T, D
    ref = ref.transpose(1, 2)

    tri = flash_attention_2(q=q, k=k, v=v, scale=scale, causal=causal)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)