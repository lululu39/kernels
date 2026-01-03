import torch

from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward

from flash_attn import flash_attn_varlen_func

from functools import lru_cache
from einops import rearrange

def calculate_chunks(
    cu_seqlen: torch.LongTensor,
    chunk_size: int,
):
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    batch_num_chunk = (batch_sizes + (chunk_size - 1)) // chunk_size # cdiv, how many chunks in each batch [B]
    cu_num_chunk = torch.zeros(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0) # [B + 1]、
    num_chunk = cu_num_chunk[-1]
    chunk_sizes = torch.full(
        (num_chunk + 1,), chunk_size, dtype=torch.int32, device=cu_seqlen.device
    ) # chun size of each chunl
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * chunk_size # [B] every last chunk size of each batch
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    cu_chunk = chunk_sizes.cumsum(dim=0, dtype=torch.int32) # offset for each chunk
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, dtype=cu_seqlen.device
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    # filter chunks, remove the last chunk because we will not select this chunk in moba (causality)
    chunk_to_remove = cu_num_chunk[1:] - 1 # -1 because using indices
    chunk_to_remain = torch.ones(num_chunk, dtype=torch.bool, device=cu_seqlen.device)
    chunk_to_remain[chunk_to_remove] = False

    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]

    num_filtered_chunk = len(filtered_chunk_indices)

    return (cu_chunk, filtered_chunk_indices, num_filtered_chunk, chunk_to_batch)


def mixture_of_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    max_seqlen: int,
    chunk_size: int,
    topk: int,
):
    # q: [T, HQ, D]
    # k: [T, H, D]
    # v: [T, H, D]
    # NOTE: this is varlen only so no B dimension, and we use the same headdim

    T, HQ, D = q.shape
    H = k.shape[-2] 

    #  NOTE: why stack k and v, because the selection applies to them both
    kv = torch.stack((k, v), dim=1) # [T, 2, H, D]

    # calculate some useful statistics

    cu_chunk, filtered_chunk_indices, num_filtered_chunk, chunk_to_batch = calculate_chunks(
        cu_seqlen=cu_seqlens,
        chunk_size=chunk_size
    )

    topk = min(topk - 1, num_filtered_chunk) # current block is always chosen

    need_moba = topk > 0

    if not need_moba:
        raise ValueError("MoBA needs topk larger than 0!")

    # then we filter kv
    filtered_kv_indices = torch.arange(
        0, chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1) # [num_filtered_chunk, chunk_size]
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None] # we add the offset for each chunk
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1)) # kv that only contains elements needs moba attn
    # shape [num_filtered_chunk * chunk_size, H, K/V]

    # then we calculate the score for block selection

    pooled_k = (
        filtered_kv[:, 0]
        .view(num_filtered_chunk, chunk_size, H, D)
        .mean(dim=1)
        .float()
    ) # [N, H, D]
    q = q.type(torch.float32)  # NOTE: use float for high-precision calculation
    pooled_k = pooled_k.type(torch.float32)
    score = torch.einsum(
        "nhd,thd->tnh", pooled_k, q
    )  # [N, T, H]
    pooled_k = pooled_k.type_as(k)
    q = q.type_as(k)

    # post process score, masking unchosen batch and apply causal mask to current chunk
    score_seq_idx = torch.arange(0, T, device=q.device, dtype=torch.int32)[:, None].repeat(num_filtered_chunk, 1) # [N, T]
    chunk_end = cu_chunk[filtered_chunk_indices + 1] # end offset for each chunk [N]
    batch_end = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1] # batch end offset for each chunk [N]
    chunk_end_mask = score_seq_idx < chunk_end[:, None] # chunk position must precede query token
    batch_end_mask = score_seq_idx >= batch_end[:, None] # other batchs
    score_mask = chunk_end_mask | batch_end_mask # [N, T]
    score.masked_fill_(score_mask.unsqueeze(-1), -float("inf")) # [N, T, H]

    


