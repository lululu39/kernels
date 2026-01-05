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

class MoBAAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        chunk_size,
        moba_q_th_indices,
    ):
        
        ctx.max_seqlen = max_seqlen
        ctx.chunk_size = chunk_size
        ctx.scale = scale = q.shape[-1] ** (-0.5)

        # NOTE: lse shape

        self_attn_out, self_attn_lse, _, _ = _flash_attn_varlen_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=scale,
            causal=True,
            dropout_p=0.0,
        )

        moba_attn_out, moba_attn_lse, _, _ = _flash_attn_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=chunk_size,
            softmax_scale=scale,
            causal=False, # NOTE: because the blocks are complete 
            dropout_p=0.0
        )

        # change lse to T H  shape
        self_attn_lse = self_attn_lse.t().contiguous() # [T, H]
        moba_attn_lse = moba_attn_lse.t().contiguous # [S, 1]

        out = torch.zeros_like(q, device=q.device, dtype=torch.float32)

        out_2d = out.view(-1, q.shape[2]) # [T * H, D]

        # calc mixed_lse
        # minus max lse to avoid exp explosion
        max_lse_1d = self_attn_lse.view(-1) # [T * H]
        max_lse_1d = max_lse_1d.index_reduce(
            0, moba_q_th_indices, moba_attn_lse.view(-1), "amax"
        )
        self_attn_lse = self_attn_lse - max_lse_1d.view_as(self_attn_lse) # for exp numerical stability
        moba_attn_lse = (
            moba_attn_lse.view(-1)
            .sub(max_lse_1d.index_select(0, moba_q_th_indices))
            .reshape_as(moba_attn_lse)
        ) # minus max value

        mixed_attn_se = self_attn_lse.exp()
        moba_attn_se = moba_attn_lse.exp()

        mixed_attn_se.view(-1).index_add_(
            0, moba_q_th_indices, moba_attn_se.view(-1)
        )
        mixed_attn_lse = mixed_attn_se.log() # [T, H] 始终按照原本q的布局来思考就好了

        # add self attn
        factor = (self_attn_lse - mixed_attn_lse).exp()  # [T, H] we use new denominator
        self_attn_out = self_attn_out * factor.unsqueeze(-1) 
        out_2d += self_attn_out.reshape_as(out_2d) # [T * H, D]

        # add moba

        mixed_attn_lse_moba = (
            mixed_attn_lse.view(-1)
            .index_select(0, moba_q_th_indices)
            .view_as(moba_attn_lse)
        )

        factor = (moba_attn_lse - mixed_attn_lse_moba) # [S, 1]

        moba_attn_out = moba_attn_out * factor.unsqueeze(-1) # [S, 1, D]

        moba_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[2]) # [S, D]

        out_2d.index_add_(0, moba_q_th_indices, moba_attn_out)

        out = out.to(q.dtype)

        # add back max lse, previously minize max is for numerical stability
        mixed_attn_lse += max_lse_1d.view_as(mixed_attn_lse)

        ctx.save_for_backward(
            out,
            mixed_attn_lse,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_th_indices
        )

        return out, mixed_attn_lse














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

    assert HQ == H, "head of q must equal to k, GQA is not allowed in this version of MoBA"

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
    
    self_attn_cu_seqlen = cu_chunk

    # then we filter kv
    filtered_kv_indices = torch.arange(
        0, chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1) # [num_filtered_chunk, chunk_size]
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None] # we add the offset for each chunk
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1)) # kv that only contains elements needs moba attn
    # shape [num_filtered_chunk * chunk_size, 2, H, D]

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
        "nhd,thd->nht", pooled_k, q
    )  # [N, H, T] NOTE: this is more convenient
    pooled_k = pooled_k.type_as(k)
    q = q.type_as(k)

    # post process score, masking unchosen batch and apply causal mask to current chunk
    score_seq_idx = torch.arange(0, T, device=q.device, dtype=torch.int32)[:, None].repeat(num_filtered_chunk, 1) # [N, T]
    chunk_end = cu_chunk[filtered_chunk_indices + 1] # end offset for each chunk [N]
    batch_end = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1] # batch end offset for each chunk [N]
    chunk_end_mask = score_seq_idx < chunk_end[:, None] # chunk end position must precede query token, because we are using complete blocks for moba
    batch_end_mask = score_seq_idx >= batch_end[:, None] # other batchs
    score_inf_mask = chunk_end_mask | batch_end_mask # [N, T]
    score.masked_fill_(score_inf_mask.unsqueeze(1), -float("inf")) # [N, H, T]

    # find topk blocks and set a mask representing them

    _, score_top_k_idx = torch.topk(score, k=topk, dim=0, largest=True, sorted=False) # [topk, H, T]

    score_mask = torch.logical_not(score.isinf())
    score_idx_mask = torch.zeros(score.shap, dtype=torch.bool, device=q.device)
    score_idx_mask = score_idx_mask.scatter_(dim=0, index=score_top_k_idx, value=True)
    score_mask = score_mask & score_idx_mask # [N, H, T]

    # NOTE: the result will be like [ C0H0 ][ C0H1 ][ C0H2 ][ ... ][ CnHm ]
    # NOTE: 这里q indices用H * T，方便下面的moba_q计算
    moba_q_indices = score_mask.reshape(score_mask.shape[0], -1).nonzero(as_tuple=True)[-1] # [chunk1 selected idx, chun2 selected idx, xxx]
    moba_seqlen_q = score_mask.sum(dim=-1).flatten() # [N * H] query seqlen of each kv chunk

    moba_q = rearrange(q, "t h d -> ( h t ) d").index_select(
        0, moba_q_indices
    )  # [ S, D ] NOTE: S is chunk1 selected + chunk2 + ...
    moba_q = moba_q.unsqueeze(1) # [S, 1, D]
    # moba_q_th_indices represents the position in the origin q tensor (flattend T* H ) of each q token inside moba_q
    moba_q_th_indices = moba_q_indices % T * HQ + moba_q_indices // T

    q_zero_mask = moba_seqlen_q == 0 # [N * H]
    valid_expert_mask = ~q_zero_mask # here expert means the kv blocks
    zero_expert_count = q_zero_mask.sum()

    if zero_expert_count > 0:
        # only keep kv blocks that had query chosen
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask] # [C]
    
    moba_cu_seqlen_q = torch.cat((
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),),dim=0).to(torch.int32)

    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d")
    moba_kv = moba_kv.split(chunk_size, dim=1)
    moba_kv = torch.cat(moba_kv, dim=0) # [N * H, chunk_size, 2, D]

    if zero_expert_count > 0:
        assert valid_expert_mask.sum() == moba_kv.shape[0] - zero_expert_count
        moba_kv = moba_kv[valid_expert_mask]  # cut off zero Q expert from kv , or the grad may be nan
        # shape [C * H, chunk_size, 2, D]
    
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2) # [C * H * chunk_size, 2, 1, D]

    moba_cu_seqlen_kv = (
        torch.arange(
            0,
            # num_filtered_chunk * H + 1 - zero_expert_count,
            int(valid_expert_mask.sum().item()) + 1,
            dtype=torch.int32,
            device=q.device,
        )
        * chunk_size
    ) # in the order of chunk-head

    # NOTE: all is the ordering of chunk-head pair
    assert (
        moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    ), f"moba_cu_seqlen_kv.shape != moba_cu_seqlen_q.shape {moba_cu_seqlen_kv.shape} != {moba_cu_seqlen_q.shape}"


    return MoBAAttention.apply(
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        chunk_size,
        moba_q_th_indices,
    )




