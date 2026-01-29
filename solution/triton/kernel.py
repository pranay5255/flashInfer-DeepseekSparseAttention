"""
DSA TopK Indexer Reference Implementation for FlashInfer Competition.

Native Sparse Attention (DSA) TopK indexer with FP8 quantization for DeepSeek-V3.
Computes sparse attention scores using ReLU activation and learned weights,
then selects top-K KV cache indices.

Kernel: dsa_topk_indexer_fp8_h64_d128_topk256_ps64
"""

import torch
import triton
import triton.language as tl


def dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize FP8 KV cache from deep_gemm format."""
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4  # 128

    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)
    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    return fp8_float * scale


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    """
    DSA TopK Indexer kernel entry point.

    Args:
        q_index_fp8: Query index tensor [batch_size, num_index_heads=64, index_head_dim=128], float8_e4m3fn
        k_index_cache_fp8: KV cache [num_pages, page_size=64, kv_cache_num_heads=1, head_dim_with_scale=132], int8
        weights: Learned weights [batch_size, num_index_heads=64], float32
        seq_lens: Sequence lengths [batch_size], int32
        block_table: Page table [batch_size, max_num_pages], int32

    Returns:
        topk_indices: Selected top-K indices [batch_size, topk=256], int32
    """
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 256

    q = q_index_fp8.to(torch.float32)
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)

    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=q.device)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)
        K_paged = K_all[page_indices]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]

        q_b = q[b]
        scores = q_b @ K.T
        scores_relu = torch.relu(scores)
        w = weights[b]
        weighted_scores = scores_relu * w[:, None]
        final_scores = weighted_scores.sum(dim=0)

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return (topk_indices,)
