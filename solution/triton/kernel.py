"""
GEMM Triton Kernel for FlashInfer Competition.

Implements C = A @ B.T for gemm_n4096_k4096 definition.
Captured from Llama 3.1 8B attention output projection.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Compute C = A @ B.T

    A: [M, K], B: [N, K] -> C: [M, N]
    Note: B is stored as [N, K] but we compute A @ B.T
    """
    pid = tl.program_id(0)

    # Swizzle for better L2 cache locality
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block start indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first block of A and B
    # A: [M, K] with strides (stride_am, stride_ak)
    # B: [N, K] with strides (stride_bn, stride_bk) - we read rows of B
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A block [BLOCK_M, BLOCK_K]
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B block [BLOCK_N, BLOCK_K]
        b_mask = (offs_n[:, None] < N) & ((k + offs_k[None, :]) < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Compute A @ B.T -> need to transpose b
        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_N, BLOCK_K]
        # We want: a @ b.T = [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, tl.trans(b))

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Convert to output dtype and store
    c = acc.to(tl.float16)

    # Store result
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@torch.no_grad()
def run(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    GEMM kernel entry point: C = A @ B.T

    Args:
        A: Input matrix [M, K], float16
        B: Weight matrix [N, K], float16

    Returns:
        C: Output matrix [M, N], float16
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "Inputs must be float16"

    M, K = A.shape
    N, K_b = B.shape
    assert K == K_b, f"K dimension mismatch: A has {K}, B has {K_b}"

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Grid: one program per output tile
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )

    return C
