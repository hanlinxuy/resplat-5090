"""Indexed short-neighborhood attention for the BF16 ReSplat point blocks."""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(QKV, Indices, Output, Probs, N: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    neighbor = tl.arange(0, 16)
    channel = tl.arange(0, 64)
    idx = tl.load(Indices + row[:, None] * 16 + neighbor[None, :], row[:, None] < N, other=-1)
    base = idx[:, :, None] * 192 + channel[None, None, :]
    q = tl.load(QKV + row[:, None] * 192 + channel[None, :], row[:, None] < N, other=0)
    k = tl.load(QKV + base + 64, idx[:, :, None] >= 0, other=0)
    score = tl.sum(q[:, None, :].to(tl.float32) * k.to(tl.float32), 2)
    # The eager BF16 matmul rounds before scaling and again before softmax.
    score = (score.to(tl.bfloat16).to(tl.float32) * 0.125).to(tl.bfloat16).to(tl.float32)
    exp = tl.exp(score - tl.max(score, 1)[:, None])
    prob = (exp / tl.sum(exp, 1)[:, None]).to(tl.bfloat16)
    v = tl.load(QKV + base + 128, idx[:, :, None] >= 0, other=0)
    out = tl.sum(prob[:, :, None].to(tl.float32) * v.to(tl.float32), 1)
    tl.store(Output + row[:, None] * 64 + channel[None, :], out, row[:, None] < N)
    tl.store(Probs + row[:, None] * 16 + neighbor[None, :], prob, row[:, None] < N)


@triton.jit
def _backward(QKV, Indices, Probs, DY, DQKV, N: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    neighbor = tl.arange(0, 16)
    channel = tl.arange(0, 64)
    prob_offset = row[:, None] * 16 + neighbor[None, :]
    idx = tl.load(Indices + prob_offset, row[:, None] < N, other=-1)
    valid = idx[:, :, None] >= 0
    base = idx[:, :, None] * 192 + channel[None, None, :]
    q_offset = row[:, None] * 192 + channel[None, :]
    q = tl.load(QKV + q_offset, row[:, None] < N, other=0).to(tl.float32)
    dy = tl.load(DY + row[:, None] * 64 + channel[None, :], row[:, None] < N, other=0).to(tl.float32)
    prob = tl.load(Probs + prob_offset, row[:, None] < N, other=0).to(tl.float32)
    v = tl.load(QKV + base + 128, valid, other=0).to(tl.float32)
    dp = tl.sum(dy[:, None, :] * v, 2).to(tl.bfloat16).to(tl.float32)
    ds = ((dp - tl.sum(dp * prob, 1)[:, None]) * prob).to(tl.bfloat16).to(tl.float32)
    ds = (ds * 0.125).to(tl.bfloat16).to(tl.float32)
    k = tl.load(QKV + base + 64, valid, other=0).to(tl.float32)
    dq = tl.sum(ds[:, :, None] * k, 1).to(tl.bfloat16).to(tl.float32)
    tl.store(DQKV + q_offset, dq, row[:, None] < N)
    dk = (ds[:, :, None] * q[:, None, :]).to(tl.bfloat16).to(tl.float32)
    dv = (prob[:, :, None] * dy[:, None, :]).to(tl.bfloat16).to(tl.float32)
    tl.atomic_add(DQKV + base + 64, dk, valid, sem="relaxed")
    tl.atomic_add(DQKV + base + 128, dv, valid, sem="relaxed")


class FusedLocalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, indices, rows_per_cta):
        rows = qkv.shape[0]
        output = qkv.new_empty((rows, 64))
        probs = qkv.new_empty((rows, 16))
        _forward[(triton.cdiv(rows, rows_per_cta),)](
            qkv, indices, output, probs, rows, rows_per_cta, num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(qkv, indices, probs)
        ctx.rows_per_cta = rows_per_cta
        return output

    @staticmethod
    def backward(ctx, grad):
        qkv, indices, probs = ctx.saved_tensors
        rows = qkv.shape[0]
        dqkv = torch.zeros(qkv.shape, device=qkv.device, dtype=torch.float32)
        _backward[(triton.cdiv(rows, ctx.rows_per_cta),)](
            qkv, indices, probs, grad.contiguous(), dqkv, rows, ctx.rows_per_cta,
            num_warps=4, enable_fp_fusion=False,
        )
        return dqkv.to(qkv.dtype), None, None
