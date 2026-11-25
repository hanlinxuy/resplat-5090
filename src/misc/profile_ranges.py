"""Emit matching labels for PyTorch and Nsight in the active capture window."""

from contextlib import nullcontext
from contextvars import ContextVar

import torch


profile_backend = ContextVar("resplat_profile_backend", default=None)


def profile_range(name):
    backend = profile_backend.get()
    if backend == "nsys":
        return torch.cuda.nvtx.range(f"resplat/{name}")
    if backend == "torch":
        return torch.profiler.record_function(f"resplat/{name}")
    return nullcontext()
