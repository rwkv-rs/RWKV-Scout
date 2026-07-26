"""Run the local RWKV7 vLLM server on WSL without CUDA UVA extensions.

The RWKV7 Model Runner V2 uses vLLM's UVA buffers. The local WSL PyTorch
build has pinned host memory but does not expose the CUDA view extension, so
we keep the same buffer API and copy staged values to regular GPU tensors.
"""

from __future__ import annotations

import runpy

import numpy as np
import torch

import vllm.utils.platform_utils as platform_utils
import vllm.v1.worker.gpu.buffer_utils as buffer_utils


class WslUvaBuffer:
    def __init__(self, size: int | tuple[int, ...], dtype: torch.dtype):
        try:
            self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        except Exception:
            self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()
        self.uva = torch.empty(size, dtype=dtype, device="cuda")


class WslUvaBufferPool:
    def __init__(
        self,
        size: int | tuple[int, ...],
        dtype: torch.dtype,
        max_concurrency: int | None = None,
    ):
        self.size = size
        self.dtype = dtype
        self.max_concurrency = max(2, max_concurrency or 2)
        self._uva_bufs = [WslUvaBuffer(size, dtype) for _ in range(self.max_concurrency)]
        self._curr = 0

    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        values = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        n = len(values)
        buf.np[:n] = values
        buf.uva[:n].copy_(buf.cpu[:n], non_blocking=buf.cpu.is_pinned())
        return buf.uva[:n]

    def copy_to_gpu(
        self,
        x: torch.Tensor | np.ndarray,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gpu = self.copy_to_uva(x)
        return gpu.clone() if out is None else out.copy_(gpu, non_blocking=True)


platform_utils.is_uva_available = lambda: True
buffer_utils.is_uva_available = lambda: True
buffer_utils.UvaBuffer = WslUvaBuffer
buffer_utils.UvaBufferPool = WslUvaBufferPool

def main() -> None:
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
