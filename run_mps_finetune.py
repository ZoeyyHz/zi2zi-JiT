#!/usr/bin/env python
"""Launcher that patches CUDA-only calls so lora_single_gpu_finetune_jit.py runs on Apple MPS."""
import os
import runpy
import sys
from contextlib import nullcontext

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

target = torch.device("mps")
torch.Tensor.cuda = lambda self, *a, **k: self.to(target)
torch.nn.Module.cuda = lambda self, *a, **k: self.to(target)
torch.cuda.empty_cache = lambda: None
torch.cuda.synchronize = lambda device=None: None
torch.cuda.amp.autocast = lambda *a, **k: nullcontext()
torch.amp.autocast = lambda *a, **k: nullcontext()
if hasattr(torch, "compile"):
    torch.compile = lambda fn=None, *a, **k: fn if fn is not None else (lambda inner: inner)

sys.argv = ["lora_single_gpu_finetune_jit.py"] + sys.argv[1:]
runpy.run_path("lora_single_gpu_finetune_jit.py", run_name="__main__")
