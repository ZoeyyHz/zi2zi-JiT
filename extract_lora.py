#!/usr/bin/env python
"""Extract LoRA weights from a fine-tuned checkpoint into a small standalone file.

The output file contains only the LoRA increments (lora_A/lora_B), the fine-tuned
font embedding, and metadata needed to rebuild the structure at load time.

Usage:
    python extract_lora.py \
        --checkpoint run/lora_ft_fzbwksjw/checkpoint-last.pth \
        --output models/lora-fzbwksjw.pth
"""
import argparse
import re

import torch

# spectral-norm power-iteration buffers (u0/sv0, u1/sv1, ...) are updated by forward
# passes during fine-tuning even though they are not trainable parameters
_SN_BUFFER_RE = re.compile(r'\.(?:u|sv)\d+$')


def main():
    parser = argparse.ArgumentParser(description="Extract LoRA weights from a fine-tuned checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint (full model)")
    parser.add_argument("--output", required=True, help="Output path for the LoRA-only file")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    train_args = ckpt.get("args") if isinstance(ckpt, dict) else None

    lora_sd = {k: v for k, v in state_dict.items() if "lora_" in k}
    if not lora_sd:
        raise SystemExit("No LoRA weights found in this checkpoint (not a LoRA fine-tune?)")
    # font embedding is unfrozen during fine-tuning (train_font_emb=True) and must travel with the LoRA
    font_emb = {k: v for k, v in state_dict.items() if "font_embedding" in k}
    sn_buffers = {k: v for k, v in state_dict.items() if _SN_BUFFER_RE.search(k)}

    meta = {
        "base_model": getattr(train_args, "model", "JiT-B/16"),
        "lora_r": getattr(train_args, "lora_r", None),
        "lora_alpha": getattr(train_args, "lora_alpha", None),
        "lora_dropout": getattr(train_args, "lora_dropout", 0.0),
        "lora_targets": getattr(train_args, "lora_targets", "qkv,proj,w12,w3"),
        "num_fonts": getattr(train_args, "num_fonts", None),
        "num_chars": getattr(train_args, "num_chars", None),
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "source_checkpoint": args.checkpoint,
        # generation defaults saved in the fine-tune args
        "cfg": getattr(train_args, "cfg", None),
        "sampling_method": getattr(train_args, "sampling_method", None),
        "num_sampling_steps": getattr(train_args, "num_sampling_steps", None),
        "noise_scale": getattr(train_args, "noise_scale", None),
    }

    out = {"lora": lora_sd, "font_embedding": font_emb, "buffers": sn_buffers, "meta": meta}
    torch.save(out, args.output)

    size_mb = sum(v.numel() * v.element_size() for v in lora_sd.values()) / 1e6
    size_mb += sum(v.numel() * v.element_size() for v in font_emb.values()) / 1e6
    size_mb += sum(v.numel() * v.element_size() for v in sn_buffers.values()) / 1e6
    print(f"Extracted {len(lora_sd)} LoRA tensors + {len(font_emb)} font embedding tensors "
          f"+ {len(sn_buffers)} spectral-norm buffers ({size_mb:.1f} MB) -> {args.output}")
    print(f"meta: {meta}")


if __name__ == "__main__":
    main()
