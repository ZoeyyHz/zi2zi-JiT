#!/usr/bin/env python
"""Stylize a sentence: input text -> glyphs in the fine-tuned font style.

Usage:
    python stylize.py "你好世界" \
        --checkpoint run/lora_ft_fzbwksjw/checkpoint-last.pth \
        --source-font data/source_font/HiraginoSansGB.ttf \
        --target-font data/target_fonts/FZBWKSJW.TTF \
        --output_dir run/stylized/

Outputs per-character comparison images plus a full-sentence strip (generated glyphs).
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data_processing.charsets import get_charset_codepoints
from data_processing.font_utils import GlyphRenderer

DEFAULT_REF_CHARS = "永国我人大小中"  # style reference glyphs taken from the target font


def build_charset_index():
    codepoints = sorted(get_charset_codepoints("gb2312"))
    return {cp: i for i, cp in enumerate(codepoints)}


def main():
    parser = argparse.ArgumentParser(description="Stylize a sentence with the fine-tuned zi2zi-JiT model")
    parser.add_argument("text", help="Sentence to stylize")
    parser.add_argument("--checkpoint", default=None, help="Full fine-tuned checkpoint (alternative to --lora)")
    parser.add_argument("--lora", default=None, help="LoRA-only file (from extract_lora.py); pairs with the shared base checkpoint")
    parser.add_argument("--base_checkpoint", default="models/zi2zi-JiT-B-16.pth", help="Vanilla base checkpoint for --lora")
    parser.add_argument("--source-font", default="data/source_font/HiraginoSansGB.ttf")
    parser.add_argument("--target-font", default="data/target_fonts/FZBWKSJW.TTF")
    parser.add_argument("--output_dir", default="run/stylized")
    parser.add_argument("--sampling_method", default="ab2")
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--num_sampling_steps", type=int, default=None)
    parser.add_argument("--font_label", type=int, default=0, help="Font embedding index (0 = first fine-tuned font)")
    args = parser.parse_args()

    index_map = build_charset_index()
    source_renderer = GlyphRenderer(args.source_font, 256)
    ref_renderer = GlyphRenderer(args.target_font, 128)

    ref_imgs = [ref_renderer.render(ord(c)) for c in DEFAULT_REF_CHARS]
    ref_imgs = [img for img in ref_imgs if img is not None]
    if not ref_imgs:
        sys.exit("Target font could not render any style reference glyphs.")

    seen = {}
    content, style, char_labels, kept_chars = [], [], [], []
    for ch in args.text:
        cp = ord(ch)
        if not ch.strip():
            continue
        if cp in seen:
            kept_chars.append((ch, seen[cp]))
            continue
        if cp not in index_map:
            print(f"Skipping {ch!r} (U+{cp:04X}): not in gb2312 charset (no char embedding)")
            continue
        c_img = source_renderer.render(cp)
        if c_img is None:
            print(f"Skipping {ch!r}: source font cannot render it")
            continue
        ref = ref_imgs[len(seen) % len(ref_imgs)]
        content.append(np.array(c_img).transpose(2, 0, 1))
        style.append(np.array(ref).transpose(2, 0, 1))
        char_labels.append(index_map[cp])
        seen[cp] = len(char_labels) - 1
        kept_chars.append((ch, len(char_labels) - 1))

    if not content:
        sys.exit("No renderable CJK characters found in the input.")

    n = len(content)
    font_labels = np.full(n, args.font_label, dtype=np.int64)
    unicode_labels = np.array([ord(ch) for ch, _ in kept_chars], dtype=np.int64)
    # generate_chars.py requires dummy target images with the same shape
    target_images = np.zeros((n, 3, 256, 256), dtype=np.uint8)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "input.npz"
    np.savez_compressed(
        npz_path,
        font_labels=font_labels,
        char_labels=np.array(char_labels, dtype=np.int64),
        unicode_labels=unicode_labels,
        content_images=np.stack(content),
        target_images=target_images,
        style_images=np.stack(style),
        num_original_samples=np.int64(n),
    )

    cmd = [
        sys.executable, "generate_chars.py",
        "--test_npz", str(npz_path),
        "--output_dir", str(out_dir),
        "--sampling_method", args.sampling_method,
    ]
    if args.lora:
        cmd += ["--lora", args.lora, "--base_checkpoint", args.base_checkpoint]
    else:
        cmd += ["--checkpoint", args.checkpoint or "run/lora_ft_fzbwksjw/checkpoint-last.pth"]
    if args.cfg is not None:
        cmd += ["--cfg", str(args.cfg)]
    if args.num_sampling_steps is not None:
        cmd += ["--num_sampling_steps", str(args.num_sampling_steps)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # locate generated images and copy them out as per-character files named by the character
    gen_dirs = sorted([d for d in out_dir.iterdir() if d.is_dir() and (d / "generated").is_dir()])
    if not gen_dirs:
        sys.exit("Generation finished but no output folder found.")
    gen_dir = gen_dirs[-1] / "generated"
    chars_dir = out_dir / "chars"
    chars_dir.mkdir(exist_ok=True)
    saved = 0
    for i, (ch, _) in enumerate(kept_chars):
        matches = sorted(gen_dir.glob(f"*_U+{ord(ch):04X}.png"))
        if not matches:
            print(f"Warning: no generated image for {ch!r}")
            continue
        dst = chars_dir / f"{i:03d}_{ch}.png"
        Image.open(matches[0]).convert("RGB").save(dst)
        saved += 1
    print(f"\nPer-character images ({saved}) saved to: {chars_dir}")
    print(f"Raw generation folder: {gen_dir}")


if __name__ == "__main__":
    main()
