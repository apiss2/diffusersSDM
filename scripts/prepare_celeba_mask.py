from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from sdm_diffusion.data import LABELS


def combine_masks(mask_root: str, out_dir: str) -> None:
    mask_root = Path(mask_root)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for mask_file in mask_root.glob("*.png"):
        stem = mask_file.stem  # e.g. 00000_hair
        img_id, part = stem.split("_")
        target = out_path / f"{img_id}.png"

        mask = Image.open(mask_file)
        mask_np = np.array(mask)
        if target.exists():
            canvas = np.array(Image.open(target))
        else:
            canvas = np.zeros_like(mask_np)
        canvas[mask_np > 127] = LABELS.get(part, 0)
        Image.fromarray(canvas).save(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", type=str, required=True, help="path to raw mask parts")
    parser.add_argument("--out", type=str, required=True, help="directory to write combined masks")
    args = parser.parse_args()
    combine_masks(args.mask_root, args.out)
