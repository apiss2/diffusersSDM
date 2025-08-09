from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABELS = {
    "skin": 1,
    "nose": 2,
    "eye_g": 3,
    "l_eye": 4,
    "r_eye": 5,
    "l_brow": 6,
    "r_brow": 7,
    "l_ear": 8,
    "r_ear": 9,
    "mouth": 10,
    "u_lip": 11,
    "l_lip": 12,
    "hair": 13,
    "hat": 14,
    "ear_r": 15,
    "neck_l": 16,
    "neck": 17,
    "cloth": 18,
}


class CelebAHQMaskDataset(Dataset):
    """Dataset yielding image and segmentation tensor pairs."""

    def __init__(self, image_root: str, mask_root: str, image_size: int = 256):
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.image_size = image_size

        self.images: List[Path] = sorted(self.image_root.glob("*.jpg"))
        if not self.images:
            self.images = sorted(self.image_root.glob("*.png"))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def _load_mask(self, img_id: str) -> torch.Tensor:
        mask_path = self.mask_root / f"{img_id}.png"
        mask = Image.open(mask_path)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np)
        one_hot = torch.nn.functional.one_hot(mask_tensor, num_classes=19)
        return one_hot.permute(2, 0, 1).float()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        img_id = img_path.stem
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        mask = self._load_mask(img_id)
        return img, mask
