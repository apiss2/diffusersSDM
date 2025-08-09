from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import load_config
from .data import CelebAHQMaskDataset
from .gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
    get_named_beta_schedule,
)
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel


def create_diffusion(steps: int, schedule: str) -> SpacedDiffusion:
    betas = get_named_beta_schedule(schedule, steps)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, [steps]),
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CelebAHQMaskDataset(
        image_root=str(Path(cfg.data_root) / "images"),
        mask_root=str(Path(cfg.mask_root) / "masks"),
        image_size=cfg.image_size,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    model = UNetModel(
        image_size=cfg.image_size,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        num_classes=cfg.num_classes,
    ).to(device)

    diffusion = create_diffusion(cfg.diffusion_steps, cfg.schedule)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for imgs, seg in pbar:
            imgs, seg = imgs.to(device), seg.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (imgs.size(0),), device=device)
            losses = diffusion.training_losses(model, imgs, t, model_kwargs={"y": seg})
            loss = losses["loss"].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss))

        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"model_epoch{epoch+1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/celeba-hq.yaml")
    args = parser.parse_args()
    main(args.config)
