import argparse
from pathlib import Path

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.data import SDMDataset
from src.model import SDMUNet


def main(args):
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "sample").mkdir(exist_ok=True)
    # precision 解析
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    prec = args.precision.lower()
    assert prec in ["fp32", "fp16", "bf16"]
    use_amp = prec != "fp32"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(prec, torch.float32)

    # Dataset / Loader
    print("データセットの準備中...")
    dataset = SDMDataset(
        args.image_dir,
        args.label_dir,
        size=args.size,
        num_classes=args.num_classes,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    in_channel = 1 if args.grayscale else 3

    # Model / Optim / Scheduler(noise)
    print("モデル構築中...")
    unet = SDMUNet(
        in_channels=in_channel,
        base_channels=args.base_ch,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        num_classes=args.num_classes,
        dropout=0.1,
        use_scale_shift_norm=True,
    ).to(device)

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    noise_scheduler = DDPMScheduler(1000, beta_schedule="squaredcos_cap_v2")
    use_autocast = device.type == "cuda" and prec in ["fp16", "bf16"]
    scaler = GradScaler(device.type, enabled=(device.type == "cuda" and prec == "fp16"))

    print("学習開始")
    unet.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, batch in pbar:
            img = batch["image"].to(device)  # [-1,1]
            seg = batch["semantic_label"].to(device)  # [B,C,H,W]
            noise = torch.randn_like(img)
            t = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (img.shape[0],),
                device=device,
            ).long()
            xt = noise_scheduler.add_noise(img, noise, t)

            opt.zero_grad(set_to_none=True)
            with autocast(device.type, dtype=amp_dtype, enabled=use_autocast):
                pred_eps = unet(xt, t, seg)
                loss = torch.mean((pred_eps - noise) ** 2)

            if prec == "fp16" and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                opt.step()

            pbar.set_description_str(f"Epoch[{epoch:06d}] loss {loss.item():.4f}")

        # サンプル出力
        unet.eval()
        with (
            torch.no_grad(),
            autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast),
        ):
            b = min(4, img.size(0))
            seg_vis = seg[:b]
            x = torch.randn(b, in_channel, args.size, args.size, device=device)
            timesteps = torch.arange(
                noise_scheduler.config.num_train_timesteps - 1, -1, -1, device=device
            )
            for tt in timesteps:
                pred = unet(x, tt.expand(b), seg_vis)
                x = noise_scheduler.step(pred, tt, x).prev_sample
            imgs = (x.clamp(-1, 1) + 1) / 2
            save_image(imgs, out_dir / f"sample/sample_epoch{epoch:06d}.png", nrow=2)

        # 指定エポック毎にモデル保存
        if epoch % args.log_every == 0:
            torch.save(unet.state_dict(), out_dir / f"unet_sdm_epoch{epoch:06d}.pt")
        unet.train()

    print("done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    args = p.parse_args()
    main(args)
