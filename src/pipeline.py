import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class SemanticDDPMPipeline:
    def __init__(self, unet, scheduler: DDPMScheduler):
        self.unet = unet
        self.scheduler = scheduler

    @torch.no_grad()
    def __call__(
        self,
        seg,
        num_inference_steps=None,
        generator=None,
        eta=0.0,
        output_type="pt",
        progress_bar=True,
    ):
        device = next(self.unet.parameters()).device
        b, _, h, w = seg.shape
        seg = seg.to(device)

        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps, device=device)

        x = torch.randn(b, 3, h, w, device=device, generator=generator)
        for t in self.scheduler.timesteps:
            eps = self.unet(x, t.expand(b), seg)
            x = self.scheduler.step(eps, t, x).prev_sample

        if output_type == "pt":
            return x
        return (x.clamp(-1, 1) + 1) / 2
