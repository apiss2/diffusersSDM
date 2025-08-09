from typing import Optional, Union, List

import torch
from PIL import Image
from torchvision import transforms
from diffusers import DiffusionPipeline

from .unet import UNetModel
from .respace import SpacedDiffusion


class SemanticDiffusionPipeline(DiffusionPipeline):
    """Pipeline for Semantic Image Synthesis via Diffusion Models."""

    def __init__(self, unet: UNetModel, diffusion: SpacedDiffusion):
        super().__init__()
        self.register_modules(unet=unet, diffusion=diffusion)

    @torch.no_grad()
    def __call__(
        self,
        cond: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
    ) -> Union[List[Image.Image], torch.Tensor]:
        """Generate images conditioned on semantic map.

        Args:
            cond: segmentation map tensor of shape (B, C, H, W).
            num_inference_steps: overrides the diffusion steps if provided.
            guidance_scale: classifier-free guidance scale.
            generator: optional random generator.
            output_type: "pil" or "tensor".
        """
        batch_size, c, h, w = cond.shape
        device = self.device
        cond = cond.to(device)
        shape = (batch_size, 3, h, w)
        if num_inference_steps is not None:
            self.diffusion.num_timesteps = num_inference_steps
        noise = torch.randn(shape, generator=generator, device=device)
        samples = self.diffusion.p_sample_loop(
            self.unet,
            shape,
            noise=noise,
            model_kwargs={"y": cond, "s": guidance_scale},
            device=device,
            progress=False,
        )
        if output_type == "tensor":
            return samples
        samples = (samples / 2 + 0.5).clamp(0, 1)
        images = [transforms.ToPILImage()(img) for img in samples]
        return images
