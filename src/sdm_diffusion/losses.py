import numpy as np
import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """KL divergence between two normals parameterized by mean and log-variance."""
    return 0.5 * (
        -1.0
        + logvar2 - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """Log likelihood for x under a discretized Gaussian.
    This is a simplified version suitable for diffusion training."""
    centered = x - means
    inv_stdv = th.exp(-log_scales)
    normal_ll = -0.5 * ((centered * inv_stdv) ** 2) - log_scales - 0.5 * np.log(2 * np.pi)
    return normal_ll
