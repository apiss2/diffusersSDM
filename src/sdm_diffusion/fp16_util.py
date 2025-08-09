import torch as th


def convert_module_to_f16(module: th.nn.Module):
    """Recursively convert module parameters to float16."""
    for p in module.parameters():
        p.data = p.data.half()
    for b in module.buffers():
        b.data = b.data.half()


def convert_module_to_f32(module: th.nn.Module):
    """Recursively convert module parameters to float32."""
    for p in module.parameters():
        p.data = p.data.float()
    for b in module.buffers():
        b.data = b.data.float()
