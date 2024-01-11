import torch as th
import torch.nn.functional as F

def simplerggb2rgb(raw):
    burst_rgb = raw[:, 0, [0, 1, 3]]
    burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
    burst_rgb = F.interpolate(burst_rgb, scale_factor=2, mode='bilinear')
    return burst_rgb

