import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class BasicSR(nn.Module):
    """
    A simple super-resolution model using bicubic upsampling.
    """

    def __init__(self, upscale=4, in_chans=3, img_size=64, img_range=1.):
        super(BasicSR, self).__init__()
        self.upscale = upscale
        self.in_chans = in_chans
        self.img_size = img_size
        self.img_range = img_range
    
    def forward(self, x):
        b, c, h, w = x.size()
        # 使用双三次插值进行上采样
        x_upscaled = F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        # 确保输出值范围在0-1之间
        x_upscaled = torch.clamp(x_upscaled, 0, 1)
        return x_upscaled 