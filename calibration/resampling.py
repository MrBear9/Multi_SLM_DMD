"""Differentiable, centred coordinate transfer between two complex-field grids."""
import torch
from torch import nn
from torch.nn import functional as F


class ComplexGridResampler(nn.Module):
    """Interpolate real/imaginary components at destination sample coordinates.

    This is an explicit approximation to continuous-field resampling, not a
    scaled-ASM kernel or an intensity-image resize. Refining both source and
    destination grids is required to assess convergence. Coarser destination
    sampling first removes frequencies above its Nyquist limit.
    """

    def __init__(self, source_size, source_pitch, destination_size, destination_pitch):
        super().__init__()
        self.source_size = source_size
        self.identity = source_size == destination_size and source_pitch == destination_pitch
        axis = (torch.arange(destination_size, dtype=torch.float32) - (destination_size - 1) / 2) * destination_pitch
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        grid = torch.stack((xx, yy), -1) / (source_size * source_pitch / 2)
        self.register_buffer('grid', grid[None])
        lowpass = None
        if destination_pitch > source_pitch:
            frequency = torch.fft.fftfreq(source_size, d=source_pitch)
            valid = frequency.abs() <= 1 / (2 * destination_pitch)
            lowpass = (valid[:, None] & valid[None, :])[None, None]
        self.register_buffer('lowpass', lowpass)

    def forward(self, field):
        if not field.is_complex() or tuple(field.shape[-2:]) != (self.source_size, self.source_size):
            raise ValueError('层间重采样需要与源网格匹配的复光场。')
        if self.identity:
            return field
        if self.lowpass is not None:
            field = torch.fft.ifft2(torch.fft.fft2(field) * self.lowpass)
        grid = self.grid.to(dtype=field.real.dtype).expand(field.shape[0], -1, -1, -1)
        real = F.grid_sample(field.real, grid, mode='bilinear', align_corners=False)
        imag = F.grid_sample(field.imag, grid, mode='bilinear', align_corners=False)
        return torch.complex(real, imag)
