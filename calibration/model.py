"""Native SLM pixels, ideal 1:1 inverted 4f relay and two ASM propagations.

Torch is imported only by training, not by the ordinary GUI startup path.
"""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.SLM.asm_propagation import ASMPropagation
from .resampling import ComplexGridResampler
from .patterns import illumination_mask


def phase_lut(config, key):
    path = getattr(config, key + '_lut')
    if path:
        values = np.loadtxt(path, dtype=np.float32).reshape(-1)
        if values.shape != (256,) or not np.isfinite(values).all():
            raise ValueError(f'{key} LUT 应包含 256 个有限相位值，单位 rad。')
    else:
        values = np.linspace(0, 2 * np.pi, 256, dtype=np.float32)
        if config.gray_inverted:
            values = values[::-1].copy()
    # Uniform grey 0 is the phase reference everywhere outside the active array.
    return values - values[0]


def encode_phase(phase, lut):
    # Circular nearest phase lookup, bounded memory, supports nonmonotonic LUTs.
    wrapped = np.mod(phase, 2 * np.pi)
    bins = np.linspace(0, 2 * np.pi, 16384, endpoint=False)
    distances = np.abs(np.angle(np.exp(1j * (bins[:, None] - lut[None, :]))))
    table = np.argmin(distances, axis=1).astype(np.uint8)
    return table[np.rint(wrapped / (2 * np.pi) * len(table)).astype(int) % len(table)]


class CalibrationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.validate()
        illumination = torch.from_numpy(illumination_mask(config).astype(np.float32))
        self.register_buffer('illumination', illumination[None, None].to(torch.complex64))
        for key in ('slm1', 'slm2'):
            count = config.grid_size_for(key)
            axis = (torch.arange(count, dtype=torch.float32) - (count - 1) / 2) * config.sampling_um_for(key)
            yy, xx = torch.meshgrid(axis, axis, indexing='ij')
            h, w = config.slm_shape(key)
            pitch = getattr(config, key + '_pitch_um')
            grid = torch.stack((xx / (w * pitch / 2), yy / (h * pitch / 2)), -1)
            self.register_buffer(key + '_grid', grid[None])
        self.propagators = nn.ModuleList([
            ASMPropagation(SimpleNamespace(), distance=distance / 1000,
                wavelength=config.wavelength_nm * 1e-9, pixel_size=config.sampling_um_for(key) * 1e-6,
                resolution=(config.grid_size_for(key),) * 2)
            for key, distance in (('slm1', config.slm1_to_slm2_mm), ('slm2', config.slm2_to_camera_mm))])
        self.interplane_resampler = ComplexGridResampler(config.slm1_grid_size, config.slm1_sampling_um,
                                                         config.slm2_grid_size, config.slm2_sampling_um)

    def native_phase(self, parameters, key):
        return F.interpolate(parameters, size=self.config.slm_shape(key), mode='bilinear',
                             align_corners=False)

    def modulate(self, field, native, key):
        # Interpolate phasors, not wrapped phases. Outside the active area the
        # phase is the same uniform reference as the full hardware canvas.
        phasor = torch.exp(1j * native)
        grid = getattr(self, key + '_grid')
        real = F.grid_sample(phasor.real - 1, grid, mode='nearest', align_corners=False) + 1
        imag = F.grid_sample(phasor.imag, grid, mode='nearest', align_corners=False)
        return field * torch.complex(real, imag)

    def forward(self, phases):
        field = self.illumination
        for index, key in enumerate(('slm1', 'slm2')):
            if index:
                field = self.interplane_resampler(field)
            if key in phases:
                field = self.modulate(field, phases[key], key)
            field = self.propagators[index](field)
        return field.abs().square()

