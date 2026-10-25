"""Standalone differentiable optics. Imported ONLY by the external Python worker.

No dependency on the parent project or neural-holography source. All lengths
are physical, and both SLMs share one computational coordinate grid. A second
SLM multiplies the propagated complex field, never a new illumination field.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ASM(nn.Module):
    def __init__(self, shape, pitch_um, wavelength_nm, distance_mm):
        super().__init__()
        self.shape = tuple(shape)
        height, width = self.shape
        padded_height, padded_width = 2 * height, 2 * width
        pitch, wavelength, z = pitch_um * 1e-6, wavelength_nm * 1e-9, distance_mm * 1e-3
        fy = torch.fft.fftfreq(padded_height, d=pitch, dtype=torch.float64)
        fx = torch.fft.fftfreq(padded_width, d=pitch, dtype=torch.float64)
        fy, fx = torch.meshgrid(fy, fx, indexing='ij')
        k2 = wavelength ** -2 - fx.square() - fy.square()
        limit_x = 1 / (wavelength * math.sqrt(1 + (2 * z / (padded_width * pitch)) ** 2))
        limit_y = 1 / (wavelength * math.sqrt(1 + (2 * z / (padded_height * pitch)) ** 2))
        mask = (k2 >= 0) & (fx.abs() <= limit_x) & (fy.abs() <= limit_y)
        kernel = torch.polar(mask.double(), 2 * math.pi * z * k2.clamp_min(0).sqrt())
        self.register_buffer('kernel', kernel.to(torch.complex64))

    def forward(self, field, adjoint=False):
        height, width = self.shape
        if tuple(field.shape[-2:]) != self.shape:
            raise ValueError('输入光场尺寸与 ASM 网格不一致。')
        top, left = height // 2, width // 2
        padded = F.pad(field, (left, width - left, top, height - top))
        kernel = self.kernel.conj() if adjoint else self.kernel
        result = torch.fft.ifft2(torch.fft.fft2(padded) * kernel)
        return result[..., top:top+height, left:left+width]


class OpticalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        height, width = config.grid_shape
        pitch = config.sampling_um
        axis_y = (torch.arange(height) - (height - 1) / 2) * pitch
        axis_x = (torch.arange(width) - (width - 1) / 2) * pitch
        yy, xx = torch.meshgrid(axis_y, axis_x, indexing='ij')
        self.register_buffer('xy', torch.stack((xx, yy), -1)[None])
        # Smooth positive illumination, anchored to mean amplitude one.
        self.source_log = nn.Parameter(torch.zeros(1, 1, 8, 8))
        self.source_phase = nn.Parameter(torch.zeros(1, 1, 8, 8))
        self.lut_steps = nn.ParameterDict({k: nn.Parameter(torch.zeros(16)) for k in config.active_slms})
        self.phase_range = nn.ParameterDict({k: nn.Parameter(torch.tensor(0.)) for k in config.active_slms})
        self.aberrations = nn.ParameterDict({k: nn.Parameter(torch.zeros(5)) for k in config.active_slms})
        # A small, smooth effective target-plane field, not a pixel-wise target
        # image memorizer. Coherent leakage interferes before intensity detection.
        self.background_real = nn.Parameter(torch.zeros(1,1,8,8))
        self.background_imag = nn.Parameter(torch.zeros(1,1,8,8))
        self.background_incoherent = nn.Parameter(torch.tensor(-12.))
        propagators = {}
        for key in config.active_slms:
            h, w = config.shape(key)
            p = getattr(config, key + '_pitch_um')
            grid = self.xy / torch.tensor([w * p / 2, h * p / 2])
            self.register_buffer(key + '_grid', grid)
            x, y = grid[0, ..., 0], grid[0, ..., 1]
            aperture = ((x.abs() <= 1) & (y.abs() <= 1)).float()[None, None]
            self.register_buffer(key + '_aperture', aperture)
            # Zernike-like low-order polynomial wavefront basis evaluated over
            # the rectangular active area: defocus, two astigmatisms, two comas.
            r2 = x.square() + y.square()
            basis = torch.stack((2*r2-1, x.square()-y.square(), 2*x*y,
                                 (3*r2-2)*x, (3*r2-2)*y))
            self.register_buffer(key + '_basis', basis)
            distance = (config.inter_slm_mm if key == 'slm1' and config.mode == 'dual'
                        else getattr(config, key + '_distance_mm'))
            propagators[key] = ASM(config.grid_shape, pitch, config.wavelength_nm, distance)
        self.propagators = nn.ModuleDict(propagators)
        ox = (torch.arange(config.output_width) + .5) / config.output_width * config.target_width_um - config.target_width_um / 2
        oy = (torch.arange(config.output_height) + .5) / config.output_height * config.target_height_um - config.target_height_um / 2
        oy, ox = torch.meshgrid(oy, ox, indexing='ij')
        extent = torch.tensor([width * pitch / 2, height * pitch / 2])
        self.register_buffer('output_grid', (torch.stack((ox, oy), -1) / extent)[None])

    def lut(self, key):
        increments = F.softplus(self.lut_steps[key]) + 1e-4
        curve = torch.cat((increments.new_zeros(1), increments.cumsum(0)))
        span = 2 * math.pi * torch.exp(self.phase_range[key].clamp(-.7, .7))
        return curve / curve[-1] * span

    def native_gray(self, controls, key, quantize=True):
        gray = F.interpolate(controls, size=self.config.shape(key), mode='bilinear', align_corners=False).clamp(0, 1)
        if quantize:
            gray = gray + (torch.round(gray * 255) / 255 - gray).detach()
        return gray

    def phase(self, gray, key):
        value = 1-gray if self.config.gray_inverted else gray
        position = value.clamp(0, 1) * 16
        index = position.long().clamp(max=15)
        curve = self.lut(key)
        return curve[index] + (curve[index+1]-curve[index]) * (position-index)

    def transmission(self, gray, key):
        phase = self.phase(gray, key)
        zero = self.phase(phase.new_zeros(()), key)
        phasor = torch.exp(1j*(phase-zero))
        grid = getattr(self, key + '_grid')
        real = F.grid_sample(phasor.real-1, grid, mode='bilinear', align_corners=False) + 1
        imag = F.grid_sample(phasor.imag, grid, mode='bilinear', align_corners=False)
        aberration = (self.aberrations[key][:, None, None] * getattr(self, key+'_basis')).sum(0)
        return torch.complex(real, imag) * torch.exp(1j*aberration*getattr(self,key+'_aperture'))

    def propagated_field(self, gray_maps, return_incident=False):
        amp = torch.exp(F.interpolate(self.source_log.clamp(-2, 2), size=self.config.grid_shape,
                                      mode='bilinear', align_corners=False))
        first = self.config.active_slms[0]
        mask = getattr(self, first + '_aperture')
        amp = amp / ((amp*mask).sum() / mask.sum()).clamp_min(1e-8)
        source_phase = F.interpolate(self.source_phase, size=self.config.grid_shape,
                                     mode='bilinear', align_corners=False)
        source_phase = source_phase - (source_phase*mask).sum()/mask.sum()
        field = torch.polar(amp*mask, source_phase)
        incident = {}
        for key in self.config.active_slms:
            if return_incident:
                incident[key] = field
            field = field * self.transmission(gray_maps[key], key)
            field = self.propagators[key](field)
        return (field, incident) if return_incident else field

    def forward(self, gray_maps):
        field = self.propagated_field(gray_maps)
        if self.config.model_background:
            real=F.interpolate(self.background_real,size=self.config.grid_shape,mode='bilinear',align_corners=False)
            imag=F.interpolate(self.background_imag,size=self.config.grid_shape,mode='bilinear',align_corners=False)
            field=field+torch.complex(real,imag)
        intensity=F.grid_sample(field.abs().square(), self.output_grid, mode='bilinear', align_corners=False)
        if self.config.model_background:
            intensity=intensity+F.softplus(self.background_incoherent)
        return intensity

    def regularization(self):
        smooth = (self.source_log[..., 1:, :]-self.source_log[..., :-1, :]).square().mean()
        smooth += (self.source_log[..., :, 1:]-self.source_log[..., :, :-1]).square().mean()
        penalty = .01*smooth + .001*self.source_log.square().mean()
        penalty = penalty + .001*self.source_phase.square().mean() + .01*(
            (self.source_phase[...,1:,:]-self.source_phase[...,:-1,:]).square().mean()
            +(self.source_phase[...,:,1:]-self.source_phase[...,:,:-1]).square().mean())
        if self.config.model_background:
            for value in (self.background_real,self.background_imag):
                penalty=penalty+.001*value.square().mean()
                penalty=penalty+.01*((value[...,1:,:]-value[...,:-1,:]).square().mean()
                    +(value[...,:,1:]-value[...,:,:-1]).square().mean())
            penalty=penalty+.001*F.softplus(self.background_incoherent)
        for key in self.config.active_slms:
            d = self.lut_steps[key]
            penalty = penalty + .001*(d[1:]-d[:-1]).square().mean() + .0001*self.aberrations[key].square().mean()
        return penalty

    def learned_state(self):
        return {key: value.detach().cpu().numpy() for key, value in self.named_parameters()}
