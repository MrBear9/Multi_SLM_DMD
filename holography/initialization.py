"""Physical-phase initialization, imported only by the external torch worker.

GS uses target-plane amplitude projection and adjoint ASM propagation. Native
pixel/control projection and a calibrated LUT make this a projected GS variant,
not an exact inverse of the cropped, band-limited propagation operator.
"""
import math

import torch
import torch.nn.functional as F
from .metrics import amplitude_loss


def phase_to_gray(model, relative_phase, key):
    """Nearest realizable 8-bit phase on the circle, including inverted LUTs."""
    gray = torch.linspace(0, 1, 256, device=relative_phase.device)
    zero = model.phase(gray.new_zeros(()), key)
    angles = torch.remainder(model.phase(gray, key) - zero, 2*math.pi)
    angles, order = angles.sort()
    wanted = torch.remainder(relative_phase, 2*math.pi).contiguous()
    right = torch.searchsorted(angles, wanted).remainder(256)
    left = (right-1).remainder(256)
    dr = torch.atan2(torch.sin(angles[right]-wanted), torch.cos(angles[right]-wanted)).abs()
    dl = torch.atan2(torch.sin(angles[left]-wanted), torch.cos(angles[left]-wanted)).abs()
    return gray[order[torch.where(dl <= dr, left, right)]]


@torch.no_grad()
def initialize_controls(model, target, emit=None):
    config = model.config
    controls = {}
    for key in config.active_slms:
        # Center near middle grey so a small range does not alternate between
        # grey 0 and 255 and interpolate through unrelated phases on coarse grids.
        center = model.phase(target.new_tensor(.5), key)-model.phase(target.new_zeros(()), key)
        phase = center + (2*torch.rand(1, 1, *config.control_shape(key), device=target.device)-1)*config.init_range_rad
        controls[key] = phase_to_gray(model, phase, key)
    def native():
        return {key: model.native_gray(value, key) for key, value in controls.items()}
    initial_loss = float(amplitude_loss(model(native()), target))
    info = {'method': config.initialization, 'range_rad': config.init_range_rad,
            'initial_loss': initial_loss, 'selected_loss': initial_loss,
            'selected_iteration': 0, 'iterations': 0}
    if config.initialization == 'random':
        return controls, info

    # Define the target in physical coordinates on the propagation plane.
    # Preserve the unconstrained field outside the target rectangle.
    extent = target.new_tensor([config.target_width_um/2, config.target_height_um/2])
    target_grid = model.xy / extent
    mask = (target_grid.abs() <= 1).all(-1)[:, None]
    intensity = F.grid_sample(target, target_grid, mode='bilinear',
                              padding_mode='border', align_corners=False)
    if not mask.any() or float(intensity[mask].sum()) <= 0:
        raise ValueError('GS 初始化的目标在 ASM 网格上没有有效采样，请减小采样间距或增大目标尺寸。')
    best = {key: value.clone() for key, value in controls.items()}
    best_loss = initial_loss
    native_grids = {}
    for key in config.active_slms:
        h, w = config.shape(key)
        p = getattr(config, key+'_pitch_um')
        y = (torch.arange(h, device=target.device)-(h-1)/2)*p
        x = (torch.arange(w, device=target.device)-(w-1)/2)*p
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        native_grids[key] = torch.stack((xx/(config.grid_width*config.sampling_um/2),
                                        yy/(config.grid_height*config.sampling_um/2)), -1)[None]

    for iteration in range(config.gs_iterations):
        maps = native()
        field, incident = model.propagated_field(maps, return_incident=True)
        scale = field.abs().square()[mask].sum()/intensity[mask].sum().clamp_min(1e-12)
        constrained = torch.polar((intensity*scale).clamp_min(0).sqrt(), field.angle())
        back = torch.where(mask, constrained, field)
        # In dual mode project the second SLM first, then propagate the desired
        # incident field back to the first. Illumination always comes from SLM1.
        for key in reversed(config.active_slms):
            back = model.propagators[key](back, adjoint=True)
            aberration = (model.aberrations[key][:, None, None]*getattr(model,key+'_basis')).sum(0)
            desired = back * incident[key].conj() * torch.exp(-1j*aberration*getattr(model,key+'_aperture'))
            grid = native_grids[key]
            real = F.grid_sample(desired.real, grid, align_corners=False)
            imag = F.grid_sample(desired.imag, grid, align_corners=False)
            # Circular interpolation avoids averaging +pi and -pi to zero.
            unit = torch.polar(torch.ones_like(real), torch.atan2(imag, real))
            shape = config.control_shape(key)
            re = F.interpolate(unit.real, size=shape, mode='bilinear', align_corners=False)
            im = F.interpolate(unit.imag, size=shape, mode='bilinear', align_corners=False)
            proposal = phase_to_gray(model, torch.atan2(im, re), key)
            strength = F.interpolate(real.square()+imag.square(), size=shape, mode='bilinear', align_corners=False)
            controls[key] = torch.where(strength > 1e-16, proposal, controls[key])
            maps[key] = model.native_gray(controls[key], key)
            back = back * model.transmission(maps[key], key).conj()
        loss = float(amplitude_loss(model(native()), target))
        if not math.isfinite(loss):
            raise ValueError('GS 初始化出现非有限误差，请检查光路和采样参数。')
        if loss < best_loss:
            best_loss = loss
            best = {key: value.clone() for key, value in controls.items()}
            info['selected_iteration'] = iteration+1
        if emit and (iteration % 5 == 0 or iteration == config.gs_iterations-1):
            emit('log', message=f'GS 初始化 {iteration+1}/{config.gs_iterations}，模拟误差={loss:.5f}，最佳={best_loss:.5f}')
    info.update(iterations=config.gs_iterations, selected_loss=best_loss)
    return best, info
