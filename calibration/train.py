"""ASM + torch.optim.SGD calibration phase training, usable from tools cwd."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'tools.calibration'

from .config import CalibrationConfig, write_json
from .patterns import TARGETS, save_gray, target_intensity
from .bundle import seal_bundle


def train_bundle(config, output, progress=lambda text: None, cancelled=lambda: False,
                 stages=('slm1', 'slm2', 'joint'), targets=TARGETS):
    import numpy as np
    import torch
    from .model import CalibrationModel, encode_phase, phase_lut
    from .registration import correlation

    config.validate()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'config.json', config.to_dict())
    torch.manual_seed(config.seed)
    device = config.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CalibrationModel(config).to(device)
    luts = {key: phase_lut(config, key) for key in ('slm1', 'slm2')}
    for key, lut in luts.items():
        np.save(root / (key + '_lut_rad.npy'), lut)
    manifest = {'version': 2, 'optimizer': 'torch.optim.SGD', 'device': device,
                'model': 'inverted 4f; SLM1 ASM; coordinate-based complex resampling; SLM2 ASM',
                'grids': config.grid_metadata(), 'output_grid': 'slm2',
                'status': 'training', 'config': config.to_dict(), 'patterns': []}
    write_json(root / 'manifest.json', manifest)
    for stage in stages:
        active = ('slm1', 'slm2') if stage == 'joint' else (stage,)
        for name in targets:
            if cancelled():
                raise InterruptedError('标定训练已取消。')
            expected = target_intensity(config, name)
            target = torch.from_numpy(expected)[None, None].to(device)
            target = target / target.mean().clamp_min(1e-8)
            parameters = {key: torch.nn.Parameter(torch.randn(
                1, 1, *config.phase_shape(key), device=device) * .1)
                for key in active}
            optimizer = torch.optim.SGD(list(parameters.values()), lr=config.learning_rate,
                                        momentum=config.momentum)
            # Pixelwise phase gradients scale with array size; normalize updates
            # to an RMS of one before SGD so the learning rate is in radians.
            for step in range(config.steps):
                if cancelled():
                    raise InterruptedError('标定训练已取消。')
                optimizer.zero_grad(set_to_none=True)
                phases = {key: model.native_phase(value, key) for key, value in parameters.items()}
                prediction = model(phases)
                normalized = prediction / prediction.mean().clamp_min(1e-8)
                loss = (torch.sqrt(normalized + 1e-8) - torch.sqrt(target + 1e-8)).square().mean()
                # A low-throughput pattern must not win solely through normalization.
                loss = loss + .02 * torch.relu(.05 - prediction.mean())
                loss.backward()
                for parameter in parameters.values():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise ValueError('训练梯度无效，请检查传播参数。')
                    parameter.grad.div_(parameter.grad.square().mean().sqrt().clamp_min(1e-8))
                torch.nn.utils.clip_grad_norm_(list(parameters.values()),
                    2 * max(value.numel() for value in parameters.values()) ** .5)
                optimizer.step()
                if step % 20 == 0 or step == config.steps - 1:
                    progress(f'训练 {stage}/{name}：{step + 1}/{config.steps}，loss={loss.item():.4f}')
            folder = root / stage / name
            folder.mkdir(parents=True)
            paths, quantized = {}, {}
            with torch.no_grad():
                for key, parameter in parameters.items():
                    phase = model.native_phase(parameter, key).squeeze().cpu().numpy()
                    gray = encode_phase(phase, luts[key])
                    path = save_gray(folder / (key + '.png'), gray)
                    paths[key] = path.relative_to(root).as_posix()
                    np.save(folder / (key + '_phase_rad.npy'), phase)
                    quantized[key] = torch.from_numpy(luts[key][gray])[None, None].to(device)
                prediction = model(quantized).squeeze().cpu().numpy()
            np.save(folder / 'expected.npy', prediction)
            np.save(folder / 'target.npy', expected)
            save_gray(folder / 'target.png', expected / max(expected.max(), 1e-9) * 255)
            save_gray(folder / 'simulation.png', prediction / max(prediction.max(), 1e-9) * 255)
            score = correlation(prediction, expected)
            manifest['patterns'].append({'stage': stage, 'name': name, 'paths': paths,
                'expected': (folder / 'expected.npy').relative_to(root).as_posix(),
                'target': (folder / 'target.npy').relative_to(root).as_posix(),
                'simulation_score': score, 'heldout': name == 'holdout'})
            write_json(root / 'manifest.json', manifest)
    manifest['status'] = 'trained'
    seal_bundle(root, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description='生成 ASM + SGD 光路标定相位图')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/calibration_training'))
    parser.add_argument('--write-default-config', type=Path)
    args = parser.parse_args(argv)
    config = CalibrationConfig.read(args.config) if args.config else CalibrationConfig()
    if args.write_default_config:
        write_json(args.write_default_config, config.to_dict())
        return 0
    train_bundle(config, args.output, progress=print)
    print(f'相位图和模拟结果已保存：{args.output.resolve()}；实物对准需运行 GUI 自动标定。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

