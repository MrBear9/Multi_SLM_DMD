"""External Python entrypoint; line-delimited JSON RPC over standard I/O.

Hardware is owned by the GUI. This process only computes and requests complete
display/capture transactions. Arrays are exchanged as local .npy files.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'tools.holography'

from .config import HolographyConfig


def emit(event_type, **values):
    print(json.dumps({'type': event_type, **values}, ensure_ascii=False), flush=True)


def run(job, rpc=None):
    import numpy as np
    from PIL import Image

    config = HolographyConfig.from_dict(job['config']).validate()
    root = Path(job['output']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root/'job.json').write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
    def request(payload):
        if rpc is not None:
            return rpc(payload)
        payload = dict(payload)
        emit(payload.pop('request_type', 'capture'), **payload)
        line = sys.stdin.readline()
        if not line:
            raise InterruptedError('硬件会话已关闭。')
        return json.loads(line)

    if config.geometry_profile and config.live:
        from .geometry import validate_profile
        validate_profile(config.geometry_profile, config, hardware=job.get('hardware'))
    if config.operation == 'geometry':
        from .geometry_worker import run_geometry
        return run_geometry(job, config, request, emit)
    import torch
    from .optics import OpticalModel
    from .camera import rectify
    from .target import prepare_target
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emit('log', message=f'独立训练进程：{device}；模式 {config.mode}；'
         f'ASM {config.grid_width}×{config.grid_height}，FFT {2*config.grid_width}×{2*config.grid_height}')
    report = config.grid_report()
    emit('log', message=f'计算视野预留倍率 {config.grid_margin:g}；网格单轴上限 {config.grid_limit}；'
         f"单个 FFT 复数数组约 {report['fft_array_mib']:.0f} MiB，训练总占用更高。")
    model = OpticalModel(config).to(device)
    emit('log', message=f'目标计算：{config.output_width}×{config.output_height}；'
         f'目标采样：{config.target_width_um/config.output_width:g}×'
         f'{config.target_height_um/config.output_height:g} μm')
    baseline = None
    if config.model_path:
        manifest = Path(config.model_path)
        info = json.loads(manifest.read_text(encoding='utf-8'))
        if info.get('schema') != 1 or not config.matches_optical_signature(info.get('signature')):
            raise ValueError('模型与当前光路、采样或有效区域不匹配。')
        if not info.get('validated'):
            raise ValueError('模型未通过留出样本验证，不能用于全息生成。')
        if config.live and info.get('hardware') != job.get('hardware'):
            raise ValueError('模型的硬件、曝光或方向设置与当前会话不同，请使用原配置或重新标定。')
        if config.live and info.get('observation') != config.observation_signature():
            raise ValueError('相机映射或黑白电平与标定模型不一致。')
        weights = manifest.parent/'model.npz'
        with np.load(weights, allow_pickle=False) as state:
            parameters = dict(model.named_parameters())
            if set(state.files) != set(parameters):
                raise ValueError('模型参数不完整。')
            with torch.no_grad():
                for key, p in parameters.items():
                    value = state[key]
                    if value.shape != tuple(p.shape) or not np.isfinite(value).all():
                        raise ValueError('模型参数尺寸或数值无效：'+key)
                    p.copy_(torch.from_numpy(value).to(device))

    def normalized(value):
        return value / value.mean().clamp_min(1e-8)

    def loss_fn(prediction, target):
        return (torch.sqrt(normalized(prediction)+1e-7)-torch.sqrt(normalized(target)+1e-7)).square().mean()

    def save_preview(path, values):
        arr = np.asarray(values, np.float32).squeeze()
        Image.fromarray(np.uint8(np.clip(arr/max(float(arr.max()), 1e-8), 0, 1)*255)).save(path)

    def capture(gray, name):
        folder = root/'captures'/name
        folder.mkdir(parents=True, exist_ok=True)
        paths = {}
        for key, value in gray.items():
            array = np.uint8(np.rint(value.detach().cpu().numpy().squeeze()*255))
            path = folder/(key+'.png')
            Image.fromarray(array).save(path)
            paths[key] = str(path)
        payload = {'paths': paths, 'capture_path': str(folder/'raw.npy')}
        result = request(payload)
        if result.get('error'):
            raise RuntimeError(result['error'])
        raw = np.load(result['path'], allow_pickle=False)
        emit('preview', path=str(folder/'raw.npy'), kind='camera_raw')
        observed = rectify(raw, config)
        np.save(folder/'observed.npy', observed)
        save_preview(folder/'observed.png', observed)
        emit('preview', path=str(folder/'observed.png'), kind='camera')
        return torch.from_numpy(observed)[None,None].to(device)

    if config.operation == 'calibrate':
        # Diverse known grey patterns excite different spatial/grey responses.
        # Held-out samples are never used in the parameter optimizer.
        dataset = []
        for index in range(config.samples):
            gray = {}
            for key in config.active_slms:
                size = (4, 8, 16, 32)[index % 4]
                controls = torch.rand(1, 1, size, size, device=device)
                gray[key] = model.native_gray(controls, key).detach()
            measured = capture(gray, f'sample_{index:04d}')
            dataset.append(({k:v.cpu() for k,v in gray.items()}, measured.cpu()))
            emit('progress', value=int((index+1)/config.samples*35), message=f'采集 {index+1}/{config.samples}')
        indices = np.random.default_rng(config.seed).permutation(len(dataset)).tolist()
        held = indices[:max(2,len(dataset)//4)]
        train = indices[len(held):]

        def evaluate(indices):
            with torch.no_grad():
                return float(np.mean([loss_fn(model({k:v.to(device) for k,v in dataset[i][0].items()}),
                                                dataset[i][1].to(device)).item() for i in indices]))

        baseline = evaluate(held)
        optimizer = torch.optim.Adam(model.parameters(), lr=min(config.learning_rate, .01))
        # Held-out data are evaluated only before and after fitting.
        for epoch in range(config.epochs):
            total = 0.
            for i in train:
                gray, observed = dataset[i]
                optimizer.zero_grad()
                prediction = model({k:v.to(device) for k,v in gray.items()})
                loss = loss_fn(prediction, observed.to(device)) + model.regularization()
                if not torch.isfinite(loss):
                    raise ValueError('模型训练出现非有限损失。')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += loss.item()
            if epoch % 5 == 0 or epoch == config.epochs-1:
                emit('progress', value=35+int((epoch+1)/config.epochs*60), message=f'模型训练 {epoch+1}/{config.epochs}，loss={total/len(train):.5f}')
        score = evaluate(held)
        passed = bool(np.isfinite(score) and score < baseline*.98)
        np.savez(root/'model.npz', **model.learned_state())
        for key in config.active_slms:
            with torch.no_grad():
                gray = torch.linspace(0, 1, 256, device=device)
                lut = model.phase(gray, key).cpu().numpy()
            np.savetxt(root/(key+'_lut_rad.txt'), lut, fmt='%.9g')
        info = {'schema': 1, 'signature': config.optical_signature(), 'config': asdict(config),
                'observation': config.observation_signature(),
                'hardware': job.get('hardware'), 'baseline_holdout_loss': baseline,
                'holdout_loss': score, 'validated': passed, 'training_indices': train, 'holdout_indices': held,
                'interpretation': 'effective model for this fixed illumination and camera geometry; not unique component metrology'}
        (root/'model.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        emit('result', path=str(root/'model.json'), message=(f'模型留出误差 {baseline:.5f} → {score:.5f}；' +
             ('通过相对改善验证，仍需独立目标实测。' if passed else '未通过改善验证，已保留诊断数据，禁止作为已标定模型使用。')))
        return info

    for p in model.parameters():
        p.requires_grad_(False)
    target = torch.from_numpy(prepare_target(config)).float()[None,None].to(device)
    if target.max() <= 0:
        raise ValueError('目标图不能全黑。')
    save_preview(root/'target.png', target.cpu().numpy())
    emit('preview', path=str(root/'target.png'), kind='target')
    from .initialization import initialize_controls
    emit('log', message=f'初始化：{config.initialization}；随机相位范围 ±{config.init_range_rad:g} rad')
    initialized, initialization_info = initialize_controls(model, target, emit)
    controls = {k: torch.nn.Parameter(value) for k, value in initialized.items()}
    emit('log', message=f"初始化模拟误差 {initialization_info['initial_loss']:.5f} → "
         f"{initialization_info['selected_loss']:.5f}；开始 Adam 优化。")
    for key in config.active_slms:
        ch, cw = config.control_shape(key)
        h, w = config.shape(key)
        emit('log', message=f'{key} 控制点 {cw}×{ch} → 导出相位 {w}×{h}')
    optimizer = torch.optim.Adam(controls.values(), lr=config.learning_rate)
    history = []
    best = None
    best_loss = float('inf')
    for step in range(config.steps):
        optimizer.zero_grad()
        gray = {k:model.native_gray(p,k) for k,p in controls.items()}
        prediction = model(gray)
        if config.operation == 'feedback' and step % config.feedback_every == 0:
            observed = capture(gray, f'iteration_{step:05d}')
            measured_loss = float(loss_fn(observed,target).item())
            if measured_loss < best_loss:
                best_loss = measured_loss
                best = {k:v.detach().clone() for k,v in gray.items()}
            # Physical forward value, differentiable surrogate-model gradient.
            correction = (normalized(observed)-normalized(prediction)).detach()
        corrected = (normalized(prediction)+correction).clamp_min(0) if config.operation == 'feedback' else prediction
        loss = loss_fn(corrected, target)
        if not torch.isfinite(loss):
            raise ValueError('相位优化出现非有限损失。')
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for p in controls.values():
                p.clamp_(0,1)
        history.append(float(loss.item()))
        if step % 10 == 0 or step == config.steps-1:
            emit('progress', value=int((step+1)/config.steps*95), message=f'全息优化 {step+1}/{config.steps}，loss={loss.item():.5f}')
    final = {k:model.native_gray(p,k).detach() for k,p in controls.items()}
    if config.operation == 'feedback':
        observed = capture(final, 'candidate_final')
        score = float(loss_fn(observed,target).item())
        if score < best_loss:
            best_loss, best = score, final
        final = best
        verified = capture(final, 'best_verified')
        final_score = float(loss_fn(verified,target).item())
    else:
        final_score = None
    paths = {}
    for key,value in final.items():
        (root/key).mkdir(exist_ok=True)
        path = root/key/'phase.png'
        Image.fromarray(np.uint8(np.rint(value.cpu().numpy().squeeze()*255))).save(path)
        paths[key] = str(path)
    with torch.no_grad():
        simulated = model(final).cpu().numpy().squeeze()
    np.save(root/'simulation.npy', simulated)
    save_preview(root/'simulation.png', simulated)
    emit('preview', path=str(root/'simulation.png'), kind='simulation')
    result = {'schema': 1, 'config': asdict(config), 'paths': paths,
              'loss_history': history, 'verified_camera_loss': final_score,
              'initialization': initialization_info,
              'hardware': job.get('hardware'), 'model_path': config.model_path}
    (root/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    emit('result', path=str(root/'result.json'), message='全息相位已导出。' + ('已保存最终实拍验证。' if config.live else '尚未进行实物验证。'))
    return result


def main():
    try:
        job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
        run(job)
    except Exception as exc:
        emit('error', message=str(exc))
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
