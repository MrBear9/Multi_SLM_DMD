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
    if config.geometry_profile.get('target_plane_assumption'):
        emit('log',message='警告：本次使用棋盘格映射，假设 SLM 面与目标平面 1:1 对应；绝对重建尺度及完整边缘尚未独立验证。')
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
    from .metrics import amplitude_loss as loss_fn
    from .model_fitting import ModelTrainer, load_model
    if config.model_path:
        load_model(model, config, job)
        emit('log', message='已加载通过验证的传播模型：'+config.model_path)

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

    trainer = ModelTrainer(model,config,root,job,capture,emit)
    if config.operation == 'calibrate':
        trainer.collect()
        info,_ = trainer.fit(config.epochs,root,progress=True)
        emit('result',path=str(root/'model.json'),message=(
            '模型已保存，可用于后续回环。' if info['validated'] else
            '模型未通过独立验证；已保存诊断，未采用该模型。'))
        return info
    if config.operation == 'feedback':
        if (config.feedback_auto_calibrate and not config.model_path) or config.feedback_online_model:
            trainer.collect()
        if config.feedback_auto_calibrate and not config.model_path:
            info,_ = trainer.fit(config.epochs,progress=True)
            if not info['validated']:
                raise ValueError('回环前模型标定未通过独立验证，已保存采集与诊断。请检查光路、映射和模型设置后重试；也可明确关闭自动标定以测试默认模型。')
        elif not config.model_path:
            emit('log',message='警告：未使用已标定传播模型；将逐步实拍验证并拒绝变差的候选。')

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
    feedback_info = None
    if config.operation == 'feedback':
        from .feedback import optimize
        final,feedback_info = optimize(model,controls,target,config,root,capture,emit,trainer)
        history = feedback_info['loss_history']
        final_score = feedback_info['verified_camera_loss']
    else:
        optimizer = torch.optim.Adam(controls.values(),lr=config.learning_rate)
        history=[]
        for step in range(config.steps):
            optimizer.zero_grad()
            prediction=model({k:model.native_gray(p,k) for k,p in controls.items()})
            loss=loss_fn(prediction,target)
            if not torch.isfinite(loss):raise ValueError('相位优化出现非有限损失。')
            loss.backward();optimizer.step()
            with torch.no_grad():
                for p in controls.values():p.clamp_(0,1)
            history.append(float(loss.detach()))
            if step%10==0 or step==config.steps-1:
                emit('metrics',simulation_loss=float(loss.detach()))
                emit('progress',value=int((step+1)/config.steps*95),message=f'模拟优化 {step+1}/{config.steps}，loss={loss.item():.5f}')
        final={k:model.native_gray(p,k).detach() for k,p in controls.items()}
        final_score=None
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
              'hardware': job.get('hardware'), 'model_path': trainer.current_path, 'feedback': feedback_info}
    (root/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    emit('result', path=str(root/'result.json'), message='全息相位已导出。' + (('已保存最终实拍验证。' if feedback_info['verification_consistent'] else '最佳复拍存在漂移，请查看回环记录。') if config.live else '尚未进行实物验证。'))
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
