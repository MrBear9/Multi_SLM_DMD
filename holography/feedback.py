"""Measured proposal acceptance with Adam rollback and validated model updates."""
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .metrics import amplitude_loss,physical_proxy_loss


def optimize(model,controls,target,config,root,capture,emit,trainer=None):
    root=Path(root);optimizer=torch.optim.Adam(controls.values(),lr=config.learning_rate)
    records=[];accepted_steps=0;stalled=0;last_candidate=None
    def native(): return {k:model.native_gray(p,k) for k,p in controls.items()}
    def record(item):
        records.append(item)
        with (root/'feedback_history.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
    def checkpoint(name, score, step):
        # A durable pointer to the accepted capture survives stop/error, too.
        state={'capture':name,'camera_loss':score,'step':step,
               'model_path':trainer.current_path if trainer else config.model_path,
               'paths':{k:str(root/'captures'/name/(k+'.png')) for k in controls}}
        temporary=root/'best_state.tmp'
        temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(root/'best_state.json')
    def measure(gray,name):
        observations=[];scores=[]
        for repeat in range(config.feedback_verify_repeats):
            label=name if repeat==0 else f'{name}_repeat_{repeat:02d}'
            observed=capture(gray,label)
            score=float(amplitude_loss(observed,target))
            if not np.isfinite(score): raise ValueError('实拍误差不是有限数值。')
            scores.append(score);observations.append(observed)
        index=int(np.argmin(np.abs(np.asarray(scores)-np.median(scores))))
        return float(np.median(scores)),observations[index],float(np.ptp(scores)),scores
    current={k:v.detach().clone() for k,v in native().items()}
    score,observed,spread,scores=measure(current,'iteration_00000')
    initial_score=score
    checkpoint('iteration_00000',score,0)
    record({'kind':'initial','capture':'iteration_00000','camera_loss':score,'repeat_losses':scores})
    emit('metrics',camera_loss=score,best_loss=score)
    emit('log',message=f'初始化实拍误差 {score:.6f}；之后每个候选均采集验证。')
    stop_reason='steps_completed'
    for step in range(config.steps):
        accepted=False
        for retry in range(config.feedback_max_retries+1):
            lr=optimizer.param_groups[0]['lr']
            if lr<config.feedback_min_lr:
                stop_reason='minimum_learning_rate';break
            saved_controls={k:v.detach().clone() for k,v in controls.items()}
            saved_optimizer=copy.deepcopy(optimizer.state_dict())
            optimizer.zero_grad()
            prediction=model(native())
            proxy=physical_proxy_loss(prediction,observed,target)
            if not torch.isfinite(proxy): raise ValueError('回环梯度损失不是有限数值。')
            proxy.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in controls.values()):
                raise ValueError('回环相位梯度无效。')
            optimizer.step()
            with torch.no_grad():
                for p in controls.values():p.clamp_(0,1)
                proposal={k:v.detach().clone() for k,v in native().items()}
                simulated=float(amplitude_loss(model(proposal),target))
            name=f'iteration_{step+1:05d}_try_{retry:02d}'
            trial,trial_observed,trial_spread,trial_scores=measure(proposal,name)
            last_candidate=name
            tolerance=max(config.feedback_min_improvement,spread,trial_spread)
            accepted=bool(trial<score-tolerance)
            record({'kind':'proposal','step':step+1,'retry':retry,'capture':name,
                    'camera_loss':trial,'reference_loss':score,'simulation_loss':simulated,
                    'accepted':accepted,'learning_rate':lr,'acceptance_margin':tolerance,
                    'repeat_losses':trial_scores,'model_version':trainer.version if trainer else 0})
            emit('metrics',camera_loss=trial,best_loss=min(score,trial) if accepted else score,simulation_loss=simulated)
            emit('log',message=f'回环 {step+1}，尝试 {retry+1}：实拍 {trial:.6f} / 当前 {score:.6f}；'
                +('接受。' if accepted else '未达到实测改善门槛，回退相位和优化器。'))
            if trainer:trainer.add(name)
            if accepted:
                current,score,observed,spread=proposal,trial,trial_observed,trial_spread
                accepted_steps+=1;stalled=0
                checkpoint(name,score,step+1)
                break
            with torch.no_grad():
                for k,p in controls.items():p.copy_(saved_controls[k])
            optimizer.load_state_dict(saved_optimizer)
            for group in optimizer.param_groups:group['lr']=lr*config.feedback_lr_decay
            # Reload the accepted pattern, also obtaining a fresh physical value
            # for the next gradient. A rejected pattern never becomes the base.
            restored=capture(current,f'restore_{step+1:05d}_{retry:02d}')
            restored_score=float(amplitude_loss(restored,target))
            drift=abs(restored_score-score)
            record({'kind':'restore','step':step+1,'camera_loss':restored_score,'reference_loss':score})
            if drift>max(.02,3*tolerance):
                stop_reason='measurement_drift';break
            observed=restored
        if not accepted:stalled+=1
        emit('progress',value=int((step+1)/config.steps*95),message=f'回环验证 {step+1}/{config.steps}；已接受 {accepted_steps} 次，当前实拍误差 {score:.6f}')
        if stop_reason=='measurement_drift':break
        if trainer and config.feedback_online_model and ((step+1)%config.model_online_every==0 or not accepted):
            info,path=trainer.fit(config.model_online_epochs)
            record({'kind':'model_update','step':step+1,'validated':info['validated'],
                    'model_loss':info['holdout_loss'],'model_baseline':info['baseline_holdout_loss'],'model_path':path})
            if info['validated']:
                # Adam moments belong to the old model gradient, not the new one.
                # Retry its new direction at the initial step size. Measured
                # acceptance still protects the current phase against regression.
                optimizer=torch.optim.Adam(controls.values(),lr=config.learning_rate)
                stalled=0;stop_reason='steps_completed'
        if stop_reason=='minimum_learning_rate':break
        if stalled>=config.feedback_patience:
            stop_reason='no_measured_improvement';break
    if last_candidate:
        shutil.copytree(root/'captures'/last_candidate,root/'captures'/'candidate_final')
    final_score,_,final_spread,final_scores=measure(current,'best_verified')
    final_consistent=abs(final_score-score)<=max(.02,3*max(spread,final_spread,config.feedback_min_improvement))
    info={'initial_camera_loss':initial_score,'selected_camera_loss':score,'verified_camera_loss':final_score,
          'verified_repeat_losses':final_scores,'verification_consistent':final_consistent,
          'accepted_steps':accepted_steps,'stop_reason':stop_reason,'history':records,
          'loss_history':[r['camera_loss'] for r in records if r['kind'] in ('initial','proposal')],
          'active_model_path':trainer.current_path if trainer else config.model_path}
    (root/'feedback_summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    emit('metrics',camera_loss=final_score,best_loss=score)
    emit('log',message=f'回环结束：{stop_reason}；初始化 {initial_score:.6f} → 最佳复拍 {final_score:.6f}。'
         +('' if final_consistent else '警告：最佳相位复拍发生明显漂移，当前结果不能视为稳定改善。'))
    return current,info
