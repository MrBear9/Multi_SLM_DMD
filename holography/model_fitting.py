"""Disk-backed calibration replay, held-out model selection and reusable snapshots."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .metrics import amplitude_loss


def snapshot(model):
    return {k:p.detach().clone() for k,p in model.named_parameters()}


def restore(model,state):
    with torch.no_grad():
        for k,p in model.named_parameters(): p.copy_(state[k])


def load_model(model, config, job):
    manifest=Path(config.model_path)
    info=json.loads(manifest.read_text(encoding='utf-8'))
    if info.get('schema') not in (1,2) or not config.matches_optical_signature(info.get('signature')):
        raise ValueError('模型与当前光路、采样或有效区域不匹配。')
    if not info.get('validated'): raise ValueError('模型未通过独立验证，不能使用。')
    if config.live and info.get('hardware')!=job.get('hardware'):
        raise ValueError('模型的硬件、曝光或方向设置与当前会话不同，请重新标定。')
    if config.live and info.get('observation')!=config.observation_signature():
        raise ValueError('相机映射或黑白电平与标定模型不一致。')
    parameters=dict(model.named_parameters())
    added={'background_real','background_imag','background_incoherent','source_phase'}
    with np.load(manifest.parent/'model.npz',allow_pickle=False) as state:
        missing=set(parameters)-set(state.files)
        if set(state.files)-set(parameters) or (missing and not (info['schema']==1 and missing==added)):
            raise ValueError('模型参数不完整或结构不匹配。')
        config.model_background=bool(info.get('model_background',False))
        with torch.no_grad():
            for key in state.files:
                p=parameters[key];value=state[key]
                if value.shape!=tuple(p.shape) or not np.isfinite(value).all():
                    raise ValueError('模型参数尺寸或数值无效：'+key)
                p.copy_(torch.from_numpy(value).to(p.device))
    return info


class ModelTrainer:
    def __init__(self, model, config, root, job, capture, emit):
        self.model,self.config,self.root,self.job=model,config,Path(root),job
        self.capture,self.emit=capture,emit
        self.train=[];self.validation=[];self.holdout=[];self.version=0;self.attempt=0
        self.current_path=config.model_path
        self.device=next(model.parameters()).device

    def collect(self):
        """Separate phase patterns, not repeated frames, define the splits."""
        names=[];c=self.config
        generator=torch.Generator(device=self.device).manual_seed(c.seed+117)
        for index in range(c.samples):
            gray={}
            for key in c.active_slms:
                h,w=c.shape(key);side=(4,8,16,32)[index%4]
                controls=torch.rand(1,1,min(h,side),min(w,side),generator=generator,device=self.device)
                # Cover both broad and local LUT excursions, at multiple scales.
                if index%3==1: controls=(controls-.5)*.25+.5
                gray[key]=self.model.native_gray(controls,key).detach()
            name=f'sample_{index:04d}'
            self.capture(gray,name);names.append(name)
            self.emit('progress',value=int((index+1)/c.samples*30),message=f'模型采集 {index+1}/{c.samples}')
        order=np.random.default_rng(c.seed).permutation(len(names)).tolist()
        n=max(2,len(names)//5)
        self.holdout=[names[i] for i in order[:n]]
        self.validation=[names[i] for i in order[n:2*n]]
        self.train=[names[i] for i in order[2*n:]]
        self.partitions={'training_indices':order[2*n:],'validation_indices':order[n:2*n],'holdout_indices':order[:n]}

    def sample(self,name):
        folder=self.root/'captures'/name
        gray={k:torch.from_numpy(np.asarray(Image.open(folder/(k+'.png')),dtype=np.float32).copy())[None,None].to(self.device)/255
              for k in self.config.active_slms}
        observed=torch.from_numpy(np.load(folder/'observed.npy',allow_pickle=False))[None,None].to(self.device)
        return gray,observed

    @torch.no_grad()
    def evaluate(self,names):
        return float(np.mean([float(amplitude_loss(self.model(g),o)) for g,o in (self.sample(n) for n in names)]))

    def add(self,name):
        if not self.validation: return
        if name not in self.train and name not in self.validation and name not in self.holdout:
            self.train.append(name)
            # Keep the original diverse phases and bound the recent replay tail.
            keep=max(self.config.model_replay_size,len(self.partitions['training_indices']))
            initial=len(self.partitions['training_indices'])
            if len(self.train)>keep: del self.train[initial]

    def fit(self, epochs, destination=None, progress=False):
        c=self.config;m=self.model
        if not self.train or not self.validation or not self.holdout:
            raise ValueError('模型训练需要不同相位组成的训练、选择与独立验证组。')
        original=snapshot(m);best=original
        base_val=self.evaluate(self.validation);base_test=self.evaluate(self.holdout)
        best_val=base_val;selected_epoch=0;history=[]
        self.attempt+=1
        destination=Path(destination or self.root/'model_updates'/f'attempt_{self.attempt:04d}')
        destination.mkdir(parents=True,exist_ok=True)
        if (destination/'model.json').exists(): raise ValueError('模型保存位置已存在，拒绝覆盖。')
        try:
            for p in m.parameters(): p.requires_grad_(True)
            optimizer=torch.optim.Adam(m.parameters(),lr=c.model_learning_rate)
            for epoch in range(epochs):
                total=0.
                for name in self.train:
                    gray,observed=self.sample(name)
                    optimizer.zero_grad();loss=amplitude_loss(m(gray),observed)+m.regularization()
                    if not torch.isfinite(loss): raise ValueError('模型训练出现非有限损失。')
                    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);optimizer.step()
                    total+=float(loss.detach())
                score=self.evaluate(self.validation)
                history.append({'epoch':epoch+1,'train_loss':total/len(self.train),'validation_loss':score})
                if score<best_val: best_val=score;best=snapshot(m);selected_epoch=epoch+1
                self.emit('metrics',model_loss=score,model_baseline=base_val)
                if progress:
                    self.emit('progress',value=30+int(65*(epoch+1)/epochs),
                              message=f'传播模型训练 {epoch+1}/{epochs}，验证误差 {score:.6f}')
                if epoch%5==0 or epoch==epochs-1:
                    self.emit('log',message=f'模型训练 {epoch+1}/{epochs}：验证误差 {score:.6f}；基线 {base_val:.6f}')
            restore(m,best)
            test=self.evaluate(self.holdout)
            passed=bool(np.isfinite(test) and best_val<base_val*(1-c.model_validation_improvement)
                        and test<base_test*(1-c.model_validation_improvement))
            info={'schema':2,'signature':c.optical_signature(),'observation':c.observation_signature(),
                  'config':asdict(c),'hardware':self.job.get('hardware'),'model_background':c.model_background,
                  'baseline_holdout_loss':base_test,'holdout_loss':test,'validated':passed,
                  'baseline_validation_loss':base_val,'validation_loss':best_val,
                  'selected_epoch':selected_epoch,'history':history,**self.partitions,
                  'training_captures':self.train,'validation_captures':self.validation,'holdout_captures':self.holdout,
                  'metric':'mean squared RMS-normalized amplitude; effective optical model, not unique component metrology',
                  'parent_model':self.current_path}
            np.savez(destination/'model.npz',**m.learned_state())
            (destination/'model.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
            for key in c.active_slms:
                with torch.no_grad(): lut=m.phase(torch.linspace(0,1,256,device=self.device),key).cpu().numpy()
                np.savetxt(destination/(key+'_lut_rad.txt'),lut,fmt='%.9g')
            if passed:
                self.version+=1;self.current_path=str(destination/'model.json')
            else: restore(m,original)
            self.emit('log',message=f'模型独立验证 {base_test:.6f} → {test:.6f}；'+('采用新模型。' if passed else '改善不足，恢复原模型。'))
            return info,self.current_path
        except BaseException:
            restore(m,original)
            raise
        finally:
            for p in m.parameters(): p.requires_grad_(False)
