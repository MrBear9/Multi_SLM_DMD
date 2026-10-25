"""Run with python -m unittest tools.holography.feedback_regressions (no hardware)."""
from dataclasses import asdict, replace
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from .config import HolographyConfig
from .feedback import optimize
from .metrics import amplitude_loss, physical_proxy_loss
from .model_fitting import ModelTrainer, load_model, snapshot
from .optics import OpticalModel


def small(**changes):
    base=HolographyConfig(mode='slm1',slm1_width=16,slm1_height=16,
        slm1_controls_width=8,slm1_controls_height=8,grid_width=64,grid_height=64,
        sampling_um=4.,output_width=16,output_height=16,target_width_um=64.,target_height_um=64.,
        slm1_distance_mm=1.,camera_center_x=31.5,camera_center_y=31.5,camera_pitch_um=4.,
        mapping_confirmed=True,target_kind='cross',samples=8,epochs=2,steps=3,
        feedback_auto_calibrate=False,feedback_max_retries=2,feedback_patience=1,
        feedback_min_improvement=.00001,model_learning_rate=.03)
    return replace(base,**changes)


def quiet(*args,**kwargs): pass


class ToyModel(torch.nn.Module):
    """Known one-parameter response, with real 8-bit phase replay."""
    def __init__(self,config):
        super().__init__();self.config=config
        self.log_gamma=torch.nn.Parameter(torch.tensor(0.))

    def native_gray(self,p,key):
        value=p.clamp(0,1)
        return value+(value.mul(255).round()/255-value).detach()

    def forward(self,gray): return (.05+gray['slm1']).pow(self.log_gamma.exp())
    def regularization(self): return self.log_gamma.square()*0
    def learned_state(self): return {k:v.detach().numpy() for k,v in self.named_parameters()}
    def phase(self,gray,key): return gray*6.283185


def disk_capture(root, transform):
    def capture(gray,name):
        folder=root/'captures'/name;folder.mkdir(parents=True)
        native={}
        for key,value in gray.items():
            array=np.rint(value.detach().cpu().numpy()[0,0]*255).astype(np.uint8)
            Image.fromarray(array).save(folder/(key+'.png'))
            native[key]=torch.from_numpy(array.astype(np.float32))[None,None]/255
        with torch.no_grad(): observed=transform(native,name).detach()
        np.save(folder/'observed.npy',observed.numpy()[0,0])
        np.save(folder/'raw.npy',observed.numpy()[0,0])
        return observed
    return capture


class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_proxy_uses_actual_score_but_model_gradient(self):
        p=torch.tensor([.2,.8],requires_grad=True);actual=torch.tensor([.8,.2]);target=torch.tensor([.9,.1])
        loss=physical_proxy_loss(p,actual,target)
        self.assertAlmostEqual(float(loss),float(amplitude_loss(actual,target)),places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)

    def run_loop(self, opposite=False, drift=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);config=small();model=ToyModel(config)
        model.requires_grad_(False)
        controls={'slm1':torch.nn.Parameter(torch.full((1,1,1,2),.5))}
        target=torch.tensor([[[[.9,.1]]]])
        def truth(gray,name):
            signal=1-gray['slm1'] if opposite else gray['slm1']
            if drift and name.startswith('best_verified'):signal=1-signal
            return signal+.05
        final,info=optimize(model,controls,target,config,root,disk_capture(root,truth),quiet)
        return root,final,info,controls

    def test_wrong_gradient_rejected_and_initialization_preserved(self):
        root,final,info,controls=self.run_loop(opposite=True)
        self.assertEqual(info['accepted_steps'],0)
        torch.testing.assert_close(controls['slm1'],torch.full((1,1,1,2),.5))
        self.assertEqual(info['initial_camera_loss'],info['verified_camera_loss'])
        proposals=[x for x in info['history'] if x['kind']=='proposal']
        self.assertEqual(len(proposals),3)
        self.assertTrue(all(not x['accepted'] for x in proposals))
        self.assertEqual([x['learning_rate'] for x in proposals],[.03,.015,.0075])
        self.assertEqual((root/'captures/iteration_00000/slm1.png').read_bytes(),
                         (root/'captures/best_verified/slm1.png').read_bytes())
        self.assertEqual(json.loads((root/'best_state.json').read_text())['step'],0)

    def test_good_gradient_improves_only_measured_history(self):
        root,final,info,_=self.run_loop()
        self.assertEqual(info['accepted_steps'],3)
        self.assertLess(info['verified_camera_loss'],info['initial_camera_loss'])
        self.assertTrue(info['verification_consistent'])
        accepted=[r for r in info['history'] if r['kind']=='proposal' and r['accepted']]
        self.assertTrue(all(r['camera_loss']<r['reference_loss'] for r in accepted))
        self.assertEqual(info['loss_history'],[r['camera_loss'] for r in info['history'] if r['kind'] in ('initial','proposal')])
        self.assertEqual(json.loads((root/'best_state.json').read_text())['step'],3)

    def test_final_drift_is_reported(self):
        _,_,info,_=self.run_loop(drift=True)
        self.assertFalse(info['verification_consistent'])

    def test_rejection_restores_adam_moments_after_an_accepted_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c=small();m=ToyModel(c).requires_grad_(False)
            controls={'slm1':torch.nn.Parameter(torch.full((1,1,1,2),.5))}
            def truth(gray,name):
                value=gray['slm1']
                return (.05+1-value) if name.startswith('iteration_00002') else .05+value
            adam=torch.optim.Adam(controls.values(),lr=c.learning_rate)
            with patch('tools.holography.feedback.torch.optim.Adam',return_value=adam):
                final,info=optimize(m,controls,torch.tensor([[[[.9,.1]]]]),c,root,disk_capture(root,truth),quiet)
            self.assertEqual(info['accepted_steps'],1)
            self.assertEqual(int(adam.state[controls['slm1']]['step']),1)
            self.assertEqual((root/'captures/iteration_00001_try_00/slm1.png').read_bytes(),
                             (root/'captures/best_verified/slm1.png').read_bytes())

    def test_worker_prefit_and_online_updates_use_saved_model_versions(self):
        from .worker import run
        c=small(operation='feedback',feedback_auto_calibrate=True,feedback_online_model=True,
                model_online_every=1,model_online_epochs=2,camera_white=65535.,epochs=4)
        truth=OpticalModel(c)
        with torch.no_grad():truth.aberrations['slm1'][0]=.3
        with tempfile.TemporaryDirectory() as directory:
            def rpc(payload):
                gray={key:torch.from_numpy(np.asarray(Image.open(path),np.float32).copy())[None,None]/255
                      for key,path in payload['paths'].items()}
                with torch.no_grad():signal=truth(gray).numpy()[0,0]*100
                raw=np.zeros((64,64),np.float32);raw[24:40,24:40]=signal
                np.save(payload['capture_path'],raw)
                return {'path':payload['capture_path']}
            with contextlib.redirect_stdout(io.StringIO()):
                result=run({'config':asdict(c),'output':directory},rpc)
            models=list((Path(directory)/'model_updates').glob('*/model.json'))
            self.assertGreaterEqual(len(models),2)
            self.assertTrue(any(r['kind']=='model_update' for r in result['feedback']['history']))
            active=json.loads(Path(result['model_path']).read_text(encoding='utf-8'))
            self.assertTrue(active['validated'])
            self.assertFalse(set(active['training_captures'])&set(active['holdout_captures']))

    def test_interrupted_capture_retains_best_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c=small();m=ToyModel(c).requires_grad_(False)
            def truth(gray,name):
                if name.startswith('iteration_00001'):raise InterruptedError('capture stopped')
                return gray['slm1']+.05
            with self.assertRaises(InterruptedError):
                optimize(m,{'slm1':torch.nn.Parameter(torch.full((1,1,1,2),.5))},
                    torch.tensor([[[[.9,.1]]]]),c,root,disk_capture(root,truth),quiet)
            state=json.loads((root/'best_state.json').read_text())
            self.assertEqual(state['step'],0);self.assertTrue(Path(state['paths']['slm1']).is_file())

    def trainer(self,root,contradict=False):
        c=small();m=ToyModel(c)
        holdout=set(np.random.default_rng(c.seed).permutation(c.samples)[:2])
        def truth(gray,name):
            index=int(name.split('_')[-1])
            gamma=.4 if contradict and index in holdout else 2.5
            return (.05+gray['slm1']).pow(gamma)
        trainer=ModelTrainer(m,c,root,{'hardware':{}},disk_capture(root,truth),quiet)
        trainer.collect();return trainer

    def test_model_learns_independent_phases_and_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer=self.trainer(Path(directory))
            groups=[set(trainer.train),set(trainer.validation),set(trainer.holdout)]
            self.assertTrue(all(not a&b for i,a in enumerate(groups) for b in groups[i+1:]))
            info,path=trainer.fit(25)
            self.assertTrue(info['validated']);self.assertEqual(trainer.version,1)
            self.assertLess(info['holdout_loss'],info['baseline_holdout_loss']*.98)
            c=replace(trainer.config,model_path=path);loaded=ToyModel(c)
            load_model(loaded,c,{'hardware':{}})
            torch.testing.assert_close(loaded.log_gamma,trainer.model.log_gamma)
            validation=list(trainer.validation);held=list(trainer.holdout)
            trainer.add(validation[0]);trainer.add(held[0])
            self.assertEqual(trainer.validation,validation);self.assertEqual(trainer.holdout,held)
            self.assertNotIn(held[0],trainer.train)

    def test_model_holdout_failure_rolls_back_all_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer=self.trainer(Path(directory),contradict=True);before=snapshot(trainer.model)
            info,path=trainer.fit(25)
            self.assertFalse(info['validated']);self.assertEqual(path,'');self.assertEqual(trainer.version,0)
            for key,value in trainer.model.named_parameters():torch.testing.assert_close(value,before[key])
            self.assertTrue((Path(directory)/'model_updates/attempt_0001/model.json').is_file())

    def test_new_wavefront_and_background_gradients_and_legacy_model(self):
        c=small();m=OpticalModel(c)
        gray={'slm1':torch.rand(1,1,16,16)}
        amplitude_loss(m(gray),torch.rand(1,1,16,16)).backward()
        for name in ('source_phase','source_log','background_real','background_imag'):
            self.assertGreater(float(getattr(m,name).grad.abs().sum()),0,name)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);legacy=OpticalModel(replace(c,model_background=False))
            state=legacy.learned_state()
            for key in ('source_phase','background_real','background_imag','background_incoherent'):state.pop(key)
            np.savez(root/'model.npz',**state)
            (root/'model.json').write_text(json.dumps({'schema':1,'validated':True,'signature':c.optical_signature()}))
            loaded=OpticalModel(c);c.model_path=str(root/'model.json');load_model(loaded,c,{})
            self.assertFalse(c.model_background)
            torch.testing.assert_close(loaded(gray),legacy(gray))

    def test_default_auto_calibration_failure_stops_before_phase_optimization(self):
        from .worker import run
        c=small(operation='feedback',feedback_auto_calibrate=True,camera_white=65535.)
        with tempfile.TemporaryDirectory() as directory:
            def rpc(payload):
                np.save(payload['capture_path'],np.full((64,64),100.,np.float32))
                return {'path':payload['capture_path']}
            # An unchanged holdout baseline is never accepted as a learned model.
            with patch.object(ModelTrainer,'evaluate',return_value=1.),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'模型标定未通过'):
                    run({'config':asdict(c),'output':directory},rpc)
            self.assertFalse((Path(directory)/'captures/iteration_00000').exists())
            self.assertTrue((Path(directory)/'model_updates/attempt_0001/model.json').exists())


class UiAndCameraTests(unittest.TestCase):
    def test_capture_exposure_mismatch_keeps_raw_diagnostics(self):
        from tools.gui.holography_controller import HolographyCaptureTask
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            task=HolographyCaptureTask(small(),root,
                {'ccdExposureSpinBox':8000.,'ccdGainSpinBox':1.},
                {'paths':{},'capture_path':str(root/'raw.npy')},None)
            def rpc(kind,value):
                if kind=='parameters':return {'auto':False,'exposure':8000.,'gain':1.}
                if kind=='capture':return np.ones((16,16),np.uint8),[{'auto':True,'exposure':9000.,'gain':1.}],None
            task.io=rpc;task.run()
            self.assertIn('曝光',task.error)
            self.assertTrue((root/'raw.npy').is_file());self.assertTrue((root/'raw.json').is_file())

    def test_parameters_roundtrip_and_metrics(self):
        os.environ['QT_QPA_PLATFORM']='offscreen'
        from PyQt5 import QtWidgets,QtCore
        from tools.gui.main_window import MainWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);settings_class=QtCore.QSettings
            with patch('tools.gui.holography_window.QtCore.QSettings',lambda *a:settings_class(str(root/'settings.ini'),settings_class.IniFormat)):
                w=MainWindow(demo=True);d=w.holography.ensure_dialog()
                try:
                    c=small(operation='feedback',feedback_online_model=True,feedback_auto_calibrate=True,
                        feedback_max_retries=3,feedback_lr_decay=.4,feedback_min_lr=.0002,
                        feedback_min_improvement=.003,feedback_verify_repeats=3,feedback_patience=4,
                        model_online_every=7,model_online_epochs=4,model_replay_size=48,
                        model_validation_improvement=.04,model_background=False)
                    d.set_values(c)
                    path=str(root/'params.json')
                    with patch.object(QtWidgets.QFileDialog,'getSaveFileName',return_value=(path,'')):d.save_config()
                    d.set_values(small())
                    with patch.object(QtWidgets.QFileDialog,'getOpenFileName',return_value=(path,'')):d.load_config()
                    for key,value in asdict(c).items():
                        if key.startswith(('feedback_','model_')):self.assertEqual(getattr(d.values(),key),value,key)
                    d.update_metrics({'camera_loss':.3,'best_loss':.2,'model_loss':.1,'simulation_loss':.05})
                    self.assertIn('0.300000',d.cameraLossLabel.text())
                    self.assertIn('0.200000',d.bestLossLabel.text())
                    self.assertFalse(d.feedback_every.isEnabled())
                finally:d.close();w.close();app.processEvents()

    def test_toup_apply_refreshes_frame_parameter_cache(self):
        from tools.devices.toupcam_camera.controller import ToupCamera
        c=object.__new__(ToupCamera);c.exposure_range=(1,100000);c.gain_range=(1,5)
        c.cached_parameters={'auto':True,'exposure':9000};c.call=lambda *a:None
        c.ensure_fresh=lambda *a:None;c.parameters=lambda:dict(auto=False,exposure=8000.,gain=1.)
        returned=c.apply(8000,1.,False)
        self.assertEqual(c.cached_parameters,returned);self.assertFalse(c.cached_parameters['auto'])


if __name__=='__main__':unittest.main()
