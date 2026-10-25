"""External-process orchestration; hardware remains owned by the main GUI."""
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np
from PIL import Image
from PyQt5 import QtCore

from ..holography.config import HolographyConfig
from .calibration_controller import CalibrationController, CalibrationTask
from .configuration import read_values


class HolographyCaptureTask(CalibrationTask):
    def __init__(self, config, root, settings, payload, parent):
        super().__init__(config, root, settings, False, parent)
        self.payload = payload

    def run(self):
        try:
            # Read back SDK state, fail if auto exposure is still enabled.
            parameters = self.io('parameters', (self.settings['ccdExposureSpinBox'], self.settings['ccdGainSpinBox'], False))
            if parameters.get('auto'):
                raise ValueError('相机没有进入固定曝光。')
            self.io('display', (self.payload['paths'], self.settings))
            image, metadata, _ = self.io('capture', self.config)
            path = Path(self.payload['capture_path'])
            np.save(path, image)
            path.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            self.result = {'path': str(path), 'exposure_us':parameters.get('exposure')}
        except Exception as exc:
            self.error = str(exc)


class HolographyController(QtCore.QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.dialog = None
        self.process = None
        self.capture_task = None
        self.bridge = CalibrationController(window)
        self.root = self.result_path = None
        self.config = None
        self.original = None
        self.buffer = b''
        self.stopping = self.disconnect_after = False
        self.process_done = False
        self.exit_code = None
        self.worker_error = ''
        self.received_result = False
        self.geometry_result = None
        self.point_selection = None

    @property
    def active(self):
        return self.process is not None or self.capture_task is not None

    @property
    def hardware_active(self):
        return self.active and self.config is not None and self.config.live

    def ensure_dialog(self):
        from .holography_window import HolographyWindow
        if self.dialog is None:
            self.dialog = HolographyWindow(self)
        return self.dialog

    def phase_test_config(self, key, probe):
        if key not in ('slm1','slm2'):
            raise ValueError('全息探测相位仅用于 SLM。')
        # Read current edits, including unsaved parameters, without changing the
        # selected mode/operation or requiring a target image and training Python.
        return replace(self.ensure_dialog().values(),mode=key,geometry_probe=probe)

    def open(self):
        self.ensure_dialog()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def log(self, message):
        if self.dialog is not None:
            self.dialog.log.appendPlainText(message)

    def start(self, operation=None):
        if self.active or self.window.calibration.active:
            raise ValueError('请先完成当前训练或自动标定。')
        w, ui = self.window, self.dialog
        config = ui.values().resolve_target_size()
        if operation == 'geometry':
            config.operation = 'geometry'
            config.geometry_profile = {}
            config.magnification_source = 'manual'
        if config.operation == 'geometry':
            from ..holography.geometry_limits import validate_probe_bandwidth
            if config.geometry_method=='probes':
                validate_probe_bandwidth(config)
            else:
                from ..holography.checkerboard import board
                board(config)
            if config.geometry_search_roi:
                from ..holography.spot_detection import search_rect
                hardware=self.hardware_signature()
                size=hardware.get('camera',{}).get('geometry',{}).get('raw_size',[])
                if len(size)!=2:
                    raise ValueError('请连接相机并重新读取第三页定位搜索 ROI。')
                search_rect(config.geometry_search_roi,size[::-1],hardware)
        if config.operation=='geometry' and config.geometry_method=='checkerboard':
            config.validate()
        else:
            config.fit_grid().validate()
        ui.grid_width.setValue(config.grid_width)
        ui.grid_height.setValue(config.grid_height)
        python = Path(ui.pythonPath.text().strip())
        worker = Path(ui.workerPath.text().strip())
        if not python.is_file() or not worker.is_file():
            raise ValueError('请选择外部开发环境的 Python 和 tools/holography/worker.py。')
        if config.target and config.target_kind == 'file':
            config.target = str(Path(config.target).resolve())
        if config.model_path:
            config.model_path = str(Path(config.model_path).resolve())
        if config.live:
            if w.demo:
                raise ValueError('演示相机不能进行真实回环或模型标定；可以使用离线全息生成。')
            if not w.camera_ready or w.camera is None or not w.camera.isRunning():
                raise ValueError('请先连接相机并等待采集就绪。')
            if w.disconnecting or w.device_task is not None:
                raise ValueError('请等待设备操作完成。')
            if set(config.active_slms)-set(w.devices):
                raise ValueError('请连接所选 SLM；单层模式不要求另一片连接。')
            if (w.camera_parameters.get('auto', True) and not config.stage_exposure_enabled
                    and not (config.operation=='geometry' and config.geometry_auto_exposure)):
                raise ValueError('请在主界面关闭自动曝光并应用参数。')
            self.validate_exposure(config.exposure_for(w.camera_parameters['exposure']))
            if w.network_dmd() is not None:
                raise ValueError('首版全息回环请使用固定照明或普通 DMD，暂不管理 F4321。')
            for key in config.active_slms:
                display = w.devices[key].display
                h, width = config.shape(key)
                if h > display.height or width > display.width:
                    raise ValueError(key+' 有效相位尺寸超过当前面板。')
            if config.geometry_profile:
                from ..holography.geometry import validate_profile
                validate_profile(config.geometry_profile, config, hardware=self.hardware_signature())
        if config.live:
            w.stop_all()
        self.config = config
        self.original = read_values(w) if config.live else None
        if self.original is not None:
            # Preserve SDK precision instead of rewriting rounded UI values.
            self.original['ccdExposureSpinBox'] = w.camera_parameters['exposure']
            self.original['ccdGainSpinBox'] = w.camera_parameters['gain']
            self.original['ccdAutoExposureCheckBox'] = w.camera_parameters.get('auto',False)
        root = Path(ui.outputPath.text().strip()).resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self.result_path = None
        self.geometry_result = None
        hardware = self.hardware_signature(config.exposure_for(self.original['ccdExposureSpinBox'])) if config.live else None
        job = {'config': asdict(config), 'output': str(root), 'hardware': hardware}
        job_path = root/'job.json'
        job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
        if config.live:
            self.bridge.directory = tempfile.TemporaryDirectory(prefix='holography_display_')
        self.stopping = self.process_done = self.received_result = False
        self.worker_error = ''
        self.exit_code = None
        self.buffer = b''
        self.process = QtCore.QProcess(self)
        self.process.setWorkingDirectory(str(worker.parent))
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONIOENCODING','utf-8')
        environment.insert('PYTHONUNBUFFERED','1')
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_error)
        self.process.finished.connect(self.finished_process)
        self.process.errorOccurred.connect(self.process_error)
        self.process.start(str(python), ['-u', str(worker), str(job_path)])
        ui.preferences.setValue('python',str(python))
        ui.preferences.setValue('worker',str(worker))
        ui.preferences.setValue('output',ui.outputPath.text())
        ui.progress.setValue(0)
        ui.update_controls()
        w.update_actions()
        self.log('开始；结果目录：'+str(root))
        if config.live:
            self.log(f'本次固定曝光：{config.exposure_for(self.original["ccdExposureSpinBox"]):g} μs；任务结束恢复原曝光。')

    def validate_exposure(self, exposure):
        limits=self.window.camera_parameters.get('exposure_range')
        if limits and not limits[0] <= exposure <= limits[1]:
            raise ValueError(f'曝光 {exposure:g} μs 超出当前相机范围 {limits[0]:g}～{limits[1]:g} μs。')

    def apply_preview_exposure(self):
        w=self.window
        if self.active or w.calibration.active or w.automation_active:
            raise ValueError('请等待当前硬件任务结束。')
        if not w.camera_ready or w.camera is None:
            raise ValueError('请先连接相机。')
        exposure=self.dialog.reconstruction_exposure_us.value()
        self.validate_exposure(exposure)
        w.camera.submit('parameters',exposure,w.camera_parameters['gain'],False)
        self.log(f'已请求重建预览曝光 {exposure:g} μs，并关闭自动曝光。')

    def hardware_signature(self, exposure=None):
        w = self.window
        values = read_values(w)
        names = [name for name in values if (name.startswith(('slm1','slm2','dmd','ccd','toupcam'))
                 and any(token in name for token in ('Rotation','Flip','Offset','Background','Exposure','Gain','Roi','Binning','Resolution','Format','Display','Monitor')))]
        identity = {'settings': {key:values[key] for key in names}, 'camera':w.camera_parameters,
                    'dmd_input':values.get('dmdInputPathEdit',''), 'dmd_sha256':None,
                    'displays':{key:device.display._asdict() for key,device in w.devices.items()
                                if hasattr(getattr(device,'display',None),'_asdict')}}
        if 'dmd' in w.devices:
            # DMD current image and test-pattern state affect illumination.
            qimage = w.images.get('dmd')
            if qimage is not None:
                bits = qimage.bits()
                bits.setsize(qimage.byteCount())
                identity['dmd_sha256'] = hashlib.sha256(bytes(bits)).hexdigest()
        # SDK ranges/ROIs may be tuples; files and worker RPC use JSON lists.
        # Compare one canonical representation before and after a round trip.
        identity=json.loads(json.dumps(identity, ensure_ascii=False))
        if exposure is not None:
            identity['settings']['ccdExposureSpinBox']=exposure
            identity['settings']['ccdAutoExposureCheckBox']=False
            identity['camera']['exposure']=exposure
            identity['camera']['auto']=False
        return identity

    def safe_path(self, value):
        path = Path(value).resolve()
        if self.root not in path.parents:
            raise ValueError('训练进程返回了结果目录之外的路径。')
        return path

    def read_output(self):
        if self.process is None:
            return
        self.buffer += bytes(self.process.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer = self.buffer.split(b'\n',1)
            if not line.strip():
                continue
            try:
                message = json.loads(line.decode('utf-8'))
                self.handle(message)
            except Exception as exc:
                self.worker_error = '训练通信错误：'+str(exc)
                self.log(self.worker_error)
                self.cancel()
                break

    def read_error(self):
        if self.process is not None:
            message = bytes(self.process.readAllStandardError()).decode('utf-8',errors='replace').strip()
            if message:
                self.log(message)

    def handle(self, message):
        kind = message.get('type')
        if kind in ('log','error','progress','result'):
            self.log(message.get('message',''))
        if kind == 'error':
            self.worker_error = message.get('message','训练失败')
        elif kind == 'progress':
            self.dialog.progress.setValue(int(message['value']))
        elif kind == 'geometry_exposure_locked':
            if self.stopping or self.process_done:
                return
            if not self.hardware_active or self.config.operation != 'geometry' or not self.config.geometry_auto_exposure:
                raise ValueError('当前任务不允许更新自动标定曝光。')
            exposure = float(message['exposure_us'])
            field = self.dialog.geometry_exposure_us
            if not np.isfinite(exposure) or not field.minimum() <= exposure <= field.maximum():
                raise ValueError('自动标定返回了无效的锁定曝光值。')
            self.validate_exposure(exposure)
            field.setValue(exposure)
            self.log(f'已将几何标定曝光更新为 {exposure:g} μs，可随全息参数保存。')
        elif kind == 'preview':
            if message['kind'] not in ('target','simulation','camera','camera_raw'):
                raise ValueError('未知预览类型。')
            self.dialog.preview(message['kind'],self.safe_path(message['path']))
        elif kind == 'result':
            path = self.safe_path(message['path'])
            self.received_result = True
            if path.name == 'result.json':
                self.result_path = path
            elif path.name == 'geometry.json':
                self.geometry_result = path
        elif kind == 'capture':
            if self.stopping:
                return
            if not self.hardware_active or self.capture_task is not None:
                raise ValueError('当前任务不允许采集或上一事务尚未完成。')
            if self.point_selection is not None and self.point_selection.pending is not None:
                raise ValueError('请先完成当前定位图的框选。')
            paths = message['paths']
            if set(paths) != set(self.config.active_slms):
                raise ValueError('训练请求与所选 SLM 不一致。')
            paths = {key:str(self.safe_path(path)) for key,path in paths.items()}
            for key,path in paths.items():
                with Image.open(path) as image:
                    if image.mode != 'L' or image.size != self.config.shape(key)[::-1]:
                        raise ValueError('训练相位的尺寸或格式无效。')
            # A connected inactive SLM remains physically in the optical path;
            # set its entire panel to grey zero, then restore it at completion.
            for key in ('slm1','slm2'):
                if key in self.window.devices and key not in paths:
                    display = self.window.devices[key].display
                    path = self.root/(key+'_fixed_black.png')
                    if not path.exists():
                        Image.new('L',(display.width,display.height),0).save(path)
                    paths[key] = str(path)
            settings = dict(self.original)
            settings['ccdExposureSpinBox']=self.config.exposure_for(settings['ccdExposureSpinBox'])
            if 'exposure_us' in message:
                if self.config.operation != 'geometry':
                    raise ValueError('仅几何标定允许辅助重拍曝光。')
                exposure = float(message['exposure_us'])
                if not np.isfinite(exposure) or exposure <= 0:
                    raise ValueError('重拍曝光无效。')
                self.validate_exposure(exposure)
                settings['ccdExposureSpinBox'] = exposure
            settings['ccdAutoExposureCheckBox']=False
            for key in paths:
                if key not in self.config.active_slms:
                    settings.update({key+'RotationSpinBox':0, key+'HorizontalFlipCheckBox':False,
                                     key+'VerticalFlipCheckBox':False,key+'OffsetXSpinBox':0,key+'OffsetYSpinBox':0})
            payload = {'paths':paths,'capture_path':str(self.safe_path(message['capture_path']))}
            task = HolographyCaptureTask(self.config,self.root,settings,payload,self)
            self.capture_task = self.bridge.task = task
            task.requested.connect(self.bridge.dispatch)
            task.finished.connect(self.finished_capture)
            task.start()
        elif kind == 'geometry_select':
            if self.stopping:
                return
            if self.point_selection is None:
                from .geometry_point_selection import GeometryPointSelection
                self.point_selection = GeometryPointSelection(self)
            self.point_selection.begin(message)

    def finished_capture(self):
        task = self.capture_task
        if task is None:
            return
        response = {'error':task.error} if task.error else task.result
        self.capture_task = self.bridge.task = None
        task.deleteLater()
        if not self.stopping and self.process is not None and not self.process_done:
            self.process.write((json.dumps(response,ensure_ascii=False)+'\n').encode('utf-8'))
        if self.process_done:
            self.cleanup()

    def process_error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.worker_error = '无法启动外部 Python：'+self.process.errorString()
            self.process_done = True
            self.cleanup()

    def finished_process(self, code, status):
        self.read_output()
        self.read_error()
        self.exit_code = code
        self.process_done = True
        if self.capture_task is not None:
            self.capture_task.requestInterruption()
        else:
            self.cleanup()

    def cancel(self):
        if not self.active:
            return
        self.stopping = True
        if self.point_selection is not None:
            self.point_selection.finish()
        if self.capture_task is not None:
            self.capture_task.requestInterruption()
        if self.hardware_active and self.window.camera is not None:
            self.window.camera.submit('cancel_calibration')
        if self.process is not None and not self.process_done:
            self.process.terminate()
            process = self.process
            QtCore.QTimer.singleShot(2000, lambda: self.kill_if_current(process))

    def kill_if_current(self, process):
        if self.process is process and process.state()!=QtCore.QProcess.NotRunning:
            process.kill()

    def cleanup(self):
        if self.capture_task is not None:
            return
        if self.point_selection is not None:
            self.point_selection.finish()
        success = self.exit_code==0 and self.received_result and not self.worker_error and not self.stopping
        had_hardware = self.original is not None
        if had_hardware:
            try:
                self.bridge.restore(self.original)
                if self.window.camera is not None:
                    self.window.camera.submit('parameters',self.original['ccdExposureSpinBox'],
                                              self.original['ccdGainSpinBox'],self.original['ccdAutoExposureCheckBox'])
            except Exception as exc:
                self.worker_error += '；恢复显示失败：'+str(exc)
                success = False
        if self.bridge.directory is not None:
            try:
                self.bridge.directory.cleanup()
            except OSError as exc:
                self.log('临时显示文件仍被 SDK 占用：'+str(exc))
            self.bridge.directory = None
        process,self.process = self.process,None
        if process is not None:
            process.deleteLater()
        self.original = None
        if success and self.geometry_result is not None:
            try:
                self.dialog.settings.show_profile(json.loads(self.geometry_result.read_text(encoding='utf-8')))
            except Exception as exc:
                self.worker_error = '几何结果读取失败：'+str(exc)
                success = False
        state = 'completed' if success else 'cancelled' if self.stopping else 'failed'
        if self.root is not None:
            try:
                (self.root/'status.json').write_text(json.dumps({'status':state,'error':self.worker_error},ensure_ascii=False,indent=2),encoding='utf-8')
            except OSError as exc:
                self.log('状态保存失败：'+str(exc))
        self.log(('完成。'+('已恢复原显示。' if had_hardware else '') if success else '任务已停止：'+(self.worker_error or state)))
        self.dialog.progress.setValue(100 if success else 0)
        self.dialog.update_controls()
        self.window.update_actions()
        if self.disconnect_after or self.window.closing:
            self.disconnect_after = False
            self.window.disconnect_devices()
            if self.window.closing and self.window.camera is None:
                self.window.close()

    def validate_result(self):
        data = json.loads(self.result_path.read_text(encoding='utf-8'))
        if data.get('schema') != 1:
            raise ValueError('未知全息结果格式。')
        config = HolographyConfig.from_dict(data['config']).validate()
        if set(data['paths']) != set(config.active_slms):
            raise ValueError('结果缺少所选 SLM 的相位。')
        paths = {}
        for key in config.active_slms:
            # Portable result directory: resolve known relative outputs, not old absolute paths.
            path = self.result_path.parent/key/'phase.png'
            with Image.open(path) as image:
                if image.mode!='L' or image.size!=config.shape(key)[::-1]:
                    raise ValueError('导出的相位尺寸或位深不匹配。')
            paths[key] = str(path)
        return paths

    def apply_result(self):
        if self.active:
            raise ValueError('请等待任务结束。')
        paths = self.validate_result()
        w = self.window
        # Import exactly the selected SLM(s); retain the user's playback pairing preference.
        paired = w.synchronizeSlmsCheckBox.isChecked()
        with QtCore.QSignalBlocker(w.synchronizeSlmsCheckBox):
            w.synchronizeSlmsCheckBox.setChecked(len(paths)==2)
        try:
            w.inputs.import_inputs(paths)
        finally:
            with QtCore.QSignalBlocker(w.synchronizeSlmsCheckBox):
                w.synchronizeSlmsCheckBox.setChecked(paired)
            w.update_actions()
        self.log('相位已导入；已连接的所选 SLM 已显示。')
        if self.dialog.values().stage_exposure_enabled and w.camera_ready:
            self.apply_preview_exposure()
