"""Serializable job description, shared without importing the training runtime."""
from dataclasses import asdict, dataclass, field
import math


@dataclass
class HolographyConfig:
    mode: str = 'slm1'
    operation: str = 'generate'
    target: str = ''
    target_kind: str = 'file'
    model_path: str = ''
    wavelength_nm: float = 532.
    slm1_width: int = 432
    slm1_height: int = 432
    slm1_pitch_um: float = 8.
    slm2_width: int = 768
    slm2_height: int = 768
    slm2_pitch_um: float = 4.5
    slm1_distance_mm: float = 300.
    slm2_distance_mm: float = 100.
    inter_slm_mm: float = 200.
    grid_width: int = 1152
    grid_height: int = 1152
    grid_margin: float = 1.5
    grid_limit: int = 4096
    sampling_um: float = 4.5
    initialization: str = 'random'
    init_range_rad: float = 3.141593  # pi at the UI's six-decimal precision
    gs_iterations: int = 30
    slm1_controls_width: int = 432
    slm1_controls_height: int = 432
    slm2_controls_width: int = 768
    slm2_controls_height: int = 768
    steps: int = 200
    learning_rate: float = .03
    samples: int = 32
    epochs: int = 100
    output_width: int = 256
    output_height: int = 256
    target_width_um: float = 1000.
    target_height_um: float = 1000.
    target_fit: str = 'contain'
    reference_source: str = 'dmd'
    sizing_mode: str = 'manual'
    reference_width: float = 640.
    reference_height: float = 640.
    reference_pitch_um: float = 5.4
    reference_physical_width_um: float = 3456.
    reference_physical_height_um: float = 3456.
    reconstruction_scale: float = 1.
    camera_optical_path: str = 'direct'
    camera_profile: str = 'connected'
    sensor_pitch_x_um: float = 3.76
    sensor_pitch_y_um: float = 3.76
    binning_x: float = 1.
    binning_y: float = 1.
    relay_magnification: float = 1.
    magnification_source: str = 'manual'
    geometry_mapping: str = 'affine'
    geometry_pattern: str = 'sequential'
    geometry_tolerance: float = 2.
    geometry_spot_max_fwhm: float = 0.
    geometry_spot_stability: float = 1.
    geometry_repeats: int = 2
    geometry_assistance: str = 'automatic'
    geometry_detector: str = 'spot'
    geometry_probe: str = 'focus'
    geometry_auto_exposure: bool = False
    geometry_exposure_target: float = .65
    geometry_movement_check: bool = True
    geometry_ring_radius: float = 100.
    geometry_probe_radius_um: float = 300.
    geometry_axicon_period_um: float = 85.
    geometry_search_roi: dict = field(default_factory=dict)
    geometry_profile: dict = field(default_factory=dict)
    camera_pitch_um: float = 3.76
    camera_center_x: float = 0.
    camera_center_y: float = 0.
    camera_angle_deg: float = 0.
    camera_flip_x: bool = False
    camera_flip_y: bool = False
    camera_dark: float = 0.
    camera_white: float = 255.
    stage_exposure_enabled: bool = False
    geometry_exposure_us: float = 150.
    reconstruction_exposure_us: float = 10000.
    mapping_confirmed: bool = False
    settle_seconds: float = .5
    average_frames: int = 3
    discard_frames: int = 3
    capture_timeout: float = 20.
    feedback_every: int = 5
    gray_inverted: bool = True
    seed: int = 17

    @classmethod
    def from_dict(cls, values):
        data = dict(values)
        if 'output_size' in data:
            side = data.pop('output_size')
            data.setdefault('output_width', side)
            data.setdefault('output_height', side)
        if 'grid_size' in data:
            side = data.pop('grid_size')
            data.setdefault('grid_width', side)
            data.setdefault('grid_height', side)
        # Old files used one square grid for both SLMs. Keep their meaning when
        # loading results/configs; newly saved files contain independent axes.
        if 'controls' in data:
            side = data.pop('controls')
            for slm in ('slm1', 'slm2'):
                for axis in ('width', 'height'):
                    data.setdefault(slm + '_controls_' + axis, side)
        return cls(**data)

    @property
    def active_slms(self):
        return ('slm1', 'slm2') if self.mode == 'dual' else (self.mode,)

    @property
    def live(self):
        return self.operation in ('feedback', 'calibrate', 'geometry')

    def reference_size(self):
        if self.reference_source == 'custom':
            return self.reference_physical_width_um, self.reference_physical_height_um
        if self.reference_source == 'slm':
            key = self.active_slms[0]
            h, w = self.shape(key)
            pitch = getattr(self, key + '_pitch_um')
            return w * pitch, h * pitch
        return self.reference_width * self.reference_pitch_um, self.reference_height * self.reference_pitch_um

    def resolve_target_size(self):
        if self.sizing_mode == 'reference':
            w, h = self.reference_size()
            self.target_width_um, self.target_height_um = w*self.reconstruction_scale, h*self.reconstruction_scale
        return self

    def exposure_for(self, current):
        if not self.stage_exposure_enabled:
            return current
        return self.geometry_exposure_us if self.operation == 'geometry' else self.reconstruction_exposure_us

    def shape(self, key):
        return getattr(self, key + '_height'), getattr(self, key + '_width')

    def control_shape(self, key):
        return getattr(self, key + '_controls_height'), getattr(self, key + '_controls_width')

    @property
    def grid_shape(self):
        return self.grid_height, self.grid_width

    def required_grid_extent(self, axis):
        extent = max(getattr(self, k + '_' + axis) * getattr(self, k + '_pitch_um')
                     for k in self.active_slms)
        return max(self.grid_margin * extent, getattr(self, 'target_' + axis + '_um'))

    def validate_grid_policy(self):
        if not math.isfinite(self.grid_margin) or not 1 <= self.grid_margin <= 8:
            raise ValueError('计算视野预留倍率需要在 1～8 之间。')
        if type(self.grid_limit) is not int or not 32 <= self.grid_limit <= 16384:
            raise ValueError('网格单轴上限需要整数 32～16384。')

    def grid_report(self):
        """Allocation-free estimate; one FFT array is NOT total training memory."""
        self.validate_grid_policy()
        if not math.isfinite(self.sampling_um) or self.sampling_um <= 0:
            raise ValueError('sampling_um 必须是有限正数。')
        sizes = []
        for axis in ('width', 'height'):
            current = getattr(self, 'grid_' + axis)
            required = self.required_grid_extent(axis) / self.sampling_um
            sizes.append(current if current >= required else math.ceil(required / 64) * 64)
        w, h = sizes
        return {'width': w, 'height': h, 'fits': max(w, h) <= self.grid_limit,
                'fft_array_mib': (2*w)*(2*h)*8/1024**2}

    def fit_grid(self):
        self.resolve_target_size()
        report = self.grid_report()
        if not report['fits']:
            raise ValueError(f"当前参数需要 ASM 网格 {report['width']}×{report['height']}，"
                             f'超过所设单轴上限 {self.grid_limit}。请调整采样、预留倍率或有效区域，'
                             '也可在资源允许时提高网格上限；硬件像素间距应保持真实值。')
        self.grid_width, self.grid_height = report['width'], report['height']
        return self

    def validate(self):
        self.resolve_target_size()
        self.validate_grid_policy()
        if self.mode not in ('slm1', 'slm2', 'dual') or self.operation not in ('generate', 'feedback', 'calibrate', 'geometry'):
            raise ValueError('未知全息模式或操作。')
        for name, choices in {
            'initialization': ('random', 'gs'),
            'target_kind': ('file', 'cross', 'frame', 'asymmetric'),
            'target_fit': ('contain', 'cover', 'stretch'),
            'reference_source': ('slm', 'dmd', 'custom'), 'sizing_mode': ('reference', 'manual'),
            'camera_optical_path': ('direct', 'relay', 'unsupported'),
            'camera_profile': ('connected', 'toup', 'dvp'),
            'magnification_source': ('manual', 'calibrated'),
            'geometry_mapping': ('similarity', 'affine', 'perspective'),
            'geometry_pattern': ('sequential', 'asymmetric'),
            'geometry_assistance': ('automatic', 'manual', 'on_failure'),
            'geometry_detector': ('spot', 'ring'),
            'geometry_probe': ('focus', 'disk', 'axicon'),
        }.items():
            if getattr(self, name) not in choices:
                raise ValueError('无效选项：' + name)
        if self.camera_optical_path == 'unsupported':
            raise ValueError('透镜参与全息传播需要相应传播模型；当前仅支持自由传播及目标平面的中继成像。')
        if self.operation == 'geometry' and self.mode == 'dual':
            raise ValueError('此入口标定单 SLM → 目标平面 → 相机；请选择 SLM1 或 SLM2。')
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(key + ' 必须是有限数值。')
        for key in ('wavelength_nm', 'slm1_pitch_um', 'slm2_pitch_um', 'slm1_distance_mm',
                    'slm2_distance_mm', 'inter_slm_mm', 'sampling_um', 'learning_rate',
                    'target_width_um', 'target_height_um', 'camera_pitch_um', 'capture_timeout',
                    'reference_width', 'reference_height', 'reference_pitch_um',
                    'reference_physical_width_um', 'reference_physical_height_um', 'reconstruction_scale',
                    'sensor_pitch_x_um', 'sensor_pitch_y_um', 'binning_x', 'binning_y',
                    'relay_magnification', 'geometry_tolerance', 'geometry_spot_stability', 'geometry_exposure_us', 'reconstruction_exposure_us'):
            if getattr(self, key) <= 0:
                raise ValueError(key + ' 必须大于零。')
        if not 0 <= self.init_range_rad <= math.pi + 1e-6:
            raise ValueError('随机相位半范围需要在 0～π rad 之间；0 表示平相位起点。')
        if not 0 <= self.geometry_spot_max_fwhm <= 512 or not isinstance(self.geometry_search_roi,dict):
            raise ValueError('定位光斑最大半高全宽需为 0～512 px，搜索 ROI 必须是有效标注。')
        if not .05 <= self.geometry_spot_stability <= 10:
            raise ValueError('定位中心稳定性限值需为 0.05～10 px。')
        if not .2 <= self.geometry_exposure_target <= .85:
            raise ValueError('自动曝光目标需要在满量程的 20%～85% 之间。')
        if not 12 <= self.geometry_ring_radius <= 1024:
            raise ValueError('圆环分析半径需要在 12～1024 原始相机像素之间。')
        if self.geometry_probe_radius_um <= 0 or self.geometry_axicon_period_um <= 0:
            raise ValueError('圆形相位区半径和轴锥周期必须大于零。')
        if self.geometry_probe != 'focus' and self.geometry_detector != 'ring':
            raise ValueError('圆形相位区与轴锥探测图案请使用圆环中心检测。')
        if self.geometry_probe=='disk' and self.mode!='dual':
            key=self.active_slms[0]
            radius_limit=(min(self.shape(key))-5)*getattr(self,key+'_pitch_um')/2
            if self.geometry_probe_radius_um>=radius_limit:
                raise ValueError(f'圆形相位区半径应小于 {radius_limit:g} μm，以便边缘完整位于有效区域内。')
        for key, low, high in (('grid_width', 32, self.grid_limit), ('grid_height', 32, self.grid_limit),
                               ('gs_iterations', 1, 1000),
                               ('geometry_repeats', 1, 5),
                               ('steps', 1, 20000), ('samples', 8, 1000), ('epochs', 1, 10000),
                               ('output_width', 16, 4096), ('output_height', 16, 4096), ('average_frames', 1, 100),
                               ('discard_frames', 0, 100), ('feedback_every', 1, 1000)):
            v = getattr(self, key)
            if type(v) is not int or not low <= v <= high:
                raise ValueError(f'{key} 需要整数 {low}～{high}。')
        for key in self.active_slms:
            ch, cw = self.control_shape(key)
            if type(ch) is not int or type(cw) is not int or not (8 <= ch <= 1920 and 8 <= cw <= 1920):
                raise ValueError(key + ' 控制点宽、高需要整数 8～1920。')
            h, w = self.shape(key)
            if type(h) is not int or type(w) is not int or not (8 <= h <= 1080 and 8 <= w <= 1920):
                raise ValueError(key + ' 有效区域超出 1920×1080 面板。')
        if self.settle_seconds < 0 or not 0 <= self.camera_dark < self.camera_white:
            raise ValueError('稳定时间或相机黑白电平无效。')
        for axis, label in (('width', '宽度'), ('height', '高度')):
            if getattr(self, 'grid_' + axis) * self.sampling_um < self.required_grid_extent(axis):
                raise ValueError('ASM 网格' + label + '视野不足，请增大对应网格。')
        if self.sampling_um > min(getattr(self, k + '_pitch_um') for k in self.active_slms):
            raise ValueError('首版全息模型要求数值采样不粗于所用 SLM 像素间距。')
        if self.geometry_profile:
            from .geometry import validate_profile
            validate_profile(self.geometry_profile, self)
        if self.magnification_source == 'calibrated' and not self.geometry_profile:
            raise ValueError('请先载入并应用通过验证的几何标定。')
        if self.live and self.operation != 'geometry' and not self.mapping_confirmed:
            raise ValueError('请确认相机坐标映射、线性输出、固定曝光及黑白电平。')
        if self.operation not in ('calibrate', 'geometry') and self.target_kind == 'file' and not self.target:
            raise ValueError('请选择目标强度图。')
        return self

    def optical_signature(self):
        names = ('mode', 'wavelength_nm', 'grid_width', 'grid_height', 'sampling_um', 'gray_inverted',
                 'slm1_distance_mm', 'slm2_distance_mm', 'inter_slm_mm', 'output_width', 'output_height',
                 'target_width_um', 'target_height_um')
        result = {key: getattr(self, key) for key in names}
        for slm in self.active_slms:
            for suffix in ('width', 'height', 'pitch_um'):
                name = slm + '_' + suffix
                result[name] = getattr(self, name)
        return result

    def matches_optical_signature(self, signature):
        if not isinstance(signature, dict):
            return False
        signature = dict(signature)
        if 'output_size' in signature:
            side = signature.pop('output_size')
            signature.setdefault('output_width', side)
            signature.setdefault('output_height', side)
        if 'grid_size' in signature:
            side = signature.pop('grid_size')
            signature.setdefault('grid_width', side)
            signature.setdefault('grid_height', side)
        return signature == self.optical_signature()

    def observation_signature(self):
        return {name:getattr(self,name) for name in (
            'camera_pitch_um','camera_center_x','camera_center_y','camera_angle_deg',
            'camera_flip_x','camera_flip_y','camera_dark','camera_white',
            'camera_optical_path','sensor_pitch_x_um','sensor_pitch_y_um',
            'binning_x','binning_y','relay_magnification','geometry_profile')}
