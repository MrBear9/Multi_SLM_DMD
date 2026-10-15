"""Explicit physical units and bounded acquisition/search settings."""
from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path
import warnings

MAX_GRID_SIZE = 4096
GRID_GROWTH_STEP = 64
GRID_EXTENT_TOLERANCE_UM = 1e-8


@dataclass
class CalibrationConfig:
    wavelength_nm: float = 532.0
    slm1_to_slm2_mm: float = 200.0
    slm2_to_camera_mm: float = 100.0
    dmd_width: int = 640
    dmd_height: int = 640
    dmd_pitch_um: float = 5.4
    # Hardware pitches mirror models/SLM/physical_defaults.py.
    # The hardware app must remain independent of the models/training package.
    slm1_pitch_um: float = 8.0
    slm2_pitch_um: float = 4.5
    camera_pitch_um: float = 4.0
    # Native-pixel sampling with padding for the default illuminated aperture.
    # fit_grids() still grows insufficient grids and preserves larger manual ones.
    slm1_grid_size: int = 704
    slm2_grid_size: int = 1152
    slm1_sampling_um: float = 8.0
    slm2_sampling_um: float = 4.5
    slm1_phase_resolution: int = 128
    slm2_phase_resolution: int = 128
    steps: int = 400
    learning_rate: float = 0.02
    momentum: float = 0.9
    seed: int = 17
    device: str = 'auto'
    gray_inverted: bool = True
    dmd_rotation: float = 0.0
    dmd_horizontal_flip: bool = False
    target_x_um: float = 0.0
    target_y_um: float = 0.0
    slm1_lut: str = ''
    slm2_lut: str = ''
    settle_seconds: float = 0.5
    average_frames: int = 5
    discard_frames: int = 3
    capture_timeout: float = 15.0
    offset_radius: int = 32
    angle_radius: float = 20.0
    search_rounds: int = 3
    coarse_orientation: bool = True
    minimum_score: float = 0.65
    minimum_coverage: float = 0.85
    max_registration_error: float = 2.0
    negative_control_margin: float = 0.04
    max_saturated_fraction: float = 0.02

    def validate(self, *, check_grid_coverage=True):
        positive = ('wavelength_nm', 'slm1_to_slm2_mm', 'slm2_to_camera_mm',
                    'dmd_width', 'dmd_height', 'dmd_pitch_um', 'slm1_pitch_um',
                    'slm2_pitch_um', 'camera_pitch_um', 'slm1_sampling_um', 'slm2_sampling_um', 'learning_rate',
                    'settle_seconds', 'capture_timeout', 'max_registration_error')
        for key in positive:
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} 必须为有限正数。')
        for key, low, high in (('slm1_grid_size', 32, MAX_GRID_SIZE), ('slm2_grid_size', 32, MAX_GRID_SIZE),
                               ('slm1_phase_resolution', 8, 1920), ('slm2_phase_resolution', 8, 1920),
                               ('steps', 1, 20000), ('average_frames', 1, 100),
                               ('discard_frames', 1, 100), ('offset_radius', 0, 512),
                               ('search_rounds', 1, 6), ('dmd_width', 8, 1920),
                               ('dmd_height', 8, 1080), ('seed', 0, 2**31 - 1)):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{key} 必须是 {low}～{high} 的整数。')
        for key in ('minimum_score', 'minimum_coverage', 'negative_control_margin',
                    'max_saturated_fraction', 'momentum'):
            if not 0 < getattr(self, key) < 1:
                raise ValueError(f'{key} 必须在 0 和 1 之间。')
        if not 0 <= self.angle_radius <= 45:
            raise ValueError('精细角度搜索范围应在 0～45°。')
        for key in ('dmd_rotation', 'target_x_um', 'target_y_um'):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f'{key} 必须为有限数值。')
        if check_grid_coverage:
            for key in ('slm1', 'slm2'):
                if self.grid_extent_um(key) + GRID_EXTENT_TOLERANCE_UM < 1.5 * max(self.aperture_um):
                    raise ValueError(f'{key.upper()} 数值计算视野过小，请增大该层网格，至少覆盖有效口径的 1.5 倍。')
            if self.grid_extent_um('slm2') > self.grid_extent_um('slm1') + GRID_EXTENT_TOLERANCE_UM:
                raise ValueError('SLM1 传播后的可用计算视野必须覆盖 SLM2 网格；请增大 SLM1 网格，不能补出未计算的复光场。')
        if self.device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device 应为 auto / cpu / cuda。')
        return self

    def fit_grids(self):
        """Grow undersized grids, leaving every already-sufficient size intact.

        Only development setup calls this. Saved phase bundles must retain the
        exact coordinates on which their reference arrays were trained.
        """
        self.validate(check_grid_coverage=False)
        minimum_extent = 1.5 * max(self.aperture_um)

        def grow(key, extent):
            current, pitch = self.grid_size_for(key), self.sampling_um_for(key)
            if current * pitch + GRID_EXTENT_TOLERANCE_UM >= extent:
                return current
            required = math.ceil((extent - GRID_EXTENT_TOLERANCE_UM) / pitch)
            size = math.ceil(required / GRID_GROWTH_STEP) * GRID_GROWTH_STEP
            if size > MAX_GRID_SIZE:
                raise ValueError(f'{key.upper()} 至少需要 {required} 个网格点，超过当前边长上限 {MAX_GRID_SIZE}；'
                                 '请增大该层数值采样间距，或减小标定区域/SLM2 计算范围。')
            return size

        slm2_size = grow('slm2', minimum_extent)
        slm1_size = grow('slm1', max(minimum_extent, slm2_size * self.slm2_sampling_um))
        return replace(self, slm1_grid_size=slm1_size, slm2_grid_size=slm2_size).validate()

    @property
    def aperture_um(self):
        return self.dmd_height * self.dmd_pitch_um, self.dmd_width * self.dmd_pitch_um

    def slm_shape(self, key):
        pitch = getattr(self, key + '_pitch_um')
        return tuple(max(1, round(value / pitch)) for value in self.aperture_um)

    def grid_size_for(self, key):
        return getattr(self, key + '_grid_size')

    def sampling_um_for(self, key):
        return getattr(self, key + '_sampling_um')

    def grid_extent_um(self, key):
        return self.grid_size_for(key) * self.sampling_um_for(key)

    def phase_shape(self, key):
        n = getattr(self, key + '_phase_resolution')
        return n, n

    @property
    def output_grid_size(self):
        return self.slm2_grid_size

    @property
    def output_sampling_um(self):
        return self.slm2_sampling_um

    def grid_metadata(self):
        return {key: {'asm_shape_hw': [self.grid_size_for(key)] * 2,
                      'asm_sampling_um': self.sampling_um_for(key),
                      'asm_extent_um': self.grid_extent_um(key),
                      'hardware_pitch_um': getattr(self, key + '_pitch_um'),
                      'export_shape_hw': list(self.slm_shape(key)),
                      'phase_parameter_shape_hw': list(self.phase_shape(key))}
                for key in ('slm1', 'slm2')}

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path, on_migration=None, *, check_grid_coverage=True):
        values = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(values, dict):
            raise ValueError('标定参数必须是 JSON 对象。')
        migrated = []
        for old in ('grid_size', 'sampling_um', 'phase_resolution'):
            if old not in values:
                continue
            value = values.pop(old)
            for key in ('slm1_' + old, 'slm2_' + old):
                if key in values and values[key] != value:
                    raise ValueError(f'旧字段 {old} 与新字段 {key} 冲突，请明确选择参数。')
                values[key] = value
            migrated.append(old)
        if migrated:
            message = ('载入了旧版公共网格参数，已按原值分别填入 SLM1/SLM2，未改成新默认值。'
                       '若要采用与硬件像素间距一致的 8 μm / 4.5 μm 默认采样，请选择“恢复默认”并重新训练。')
            if on_migration:
                on_migration(message)
            else:
                warnings.warn(message, UserWarning, stacklevel=2)
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f'未知标定字段：{sorted(unknown)}')
        # LUT paths in a saved configuration are relative to that file.
        for key in ('slm1_lut', 'slm2_lut'):
            if values.get(key):
                values[key] = str((Path(path).resolve().parent / values[key]).resolve())
        return cls(**values).validate(check_grid_coverage=check_grid_coverage)


def write_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

