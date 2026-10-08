"""DMD choices share a preview while keeping independent hardware settings."""
from dataclasses import dataclass

from .displays import DisplayAdapter


@dataclass(frozen=True)
class DmdBackend:
    name: str
    enabled_field: str
    input_field: str
    title: str


DMD_BACKENDS = (
    DmdBackend('desktop', 'dmdEnabledCheckBox', 'dmdInputPathEdit', 'DMD · 显示器输出'),
    DmdBackend('f4321', 'f4321EnabledCheckBox', 'f4321InputPathEdit', 'DMD · F4321 网络控制器'),
)


def selected_dmd(values):
    selected = [backend for backend in DMD_BACKENDS if values.get(backend.enabled_field, False)]
    if len(selected) > 1:
        raise ValueError('请只启用一种 DMD 硬件。')
    return selected[0] if selected else None


def create_display(key, settings, paths, demo=False):
    if key == 'dmd' and settings.get('f4321EnabledCheckBox', False):
        from .f4321_display import F4321DisplayAdapter
        return F4321DisplayAdapter(settings, paths, demo)
    return DisplayAdapter(key, settings, paths, demo)
