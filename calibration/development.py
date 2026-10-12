"""Explicit development-only entry points, outside the packaged hardware flow."""
import importlib
import sys


def training_module():
    if getattr(sys, 'frozen', False):
        raise RuntimeError('打包版不提供训练，请在开发环境训练后导入标定包。')
    try:
        module = importlib.import_module('tools.calibration.train')
        importlib.import_module('torch')
        return module
    except ImportError as exc:
        raise RuntimeError('本地训练需要 calibration/requirements.txt 和主项目 models；打包版只需导入标定包。') from exc


def check_training(config):
    training_module()
    model = importlib.import_module('tools.calibration.model')
    for key in ('slm1', 'slm2'):
        model.phase_lut(config, key)


def train_bundle(*args, **kwargs):
    return training_module().train_bundle(*args, **kwargs)
