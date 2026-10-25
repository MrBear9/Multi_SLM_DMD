"""Shared, scale-invariant amplitude metric; physical and proxy scores stay separate."""
import torch


def amplitude(intensity):
    return torch.sqrt(intensity.clamp_min(0)/intensity.mean().clamp_min(1e-8)+1e-7)


def amplitude_loss(prediction, target):
    return (amplitude(prediction)-amplitude(target)).square().mean()


def physical_proxy_loss(prediction, observed, target):
    predicted_amp=amplitude(prediction)
    actual_amp=amplitude(observed).detach()
    proxy=predicted_amp+(actual_amp-predicted_amp).detach()
    return (proxy-amplitude(target).detach()).square().mean()
