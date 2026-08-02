"""
Tests for device utilities.
"""
import pytest
import torch
from src.utils.device import get_device, get_device_info, to_device

def test_get_device_returns_device():
    """get_device should return a torch.device."""
    device = get_device()
    assert isinstance(device, torch.device)

def test_get_device_force_cpu():
    """force_cpu should return CPU device."""
    device = get_device(force_cpu=True)
    assert device.type == "cpu"

def test_get_device_info_returns_dict():
    """get_device_info should return a dictionary."""
    info = get_device_info()
    assert isinstance(info, dict)
    assert "device" in info
    assert "device_type" in info

def test_to_device_tensor():
    """to_device should move tensor to specified device."""
    device = get_device(force_cpu=True)
    tensor = torch.randn(5)
    moved = to_device(tensor, device)
    assert moved.device.type == device.type

def test_to_device_list():
    """to_device should handle lists of tensors."""
    device = get_device(force_cpu=True)
    tensors = [torch.randn(3), torch.randn(4)]
    moved = to_device(tensors, device)
    assert all(t.device.type == device.type for t in moved)

def test_to_device_dict():
    """to_device should handle dicts of tensors."""
    device = get_device(force_cpu=True)
    tensors = {"a": torch.randn(3), "b": torch.randn(4)}
    moved = to_device(tensors, device)
    assert all(t.device.type == device.type for t in moved.values())
