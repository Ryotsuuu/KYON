"""
agents/Resource_Management/hardware_manager.py
=============================================
Unified Hardware Detection, Telemetry Auditing & Auto-Tuning Subsystem.

Capabilities:
- Comprehensive CPU auditing (Physical & Logical core mapping, frequency, utilization)
- Comprehensive GPU auditing (Dedicated NVIDIA RTX 4050 Laptop vs Integrated Intel UHD, VRAM, Shared Memory)
- Dynamic worker allocation optimized for 12 CPU logical cores + GPU Tensor cores
- Safe fallback protocols for CPU-only execution
"""
import os
import sys
import platform
import logging
from typing import Dict, Any, Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from agents.GPU_config import configure_gpu, get_device_info

logger = logging.getLogger(__name__)


class HardwareManager:
    """Audits system hardware, manages resource quotas, and tunes concurrency."""

    _instance: Optional['HardwareManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HardwareManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.profile = self._audit_hardware()

    def _audit_hardware(self) -> Dict[str, Any]:
        """Collect full hardware telemetry across OS, CPU, RAM, and GPU."""
        telemetry: Dict[str, Any] = {
            'os': platform.system(),
            'os_release': platform.release(),
            'os_version': platform.version(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'cpu': {
                'logical_cores': os.cpu_count() or 4,
                'physical_cores': os.cpu_count() or 4,
                'current_load_percent': 0.0,
            },
            'ram': {
                'total_gb': 16.0,
                'available_gb': 8.0,
                'used_percent': 50.0,
                'cap_threshold_percent': 95.0,  # Strict user cap
            },
            'gpu': {
                'cuda_available': False,
                'active_device_str': 'cpu',
                'device_name': 'CPU',
                'vram_total_gb': 0.0,
                'vram_allocated_gb': 0.0,
                'vram_reserved_gb': 0.0,
                'shared_vram_gb': 14.0,
                'is_nvidia': False,
            },
            'optimal_workers': 8,
        }

        # Audit CPU and RAM via psutil
        if _PSUTIL_AVAILABLE:
            try:
                logical = psutil.cpu_count(logical=True) or (os.cpu_count() or 4)
                physical = psutil.cpu_count(logical=False) or max(1, logical // 2)
                telemetry['cpu']['logical_cores'] = logical
                telemetry['cpu']['physical_cores'] = physical
                telemetry['cpu']['current_load_percent'] = psutil.cpu_percent(interval=None)

                vm = psutil.virtual_memory()
                telemetry['ram']['total_gb'] = round(vm.total / (1024 ** 3), 2)
                telemetry['ram']['available_gb'] = round(vm.available / (1024 ** 3), 2)
                telemetry['ram']['used_percent'] = vm.percent
            except Exception as e:
                logger.debug(f"psutil telemetry error: {e}")

        # Audit GPU via GPU_config & PyTorch
        gpu_cfg = configure_gpu(force_refresh=True)
        telemetry['gpu']['cuda_available'] = gpu_cfg['cuda_available']
        telemetry['gpu']['active_device_str'] = gpu_cfg['device_str']
        telemetry['gpu']['device_name'] = gpu_cfg['device_name']
        telemetry['gpu']['vram_total_gb'] = gpu_cfg['gpu_vram_gb']
        telemetry['gpu']['shared_vram_gb'] = gpu_cfg['shared_memory_gb']
        telemetry['gpu']['is_nvidia'] = gpu_cfg['nvidia_detected']

        if _TORCH_AVAILABLE and gpu_cfg['cuda_available']:
            try:
                dev_id = gpu_cfg['cuda_device_id'] or 0
                telemetry['gpu']['vram_allocated_gb'] = round(torch.cuda.memory_allocated(dev_id) / (1024 ** 3), 3)
                telemetry['gpu']['vram_reserved_gb'] = round(torch.cuda.memory_reserved(dev_id) / (1024 ** 3), 3)
            except Exception:
                pass

        # Calculate optimal worker concurrency for simulations
        # Leave 2 logical cores headroom for OS, GPU dispatch, and background I/O
        logical_cores = telemetry['cpu']['logical_cores']
        telemetry['optimal_workers'] = max(2, min(logical_cores - 2, 12))

        return telemetry

    def refresh(self) -> Dict[str, Any]:
        """Refresh dynamic telemetry (RAM, CPU, VRAM load)."""
        self.profile = self._audit_hardware()
        return self.profile

    def get_optimal_worker_count(self) -> int:
        """Get recommended parallel worker threads for simulation throughput."""
        return self.profile.get('optimal_workers', 8)

    def get_summary_report(self) -> str:
        """Generate a formatted telemetry string."""
        p = self.refresh()
        gpu_info = get_device_info()
        cpu_info = f"{p['cpu']['logical_cores']} Logical Cores ({p['cpu']['physical_cores']} Physical)"
        ram_info = f"{p['ram']['total_gb']} GB Total ({p['ram']['used_percent']}% used, {p['ram']['available_gb']} GB free) | Cap: 95%"
        return (
            f"=== Hardware Telemetry ===\n"
            f"OS: {p['os']} {p['os_release']} ({p['architecture']})\n"
            f"CPU: {cpu_info}\n"
            f"RAM: {ram_info}\n"
            f"GPU Acceleration: {gpu_info}\n"
            f"Optimal Simulation Workers: {p['optimal_workers']}\n"
            f"=========================="
        )


_hardware_manager: Optional[HardwareManager] = None


def get_hardware_manager() -> HardwareManager:
    global _hardware_manager
    if _hardware_manager is None:
        _hardware_manager = HardwareManager()
    return _hardware_manager
