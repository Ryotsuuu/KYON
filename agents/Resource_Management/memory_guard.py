"""
agents/Resource_Management/memory_guard.py
=========================================
Dynamic RAM & VRAM Safety Guardian with 95% Memory Cap Enforcement.

Features:
- Enforces strict 95% system RAM usage ceiling
- Proactive OOM prevention with multi-stage memory reclamation
- Real-time VRAM monitoring & garbage collection integration
- Streaming disk flusher to persist simulation replays and free heap memory
"""
import os
import gc
import time
import logging
from typing import Callable, List, Optional, Dict, Any

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

logger = logging.getLogger(__name__)


class MemoryCapExceededException(Exception):
    """Raised when system RAM exceeds the hard 95% threshold and cannot be reclaimed."""
    pass


class MemoryGuard:
    """Monitors system RAM and VRAM, enforcing a strict 95% cap and auto-saving simulation state."""

    def __init__(self, ram_cap_percent: float = 95.0, warning_threshold_percent: float = 88.0):
        self.ram_cap_percent = ram_cap_percent
        self.warning_threshold_percent = warning_threshold_percent
        self._flush_callbacks: List[Callable[[], int]] = []
        self._last_clean_time = time.time()

    def register_flush_callback(self, callback: Callable[[], int]) -> None:
        """Register a callback that saves in-memory buffers to disk and frees RAM."""
        if callback not in self._flush_callbacks:
            self._flush_callbacks.append(callback)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current system RAM and GPU VRAM statistics."""
        stats = {
            'ram_used_percent': 0.0,
            'ram_total_gb': 16.0,
            'ram_available_gb': 8.0,
            'vram_allocated_gb': 0.0,
            'vram_reserved_gb': 0.0,
            'is_critical': False,
            'is_warning': False,
        }

        if _PSUTIL_AVAILABLE:
            try:
                vm = psutil.virtual_memory()
                stats['ram_used_percent'] = vm.percent
                stats['ram_total_gb'] = round(vm.total / (1024 ** 3), 2)
                stats['ram_available_gb'] = round(vm.available / (1024 ** 3), 2)
                stats['is_warning'] = vm.percent >= self.warning_threshold_percent
                stats['is_critical'] = vm.percent >= self.ram_cap_percent
            except Exception:
                pass

        if _TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                stats['vram_allocated_gb'] = round(torch.cuda.memory_allocated() / (1024 ** 3), 3)
                stats['vram_reserved_gb'] = round(torch.cuda.memory_reserved() / (1024 ** 3), 3)
            except Exception:
                pass

        return stats

    def reclaim_memory(self, force: bool = False) -> int:
        """Perform garbage collection and invoke registered disk flush callbacks.

        Returns:
            Number of records flushed to disk by callbacks.
        """
        flushed_records = 0

        # Execute registered disk flushes to free heap
        for callback in self._flush_callbacks:
            try:
                flushed_records += callback()
            except Exception as e:
                logger.warning(f"MemoryGuard flush callback error: {e}")

        # Python heap collection
        gc.collect()

        # PyTorch CUDA cache purging
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        self._last_clean_time = time.time()
        return flushed_records

    def check_and_enforce(self, raise_on_critical: bool = False) -> bool:
        """Check current memory against the 95% cap. Reclaim memory if high.

        Returns:
            True if memory is within safe operating limits, False if near/above cap.
        """
        stats = self.get_memory_stats()
        current_percent = stats['ram_used_percent']

        if current_percent >= self.warning_threshold_percent:
            logger.info(f"MemoryGuard: RAM utilization at {current_percent:.1f}% (Threshold: {self.warning_threshold_percent}%). Initiating proactive cleanup.")
            self.reclaim_memory(force=True)

            # Re-check after reclamation
            updated_stats = self.get_memory_stats()
            if updated_stats['ram_used_percent'] >= self.ram_cap_percent:
                msg = (
                    f"CRITICAL MEMORY CAP REACHED: System RAM at {updated_stats['ram_used_percent']:.1f}% "
                    f"(Hard Limit: {self.ram_cap_percent}%). Halting allocation to prevent OOM crash."
                )
                logger.error(msg)
                if raise_on_critical:
                    raise MemoryCapExceededException(msg)
                return False

        return True


_global_memory_guard: Optional[MemoryGuard] = None


def get_memory_guard() -> MemoryGuard:
    global _global_memory_guard
    if _global_memory_guard is None:
        _global_memory_guard = MemoryGuard(ram_cap_percent=95.0, warning_threshold_percent=88.0)
    return _global_memory_guard
