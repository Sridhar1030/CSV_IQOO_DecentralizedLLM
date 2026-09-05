"""Host and process resource sampling (RAM, CPU) via psutil (BSD-3-Clause).

Readings are cached for a short TTL: `virtual_memory()` is a syscall/procfs
read, and a busy inference node would otherwise pay for it on every request.
"""

from __future__ import annotations

import os
import platform
import threading
import time
from dataclasses import asdict, dataclass

import psutil

from . import semconv as sc


@dataclass(frozen=True)
class MemorySnapshot:
    """A point-in-time view of memory for this process and its host."""

    process_rss_bytes: int
    process_vms_bytes: int
    process_memory_percent: float
    system_total_bytes: int
    system_used_bytes: int
    system_available_bytes: int
    system_percent: float
    process_cpu_percent: float
    captured_at: float

    def as_span_attributes(self) -> dict[str, int | float]:
        """Flatten into OTel span attributes."""
        return {
            sc.MEM_PROCESS_RSS: self.process_rss_bytes,
            sc.MEM_PROCESS_VMS: self.process_vms_bytes,
            sc.MEM_PROCESS_PERCENT: round(self.process_memory_percent, 3),
            sc.MEM_SYSTEM_TOTAL: self.system_total_bytes,
            sc.MEM_SYSTEM_USED: self.system_used_bytes,
            sc.MEM_SYSTEM_AVAILABLE: self.system_available_bytes,
            sc.MEM_SYSTEM_PERCENT: round(self.system_percent, 3),
            sc.CPU_PROCESS_PERCENT: round(self.process_cpu_percent, 3),
        }

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


_EMPTY = MemorySnapshot(0, 0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0)


class ResourceSampler:
    """TTL-cached psutil sampler, safe to call from many request threads."""

    def __init__(self, cache_ms: int = 250) -> None:
        self._cache_ttl = max(cache_ms, 0) / 1000.0
        self._lock = threading.Lock()
        self._cached: MemorySnapshot | None = None
        self._process = psutil.Process(os.getpid())
        # First call to cpu_percent() always returns 0.0; prime it so the
        # first real request gets a meaningful number.
        try:
            self._process.cpu_percent(interval=None)
        except Exception:  # pragma: no cover - platform dependent
            pass

    def sample(self, force: bool = False) -> MemorySnapshot:
        now = time.time()
        with self._lock:
            cached = self._cached
            if (
                not force
                and cached is not None
                and (now - cached.captured_at) < self._cache_ttl
            ):
                return cached

        snapshot = self._read(now)
        with self._lock:
            self._cached = snapshot
        return snapshot

    def _read(self, now: float) -> MemorySnapshot:
        try:
            vm = psutil.virtual_memory()
            mem_info = self._process.memory_info()
            try:
                proc_percent = self._process.memory_percent()
            except Exception:  # pragma: no cover
                proc_percent = 0.0
            try:
                # interval=None -> non-blocking, delta since the previous call.
                cpu_percent = self._process.cpu_percent(interval=None)
            except Exception:  # pragma: no cover
                cpu_percent = 0.0

            return MemorySnapshot(
                process_rss_bytes=int(mem_info.rss),
                process_vms_bytes=int(getattr(mem_info, "vms", 0)),
                process_memory_percent=float(proc_percent),
                system_total_bytes=int(vm.total),
                system_used_bytes=int(vm.used),
                system_available_bytes=int(vm.available),
                system_percent=float(vm.percent),
                process_cpu_percent=float(cpu_percent),
                captured_at=now,
            )
        except Exception:  # pragma: no cover - never break the request path
            return MemorySnapshot(**{**_EMPTY.as_dict(), "captured_at": now})


def host_attributes() -> dict[str, str | int]:
    """Static host facts for the OTel Resource."""
    return {
        sc.HOST_NAME: platform.node(),
        sc.HOST_ARCH: platform.machine(),
        sc.OS_TYPE: platform.system().lower(),
        sc.PROCESS_PID: os.getpid(),
    }
