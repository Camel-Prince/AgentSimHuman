"""GPUIdleFiller: occupy GPUs with synthetic workload while external API calls are in flight.

Usage (context-manager):
    filler = GPUIdleFiller(gpu_ids=[0, 1], agent_sas_path='/path/to/agent_SAS.py')
    with filler:
        results = call_external_api(...)  # GPUs occupied during this block

The subprocess is started on __enter__ and terminated (with wait) on __exit__, so
GPU memory is fully released before the training loop resumes GPU operations.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional


class GPUIdleFiller:
    """Context manager that launches agent_SAS.py as a subprocess during __enter__
    and terminates it during __exit__.

    Attributes:
        gpu_ids: GPU device indices to pass to agent_SAS.py. If None, all visible
            GPUs are used (determined from CUDA_VISIBLE_DEVICES or torch).
        agent_sas_path: Absolute path to agent_SAS.py.
        enabled: If False, start/stop are no-ops.
    """

    def __init__(
        self,
        agent_sas_path: str,
        gpu_ids: Optional[List[int]] = None,
        enabled: bool = True,
    ):
        self.agent_sas_path = str(agent_sas_path)
        self.enabled = enabled
        self._proc: Optional[subprocess.Popen] = None

        if gpu_ids is not None:
            self._gpu_ids = list(gpu_ids)
        else:
            self._gpu_ids = self._detect_gpu_ids()

    def _detect_gpu_ids(self) -> List[int]:
        """Return list of visible GPU indices (0-based within CUDA_VISIBLE_DEVICES)."""
        # Prefer torch if available; otherwise fall back to env var / single GPU.
        try:
            import torch
            n = torch.cuda.device_count()
            if n > 0:
                return list(range(n))
        except ImportError:
            pass
        cuda_vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_vis:
            try:
                ids = [int(x.strip()) for x in cuda_vis.split(',') if x.strip()]
                # Return 0-based indices within the visible set
                return list(range(len(ids)))
            except ValueError:
                pass
        return [0]

    def start(self):
        """Launch agent_SAS.py subprocess. No-op if disabled or already running."""
        if not self.enabled:
            return
        if self._proc is not None and self._proc.poll() is None:
            # Already running
            return
        if not self._gpu_ids:
            return
        gpus_str = ','.join(str(g) for g in self._gpu_ids)
        cmd = [sys.executable, self.agent_sas_path, '--gpus', gpus_str]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f'[GPUIdleFiller] started (pid={self._proc.pid}, gpus={gpus_str})')
        except Exception as e:
            print(f'[GPUIdleFiller] failed to start: {e}')
            self._proc = None

    def stop(self):
        """Terminate agent_SAS subprocess and wait for full exit. No-op if not running."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            # Already exited
            self._proc = None
            return
        pid = self._proc.pid
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            print(f'[GPUIdleFiller] stopped (pid={pid})')
        except Exception as e:
            print(f'[GPUIdleFiller] error during stop: {e}')
        finally:
            self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False  # Do not suppress exceptions
