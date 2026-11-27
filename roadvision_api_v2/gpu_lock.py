"""
Shared GPU execution lock and single-process worker guard.

Prevents uncontrolled simultaneous CUDA inference between the legacy V1 `/api/detect`
endpoint and the RoadVision V2 bounded GPU worker on 4 GB VRAM hardware, and guards
against duplicate worker processes via an OS-level lock file.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

# In-process mutual exclusion shared by legacy /api/detect and V2 GPU worker
_GPU_INFERENCE_LOCK = threading.Lock()


@contextmanager
def gpu_inference_lock(timeout: float = -1.0) -> Iterator[bool]:
    """
    Acquire the process-wide GPU inference lock.

    When timeout < 0, blocks until the lock is free.
    When timeout >= 0, attempts acquisition up to `timeout` seconds and yields False if busy.
    """
    if timeout < 0:
        _GPU_INFERENCE_LOCK.acquire()
        acquired = True
    else:
        acquired = _GPU_INFERENCE_LOCK.acquire(timeout=timeout)
    try:
        yield acquired
    finally:
        if acquired:
            _GPU_INFERENCE_LOCK.release()


def is_gpu_busy() -> bool:
    """Non-blocking probe indicating whether the GPU lock is currently held."""
    if _GPU_INFERENCE_LOCK.acquire(blocking=False):
        _GPU_INFERENCE_LOCK.release()
        return False
    return True


class SingleProcessWorkerGuard:
    """
    OS-level non-blocking file lock ensuring at most one active V2 GPU worker
    claims jobs against a given SQLite database across local processes.
    """

    def __init__(self, lock_path: str) -> None:
        self.lock_path = os.path.abspath(lock_path)
        self._fh: Optional[object] = None
        self._acquired = False

    def acquire(self) -> bool:
        if self._acquired:
            return True
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        fh = open(self.lock_path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                if os.path.getsize(self.lock_path) == 0:
                    fh.write(b"0")
                    fh.flush()
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fh.seek(0)
            fh.truncate()
            fh.write(f"pid={os.getpid()}\n".encode("ascii"))
            fh.flush()
            self._fh = fh
            self._acquired = True
            return True
        except OSError:
            try:
                fh.close()
            except Exception:
                pass
            return False

    def release(self) -> None:
        if not self._acquired or self._fh is None:
            return
        fh = self._fh
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                fh.close()
            except Exception:
                pass
            self._fh = None
            self._acquired = False
