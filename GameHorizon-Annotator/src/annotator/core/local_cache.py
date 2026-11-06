from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .schema import atomic_write_json


_LOCK_OWNER_RE = re.compile(r"^pid=(\d+)\b")


def _lock_owner_snapshot(lock_path: Path) -> tuple[int | None, str | None]:
    try:
        snapshot = lock_path.read_text(encoding="ascii")
    except (FileNotFoundError, OSError, UnicodeError):
        return None, None
    match = _LOCK_OWNER_RE.match(snapshot)
    return (int(match.group(1)) if match else None), snapshot


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        # Windows reports access denied when the process exists but cannot be
        # queried by the current token. Other errors mean the PID is gone.
        return getattr(exc, "winerror", None) == 5
    return True


def _reclaim_dead_owner_lock(lock_path: Path) -> int | None:
    owner_pid, snapshot = _lock_owner_snapshot(lock_path)
    if owner_pid is None or snapshot is None or _pid_is_alive(owner_pid):
        return None
    try:
        # Do not unlink a lock that was replaced between inspection and
        # reclamation by another process.
        if lock_path.read_text(encoding="ascii") != snapshot:
            return None
        lock_path.unlink()
    except (FileNotFoundError, OSError, UnicodeError):
        return None
    return owner_pid


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def fingerprint_key(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@contextmanager
def exclusive_path_lock(
    lock_path: Path,
    *,
    timeout_s: float = 4 * 60 * 60,
    stale_s: float = 4 * 60 * 60,
    ready: Callable[[], bool] | None = None,
) -> Iterator[bool]:
    """Acquire a small cross-process lock using exclusive file creation.

    The yielded value is false when another process completed the requested
    work while this process was waiting. Stale locks left by terminated
    workers are removed after ``stale_s``.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    descriptor: int | None = None
    while descriptor is None:
        if ready is not None and ready():
            yield False
            return
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            reclaimed_pid = _reclaim_dead_owner_lock(lock_path)
            if reclaimed_pid is not None:
                print(
                    f"[cache-lock] reclaimed dead owner pid={reclaimed_pid}: {lock_path}",
                    flush=True,
                )
                continue
            try:
                age_s = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age_s > stale_s:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for cache lock: {lock_path}")
            time.sleep(0.25)
    try:
        os.write(
            descriptor,
            f"pid={os.getpid()} created={time.time():.6f}\n".encode("ascii"),
        )
        os.close(descriptor)
        descriptor = None
        yield True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def ensure_local_file_copy(source: Path, cache_dir: Path) -> Path:
    """Return an immutable fingerprinted local copy of ``source``.

    A cache-wide copy lock intentionally serializes reads from the benchmark's
    USB HDD. Once copied, every model shard can decode the same recording from
    NVMe without repeatedly scanning the slow source disk.
    """

    source = source.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"cache source not found: {source}")
    fingerprint = file_fingerprint(source)
    key = fingerprint_key({"schema_version": 1, "source": fingerprint})
    target = cache_dir / key[:2] / f"{key[:20]}-{source.name}"
    metadata_path = target.with_name(f"{target.name}.cache.json")

    def is_ready() -> bool:
        if not target.is_file() or not metadata_path.is_file():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            metadata.get("source") == fingerprint
            and target.stat().st_size == int(fingerprint["size"])
        )

    if is_ready():
        print(f"[local-video-cache] hit {source.name} -> {target}", flush=True)
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    copy_lock = cache_dir / ".source-copy.lock"
    with exclusive_path_lock(copy_lock, ready=is_ready) as acquired:
        if not acquired or is_ready():
            print(f"[local-video-cache] hit {source.name} -> {target}", flush=True)
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            temporary.unlink(missing_ok=True)
            print(
                f"[local-video-cache] copying {source.name} "
                f"({int(fingerprint['size']) / (1024 ** 3):.2f} GiB) -> {target}",
                flush=True,
            )
            shutil.copyfile(source, temporary)
            if temporary.stat().st_size != int(fingerprint["size"]):
                raise OSError(
                    f"cached copy size mismatch for {source.name}: "
                    f"{temporary.stat().st_size} != {fingerprint['size']}"
                )
            shutil.copystat(source, temporary)
            os.replace(temporary, target)
            atomic_write_json(
                metadata_path,
                {
                    "schema_version": 1,
                    "source": fingerprint,
                    "cached_file": str(target),
                },
            )
        finally:
            temporary.unlink(missing_ok=True)
    print(f"[local-video-cache] ready {source.name} -> {target}", flush=True)
    return target
