"""
app/core/job_store.py
======================
Thread-safe job store.
  - Uses Redis when available (production)
  - Falls back to in-memory dict (dev / single-instance)

Job lifecycle: queued → processing → done | failed
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


def _new_job_id() -> str:
    return str(uuid.uuid4())


# ── In-memory store ────────────────────────────────────────────────────────────

class InMemoryJobStore:
    """Thread-safe in-memory job store. Suitable for single-process deployments."""

    def __init__(self, ttl_seconds: int = 4 * 3600):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ttl  = ttl_seconds

    def create(self, **fields) -> str:
        job_id = _new_job_id()
        job = {
            "job_id":      job_id,
            "status":      JobStatus.QUEUED,
            "created_at":  time.time(),
            "updated_at":  time.time(),
            "started_at":  None,
            "finished_at": None,
            "input_path":  None,
            "output_path": None,
            "error":       None,
            "result":      None,
            **fields,
        }
        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False
            self._jobs[job_id].update(fields)
            self._jobs[job_id]["updated_at"] = time.time()
            return True

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def list_all(self) -> list[dict]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True)

    def count_by_status(self) -> dict[str, int]:
        counts = {s.value: 0 for s in JobStatus}
        with self._lock:
            for j in self._jobs.values():
                status = j.get("status", "unknown")
                if isinstance(status, JobStatus):
                    status = status.value
                counts[status] = counts.get(status, 0) + 1
        return counts

    def purge_expired(self) -> int:
        cutoff = time.time() - self._ttl
        with self._lock:
            expired = [jid for jid, j in self._jobs.items() if j.get("created_at", 0) < cutoff]
            for jid in expired:
                del self._jobs[jid]
        return len(expired)


# ── Redis store ────────────────────────────────────────────────────────────────

class RedisJobStore:
    """
    Redis-backed job store. Supports multi-process / multi-worker deployments.
    Falls back to InMemoryJobStore automatically if Redis is unreachable.
    """
    KEY_PREFIX = "pdftrans:job:"

    def __init__(self, redis_url: str, ttl_seconds: int = 4 * 3600):
        self._ttl = ttl_seconds
        try:
            import redis as redis_lib
            self._r = redis_lib.from_url(redis_url, decode_responses=True, socket_timeout=3)
            self._r.ping()
            log.info(f"RedisJobStore connected: {redis_url}")
        except Exception as exc:
            log.warning(f"Redis unavailable ({exc}) — falling back to in-memory store")
            self._r = None
            self._fallback = InMemoryJobStore(ttl_seconds)

    def _key(self, job_id: str) -> str:
        return f"{self.KEY_PREFIX}{job_id}"

    def create(self, **fields) -> str:
        if self._r is None:
            return self._fallback.create(**fields)
        job_id = _new_job_id()
        job = {
            "job_id":      job_id,
            "status":      JobStatus.QUEUED.value,
            "created_at":  str(time.time()),
            "updated_at":  str(time.time()),
            "started_at":  "",
            "finished_at": "",
            "input_path":  "",
            "output_path": "",
            "error":       "",
            "result":      "",
        }
        job.update({k: (json.dumps(v) if isinstance(v, dict) else str(v or "")) for k, v in fields.items()})
        self._r.hset(self._key(job_id), mapping=job)
        self._r.expire(self._key(job_id), self._ttl)
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        if self._r is None:
            return self._fallback.get(job_id)
        raw = self._r.hgetall(self._key(job_id))
        if not raw:
            return None
        # Deserialise types
        for k in ("created_at", "updated_at", "started_at", "finished_at"):
            if raw.get(k):
                try:
                    raw[k] = float(raw[k])
                except ValueError:
                    raw[k] = None
            else:
                raw[k] = None
        if raw.get("result"):
            try:
                raw["result"] = json.loads(raw["result"])
            except Exception:
                pass
        return raw

    def update(self, job_id: str, **fields) -> bool:
        if self._r is None:
            return self._fallback.update(job_id, **fields)
        key = self._key(job_id)
        if not self._r.exists(key):
            return False
        mapping = {}
        for k, v in fields.items():
            if isinstance(v, dict):
                mapping[k] = json.dumps(v)
            elif v is None:
                mapping[k] = ""
            elif isinstance(v, JobStatus):
                mapping[k] = v.value
            else:
                mapping[k] = str(v)
        mapping["updated_at"] = str(time.time())
        self._r.hset(key, mapping=mapping)
        self._r.expire(key, self._ttl)
        return True

    def delete(self, job_id: str) -> bool:
        if self._r is None:
            return self._fallback.delete(job_id)
        return bool(self._r.delete(self._key(job_id)))

    def list_all(self) -> list[dict]:
        if self._r is None:
            return self._fallback.list_all()
        keys = self._r.keys(f"{self.KEY_PREFIX}*")
        jobs = [self.get(k.replace(self.KEY_PREFIX, "")) for k in keys]
        jobs = [j for j in jobs if j]
        return sorted(jobs, key=lambda j: j.get("created_at") or 0, reverse=True)

    def count_by_status(self) -> dict[str, int]:
        if self._r is None:
            return self._fallback.count_by_status()
        counts = {s.value: 0 for s in JobStatus}
        for j in self.list_all():
            s = j.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return counts

    def purge_expired(self) -> int:
        """Redis TTL handles expiry automatically. This is a no-op."""
        if self._r is None:
            return self._fallback.purge_expired()
        return 0


# ── Factory ────────────────────────────────────────────────────────────────────

_store_instance: Optional[InMemoryJobStore | RedisJobStore] = None
_store_lock = threading.Lock()


def get_job_store(config=None) -> InMemoryJobStore | RedisJobStore:
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    with _store_lock:
        if _store_instance is not None:
            return _store_instance
        ttl = getattr(config, "JOB_TTL_SECONDS", 4 * 3600) if config else 4 * 3600
        redis_url = (
            getattr(config, "REDIS_URL", None) if config
            else os.environ.get("REDIS_URL", "")
        )
        if redis_url:
            _store_instance = RedisJobStore(redis_url, ttl)
        else:
            log.info("No REDIS_URL — using in-memory job store")
            _store_instance = InMemoryJobStore(ttl)
        return _store_instance
