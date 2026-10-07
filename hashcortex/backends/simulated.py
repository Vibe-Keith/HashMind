"""CPU simulator of an S9 hashboard using hashlib SHA-256."""

from __future__ import annotations

from dataclasses import dataclass

from .base import ASICBackend, HashJob, HashResult, header_with_nonce, meets_target, sha256d


@dataclass
class SimStats:
    jobs: int = 0
    hashes: int = 0
    nonces_returned: int = 0


class SimulatedS9Backend(ASICBackend):
    """Bit-exact functional model of BM1387 job semantics (not timing).

    ``min_difficulty_bits`` models the chip's ticket mask: real hardware will
    not report shares below some difficulty. The true floor for a BM1387 is an
    open question for phase 2; the default 0 means "idealized, report anything
    the job asks for".
    """

    def __init__(self, min_difficulty_bits: int = 0) -> None:
        self.min_difficulty_bits = min_difficulty_bits
        self.stats = SimStats()
        self._results: dict[int, HashResult] = {}
        self._pending: dict[int, HashJob] = {}

    def submit_job(self, job: HashJob) -> int:
        if job.difficulty_bits < self.min_difficulty_bits:
            raise ValueError(
                f"job difficulty {job.difficulty_bits} below chip floor {self.min_difficulty_bits}"
            )
        if job.job_id in self._pending or job.job_id in self._results:
            raise ValueError(f"duplicate job_id {job.job_id}")
        self._pending[job.job_id] = job
        return job.job_id

    def _execute(self, job: HashJob) -> HashResult:
        res = HashResult(job.job_id)
        for n in range(job.nonce_start, job.nonce_start + job.nonce_count):
            if meets_target(sha256d(header_with_nonce(job.header_prefix, n)), job.difficulty_bits):
                res.nonces.append(n)
        res.hashes_computed = job.nonce_count
        self.stats.jobs += 1
        self.stats.hashes += job.nonce_count
        self.stats.nonces_returned += len(res.nonces)
        return res

    def get_result(self, job_id: int, timeout: float | None = None) -> HashResult | None:
        if job_id in self._pending:
            self._results[job_id] = self._execute(self._pending.pop(job_id))
        return self._results.pop(job_id, None)
