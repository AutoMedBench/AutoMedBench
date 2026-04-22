"""Preflight checks for sweep.py (shm sizing guardrail)."""

from __future__ import annotations

import os
from unittest import mock

from sweep import preflight_shm


def test_preflight_shm_single_worker_always_ok():
    ok, msg = preflight_shm(parallel_workers=1)
    assert ok is True
    assert msg == ""


def _fake_statvfs(free_gib: float):
    block = 4096
    avail = int(free_gib * (1024 ** 3) / block)
    return os.statvfs_result((block, block, 0, 0, avail, 0, 0, 0, 0, 255))


def test_preflight_shm_enough_space_passes(tmp_path):
    with mock.patch("sweep.os.statvfs", return_value=_fake_statvfs(32.0)):
        ok, msg = preflight_shm(parallel_workers=4)
    assert ok is True
    assert "32.0" in msg


def test_preflight_shm_low_space_fails():
    with mock.patch("sweep.os.statvfs", return_value=_fake_statvfs(2.0)):
        ok, msg = preflight_shm(parallel_workers=4)
    assert ok is False
    assert "2.0" in msg
    assert "required" in msg


def test_preflight_shm_missing_path_skips():
    with mock.patch("sweep.os.statvfs", side_effect=OSError("nope")):
        ok, msg = preflight_shm(parallel_workers=4)
    assert ok is True
    assert "skipping" in msg
