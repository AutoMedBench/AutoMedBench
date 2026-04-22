"""Sandbox violation tests.

Covers two layers:
1. Static regex block (pre-execution) — `_check_isolation` rejects obviously
   escape-y code without running it.
2. Runtime audit hook — code that tries to access /data/private or similar
   forbidden paths exits with the SANDBOX VIOLATION sentinel.
"""

from __future__ import annotations

import os
import tempfile

from benchmark_runner import _check_isolation, execute_code


def test_static_block_catches_ground_truth():
    code = "with open('ground_truth.csv') as f: print(f.read())"
    assert "BLOCKED" in _check_isolation(code)


def test_static_block_catches_listdir_root():
    code = "import os; print(os.listdir('/'))"
    assert "BLOCKED" in _check_isolation(code)


def test_static_block_allows_normal_code():
    code = "print('hello'); x = 1 + 1"
    assert _check_isolation(code) == ""


def test_runtime_audit_blocks_private_access():
    with tempfile.TemporaryDirectory() as cwd:
        # Escapes static regex but runtime audit must still block it.
        code = (
            "import os\n"
            "try:\n"
            "    open(chr(47)+'data'+chr(47)+'private'+chr(47)+'x', 'r')\n"
            "    print('LEAK')\n"
            "except Exception as exc:\n"
            "    print('err:', exc)\n"
        )
        result = execute_code("python", code, cwd=cwd, timeout=20)
        assert result["exit_code"] == -1, result
        assert "SANDBOX VIOLATION" in result["stderr"]
        assert "/data/private" in result["stderr"]


def test_runtime_audit_blocks_repeated_access_each_call():
    """Each execute_code call that touches /data/private must be blocked.

    Multi-turn kill semantics are enforced by the runner's violation counter
    (caller-side), not by execute_code itself — but execute_code must emit the
    BLOCKED signal every time so the runner can count.
    """
    with tempfile.TemporaryDirectory() as cwd:
        code = (
            "import os\n"
            "try:\n"
            "    os.listdir(chr(47)+'data'+chr(47)+'private')\n"
            "except Exception as exc:\n"
            "    print('err:', exc)\n"
        )
        r1 = execute_code("python", code, cwd=cwd, timeout=20)
        r2 = execute_code("python", code, cwd=cwd, timeout=20)
        for result in (r1, r2):
            assert result["exit_code"] == -1
            assert "SANDBOX VIOLATION" in result["stderr"]


def test_shared_hf_cache_env_respected():
    with tempfile.TemporaryDirectory() as cwd, tempfile.TemporaryDirectory() as hf:
        os.environ["SHARED_HF_CACHE"] = hf
        try:
            code = "import os; print('HF', os.environ.get('HF_HOME'))"
            result = execute_code("python", code, cwd=cwd, timeout=20)
            assert result["exit_code"] == 0, result
            assert hf in result["stdout"]
        finally:
            os.environ.pop("SHARED_HF_CACHE", None)
