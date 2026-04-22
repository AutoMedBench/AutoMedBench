#!/usr/bin/env python3
"""Sandboxed code execution for the image-enhancement agent container.

Runs Python or bash code as a subprocess inside the container. No conda —
the container IS the environment.

Isolation enforcement (3 layers):
  Layer 1: Static — blocked path substrings + INFERENCE-ONLY training patterns
  Layer 2: Static — regex-based escape detection (broad filesystem scans, etc.)
  Layer 3: Runtime — Python sys.addaudithook() sandbox preamble injected into
           every Python script. Catches dynamically constructed paths
           (os.sep, bytes([47,...]), join(), replace(), etc.) at the C level.
"""

import os
import re
import subprocess
import tempfile
import textwrap


# ── Layer 1: Paths the agent must NEVER reference ──────────────────────
BLOCKED_PATTERNS = [
    "/data/private",
    "/eval",
    "/results",
    "/bands",
    "reference.npy",
    "ground_truth.csv",
    "baseline_bands.json",
    # Agent must NOT read its own harness
    "/app/agent_loop",
    "/app/agent_code_executor",
    "/app/entrypoint_agent",
]

# ── Layer 1b: INFERENCE-ONLY — no gradient / optimizer / training calls ─
TRAINING_PATTERNS = [
    r"\.backward\s*\(",
    r"loss\.backward",
    r"optimizer\.step\s*\(",
    r"optim\.(?:SGD|Adam|AdamW|RMSprop|Adagrad)\s*\(",
    r"torch\.optim\.",
    r"\.zero_grad\s*\(",
    r"model\.train\s*\(",
]
_TRAIN_RE = [re.compile(p) for p in TRAINING_PATTERNS]


# ── Layer 2: Regex-based escape detection ──────────────────────────────
BLOCKED_ESCAPE_REGEXES = [
    # Broad filesystem scans — find
    r"find\s+/\s", r"find\s+/\n", r"find\s+/\"", r"find\s+/'", r"find\s+/$",
    r"find\s+/home\b", r"find\s+/opt\b", r"find\s+/usr\b",
    r"find\s+/root\b", r"find\s+/tmp\b", r"find\s+/app\b",

    # Broad filesystem scans — ls
    r"\bls\s+(-[a-zA-Z]+\s+)?/\s", r"\bls\s+(-[a-zA-Z]+\s+)?/\n",
    r"\bls\s+(-[a-zA-Z]+\s+)?/\"", r"\bls\s+(-[a-zA-Z]+\s+)?/'",
    r"\bls\s+(-[a-zA-Z]+\s+)?/$",
    r"\bls\s+(-[a-zA-Z]+\s+)?/home\b", r"\bls\s+(-[a-zA-Z]+\s+)?/etc\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/opt\b", r"\bls\s+(-[a-zA-Z]+\s+)?/usr\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/root\b", r"\bls\s+(-[a-zA-Z]+\s+)?/tmp\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/var\b", r"\bls\s+(-[a-zA-Z]+\s+)?/app\b",

    # tree / du
    r"\btree\s+/\s", r"\btree\s+/\n", r"\btree\s+/$", r"\btree\s+/[a-z]",
    r"\bdu\s+(-[a-zA-Z]+\s+)?/\s", r"\bdu\s+(-[a-zA-Z]+\s+)?/$",

    # Parent traversal
    r"\.\./\.\.", r"os\.path\.join\([^)]*\.\.",

    # Python filesystem APIs: root traversal
    r"os\.walk\s*\(\s*['\"]\/['\"]",
    r"os\.listdir\s*\(\s*['\"]\/['\"]",
    r"os\.scandir\s*\(\s*['\"]\/['\"]",
    r"glob\.glob\s*\(\s*['\"]\/(?!data\/|workspace\/)",
    r"glob\.iglob\s*\(\s*['\"]\/(?!data\/|workspace\/)",

    # Python filesystem APIs: parent traversal
    r"os\.walk\s*\(\s*['\"]\.\.[\/'\"\\]",
    r"os\.listdir\s*\(\s*['\"]\.\.[\/'\"\\]",
    r"os\.scandir\s*\(\s*['\"]\.\.[\/'\"\\]",

    # pathlib
    r"Path\s*\(\s*['\"]\/['\"\)]",
    r"Path\s*\(\s*['\"]\.\.[\/'\"\\]",
    r"\.parent\s*\.parent",

    # System paths
    r"\/proc\/", r"\/sys\/",
    r"\/etc\/passwd", r"\/etc\/shadow", r"\/etc\/hosts",

    # Container escape
    r"docker\.sock", r"\bnsenter\b", r"\bchroot\b",
    r"\/var\/run\/", r"\/proc\/1\/root",

    # Recon
    r"\blocate\s+", r"\bwhereis\s+",

    # String construction evasion (static catches)
    r"chr\s*\(\s*47\s*\)", r"\\x2f",
    r"b64decode|b64encode",
    r"codecs\.decode",
    r"__import__\s*\(\s*['\"]os['\"]",
    r"getattr\s*\(\s*os\s*,",

    # subprocess with list args to dangerous commands
    r"subprocess\.\w+\(\s*\[\s*['\"]find['\"]",
    r"subprocess\.\w+\(\s*\[\s*['\"]ls['\"]",
    r"subprocess\.\w+\(\s*\[\s*['\"]cat['\"].*\/app\/",
    r"subprocess\.\w+\(\s*\[\s*['\"]tree['\"]",

    # open() on system/harness paths
    r"open\s*\(\s*['\"]\/app\/",
    r"open\s*\(\s*['\"]\/etc\/",

    # Runtime path construction tricks
    r"\bos\.sep\b", r"\bos\.path\.sep\b",
    r"\bbytes\s*\(\s*\[", r"\bbytearray\s*\(\s*\[",
    r"\bstruct\.pack\b",
    r"\bbinascii\.",
    r"\bimportlib\.util\b",
    r"\bprintf\s+['\"]%s['\"]", r"\$\(printf\b",
    r"\$\{[a-z]\}\$\{[a-z]\}",
    r"\bsource\s+\/dev\/stdin\b",
    r"\beval\s+\"\$",
    r"\bIFS\s*=",

    # String morph
    r"\.replace\s*\(\s*['\"]public['\"]",
    r"\.replace\s*\(\s*['\"]workspace['\"]",

    # path-join unpacking
    r"os\.path\.join\s*\(\s*\*",

    # map(chr,...) evasion
    r"\bmap\s*\(\s*chr\b",
    r"chr\s*\(\s*[a-z_]\w*\s*\)",

    # bytes.fromhex / os.popen
    r"\bbytes\.fromhex\b", r"\bos\.popen\b",

    # unicode named chars
    r"\\N\{SOLIDUS\}", r"\\N\{REVERSE SOLIDUS\}",

    # pathlib division hiding segments
    r"Path\s*\(\s*['\"]['\"]?\s*\)\s*\/",
    r"Path\s*\(\s*\)\s*\/",

    # Directory traversal
    r"os\.chdir\s*\(\s*['\"]\/['\"]",
    r"os\.chdir\s*\(\s*['\"]\.\.?['\"]",
    r"\bos\.fchdir\b",
    r"\bcd\s+/workspace\s*&&\s*cd\s+\.\.",
    r"\bcd\s+\.\.\s",
    r"\bcd\s+/\s", r"\bcd\s+/$",

    # ast / compile / exec
    r"\bast\.parse\b", r"\bcompile\s*\(", r"\bexec\s*\(",
    r"\bimportlib\.import_module\b",
    r"\bos\.altsep\b", r"\bsys\.exec_prefix\b",
]
_ESCAPE_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE)
              for p in BLOCKED_ESCAPE_REGEXES]


# ── Layer 3: Runtime audit hook (Python) ──────────────────────────────
# Injected as a preamble into every Python script. Uses sys.addaudithook()
# (irremovable once installed) to catch all file access at the C level.
# Also forces HF / Torch / pip caches to live inside /workspace to avoid
# polluting the read-only system tree.
SANDBOX_PREAMBLE = textwrap.dedent("""\
    import sys as _sys, os as _os

    _ws = '/workspace'
    _os.environ.setdefault('HF_HOME', _os.path.join(_ws, '.cache', 'huggingface'))
    _os.environ.setdefault('TORCH_HOME', _os.path.join(_ws, '.cache', 'torch'))
    _os.environ.setdefault('XDG_CACHE_HOME', _os.path.join(_ws, '.cache'))
    _os.environ.setdefault('PIP_CACHE_DIR', _os.path.join(_ws, '.cache', 'pip'))

    # PyTorch 2.6+ weights_only default would break pretrained DNN loading.
    try:
        import torch as _torch
        _orig_torch_load = _torch.load
        def _patched_torch_load(*_a, **_kw):
            _kw.setdefault('weights_only', False)
            return _orig_torch_load(*_a, **_kw)
        _torch.load = _patched_torch_load
    except ImportError:
        pass

    def _sandbox_hook(_event, _args):
        _FORBIDDEN_PREFIXES = ('/data/private', '/eval/', '/results/', '/bands/',
                               '/app/agent_loop', '/app/agent_code_executor',
                               '/app/entrypoint_agent')
        _FORBIDDEN_SUFFIXES = ('reference.npy', 'ground_truth.csv',
                               'baseline_bands.json')
        _AUDIT_EVENTS = ('open', 'os.listdir', 'os.scandir',
                         'os.chdir', 'os.mkdir', 'os.rename',
                         'os.remove', 'os.symlink',
                         'shutil.copyfile', 'shutil.copytree',
                         'shutil.rmtree')
        if _event in _AUDIT_EVENTS and _args:
            _path = str(_args[0])
            try:
                _resolved = _os.path.realpath(_path)
            except Exception:
                _resolved = _path
            for _fp in _FORBIDDEN_PREFIXES:
                if _resolved.startswith(_fp) or _path.startswith(_fp):
                    _sys.stderr.write(
                        f"SANDBOX VIOLATION: access to {_resolved} is FORBIDDEN.\\n"
                        f"All remaining scores will be zeroed. Rating = F.\\n")
                    _sys.stderr.flush()
                    _os._exit(99)
            for _sf in _FORBIDDEN_SUFFIXES:
                if _resolved.endswith(_sf) or _path.endswith(_sf):
                    _sys.stderr.write(
                        f"SANDBOX VIOLATION: access to {_resolved} is FORBIDDEN "
                        f"(ground-truth file).\\n"
                        f"All remaining scores will be zeroed. Rating = F.\\n")
                    _sys.stderr.flush()
                    _os._exit(99)

    _sys.addaudithook(_sandbox_hook)
    del _sandbox_hook
    # ── end sandbox preamble ──
""")


# ── Bash sandbox wrapper ──────────────────────────────────────────────
BASH_PREAMBLE = textwrap.dedent("""\
    # ── sandbox preamble ──
    _sandbox_check() {
        local resolved
        for arg in "$@"; do
            resolved=$(readlink -f "$arg" 2>/dev/null || echo "$arg")
            case "$resolved" in
                /data/private*|/eval/*|/results/*|/bands/*|*/reference.npy|*/ground_truth.csv|*/baseline_bands.json|/app/agent_loop*|/app/agent_code_executor*|/app/entrypoint_agent*)
                    echo "SANDBOX VIOLATION: access to $resolved is FORBIDDEN." >&2
                    echo "All remaining scores will be zeroed. Rating = F." >&2
                    exit 99
                    ;;
            esac
        done
    }
    cat()    { _sandbox_check "$@"; command cat "$@"; }
    head()   { _sandbox_check "$@"; command head "$@"; }
    tail()   { _sandbox_check "$@"; command tail "$@"; }
    less()   { _sandbox_check "$@"; command less "$@"; }
    more()   { _sandbox_check "$@"; command more "$@"; }
    cp()     { _sandbox_check "$@"; command cp "$@"; }
    mv()     { _sandbox_check "$@"; command mv "$@"; }
    ln()     { _sandbox_check "$@"; command ln "$@"; }
    # ── end sandbox preamble ──
""")


def check_isolation(code: str, extra_blocked: list = None) -> str:
    """Return error string if code violates static rules, else empty string."""
    all_blocked = BLOCKED_PATTERNS + (extra_blocked or [])
    for bp in all_blocked:
        if bp in code:
            return f"BLOCKED: code references restricted pattern '{bp}'"

    for rx in _TRAIN_RE:
        m = rx.search(code)
        if m:
            return (f"BLOCKED: INFERENCE-ONLY violation — matched '{m.group()}'. "
                    f"Fine-tuning / gradient steps are not permitted.")

    for rx in _ESCAPE_RE:
        m = rx.search(code)
        if m:
            return (f"BLOCKED: sandbox escape detected — matched '{m.group()}'. "
                    f"You may ONLY access /data/public/ and /workspace/.")

    anti_tamper_patterns = [
        r"_sandbox_hook",
        r"_sandbox_check",
        r"sys\.addaudithook",
        r"addaudithook",
        r"PySys_AddAuditHook",
        r"PySys_Audit",
        r"ctypes.*pythonapi",
        r"sandbox.*preamble",
        r"SANDBOX_PREAMBLE",
        r"# ── end sandbox preamble",
    ]
    for pat in anti_tamper_patterns:
        if re.search(pat, code, re.IGNORECASE):
            return ("BLOCKED: anti-tampering — code attempts to reference or "
                    "disable the sandbox enforcement layer.")

    return ""


def execute_code(language: str, code: str, cwd: str,
                 timeout: int = None,
                 extra_blocked: list = None) -> dict:
    """Run code in the container, return stdout + stderr + exit_code."""
    violation = check_isolation(code, extra_blocked=extra_blocked)
    if violation:
        return {"exit_code": -1, "stdout": "", "stderr": violation}

    if language == "python":
        full_code = SANDBOX_PREAMBLE + code
    else:
        full_code = BASH_PREAMBLE + code

    suffix = ".py" if language == "python" else ".sh"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, dir=cwd, delete=False
    ) as f:
        f.write(full_code)
        script_path = f.name

    try:
        if language == "python":
            cmd = ["python3", script_path]
        else:
            cmd = ["bash", script_path]

        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
        )

        stdout = proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout
        stderr = proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr

        if proc.returncode == 99 and "SANDBOX VIOLATION" in stderr:
            return {
                "exit_code": -1,
                "stdout": stdout,
                "stderr": f"BLOCKED: {stderr.strip()}",
            }

        return {"exit_code": proc.returncode, "stdout": stdout, "stderr": stderr}
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "",
                "stderr": f"TIMEOUT: execution exceeded {timeout}s"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "",
                "stderr": f"Execution error: {e}"}
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
