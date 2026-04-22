#!/usr/bin/env python3
"""Sandboxed code execution for the agent container.

Runs Python or bash code as subprocess inside the container.
No conda wrapping — the container IS the environment.

Isolation enforcement (3 layers):
  Layer 1: Static — blocked path substrings
  Layer 2: Static — regex-based escape detection
  Layer 3: Runtime — Python sys.addaudithook() sandbox preamble
           injected into every Python script to catch dynamically
           constructed paths (os.sep, bytes([47,...]), join(), etc.)
"""

import os
import re
import subprocess
import tempfile
import textwrap

# ── Layer 1: Paths the agent must NEVER reference ──────────────────────
# Defense-in-depth: the container doesn't mount these, but we block anyway.
BLOCKED_PATTERNS = [
    "/data/private",
    "/eval",
    "/results",
    "ground_truth",
    "det2d_scorer",
    "format_checker",
    "medal_tier",
    "aggregate.py",
    "failure_classifier",
    "detail_report",
    # Agent must NOT read its own harness (contains blocked-pattern list)
    "/app/agent_loop",
    "/app/agent_code_executor",
    "/app/entrypoint_agent",
]

# ── Layer 2: Regex-based escape detection ──────────────────────────────
# Uses word-boundary / whitespace anchors so that
# "ls /data/public" does NOT match the "ls /" rule.
BLOCKED_ESCAPE_REGEXES = [
    # -- Broad filesystem scans: find --
    r"find\s+/\s",           # find / ...
    r"find\s+/\n",           # find / at end of line
    r"find\s+/\"",           # find /"
    r"find\s+/'",            # find /'
    r"find\s+/$",            # find / at end of string
    r"find\s+/lustre\b",    # find /lustre
    r"find\s+/home\b",      # find /home
    r"find\s+/opt\b",       # find /opt
    r"find\s+/usr\b",       # find /usr
    r"find\s+/root\b",      # find /root
    r"find\s+/tmp\b",       # find /tmp
    r"find\s+/app\b",       # find /app (harness code)

    # -- Broad filesystem scans: ls (with optional flags like ls -la /) --
    r"\bls\s+(-[a-zA-Z]+\s+)?/\s",     # ls / ... (with optional flags)
    r"\bls\s+(-[a-zA-Z]+\s+)?/\n",     # ls / at EOL
    r"\bls\s+(-[a-zA-Z]+\s+)?/\"",     # ls /"
    r"\bls\s+(-[a-zA-Z]+\s+)?/'",      # ls /'
    r"\bls\s+(-[a-zA-Z]+\s+)?/$",      # ls / at end of string
    r"\bls\s+(-[a-zA-Z]+\s+)?/lustre\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/home\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/etc\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/opt\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/usr\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/root\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/tmp\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/var\b",
    r"\bls\s+(-[a-zA-Z]+\s+)?/app\b",

    # -- Broad filesystem scans: tree --
    r"\btree\s+/\s",
    r"\btree\s+/\n",
    r"\btree\s+/$",
    r"\btree\s+/[a-z]",

    # -- Broad filesystem scans: du --
    r"\bdu\s+(-[a-zA-Z]+\s+)?/\s",
    r"\bdu\s+(-[a-zA-Z]+\s+)?/$",

    # -- Relative path traversal (parent directory escape) --
    r"\.\./\.\.",                        # ../../  (two levels up = always escaping)
    r"os\.path\.join\([^)]*\.\.",        # os.path.join(x, '..')

    # -- Python filesystem APIs: root traversal --
    r"os\.walk\s*\(\s*['\"]\/['\"]",          # os.walk('/')
    r"os\.listdir\s*\(\s*['\"]\/['\"]",       # os.listdir('/')
    r"os\.scandir\s*\(\s*['\"]\/['\"]",       # os.scandir('/')
    r"glob\.glob\s*\(\s*['\"]\/(?!data\/|workspace\/)",
    r"glob\.iglob\s*\(\s*['\"]\/(?!data\/|workspace\/)",

    # -- Python filesystem APIs: parent traversal --
    r"os\.walk\s*\(\s*['\"]\.\.[\/'\"\\]",
    r"os\.listdir\s*\(\s*['\"]\.\.[\/'\"\\]",
    r"os\.scandir\s*\(\s*['\"]\.\.[\/'\"\\]",

    # -- pathlib traversal --
    r"Path\s*\(\s*['\"]\/['\"\)]",            # Path('/')
    r"Path\s*\(\s*['\"]\.\.[\/'\"\\]",        # Path('..')
    r"\.parent\s*\.parent",                    # path.parent.parent

    # -- System paths --
    r"\/proc\/",
    r"\/sys\/",
    r"\/etc\/passwd",
    r"\/etc\/shadow",
    r"\/etc\/hosts",

    # -- Container escape --
    r"docker\.sock",
    r"\bnsenter\b",
    r"\bchroot\b",
    r"\/var\/run\/",
    r"\/proc\/1\/root",

    # -- Recon commands --
    r"\blocate\s+",
    r"\bwhereis\s+",

    # -- String construction evasion (static catches) --
    r"chr\s*\(\s*47\s*\)",                    # chr(47) == '/'
    r"\\x2f",                                  # hex escape for '/'
    r"b64decode|b64encode",                    # base64 evasion
    r"codecs\.decode",                         # codec-based evasion
    r"__import__\s*\(\s*['\"]os['\"]",        # dynamic import of os
    r"getattr\s*\(\s*os\s*,",                 # getattr(os, 'listdir')

    # -- subprocess with list args --
    r"subprocess\.\w+\(\s*\[\s*['\"]find['\"]",
    r"subprocess\.\w+\(\s*\[\s*['\"]ls['\"]",
    r"subprocess\.\w+\(\s*\[\s*['\"]cat['\"].*\/app\/",
    r"subprocess\.\w+\(\s*\[\s*['\"]tree['\"]",

    # -- open() on system/harness paths --
    r"open\s*\(\s*['\"]\/app\/",
    r"open\s*\(\s*['\"]\/etc\/",

    # ── Red-team discovered bypasses (runtime path construction) ──

    # os.sep / os.path.sep used to construct paths
    r"\bos\.sep\b",                            # os.sep
    r"\bos\.path\.sep\b",                      # os.path.sep

    # bytes/bytearray with ASCII 47 (= '/') to construct paths
    r"\bbytes\s*\(\s*\[",                      # bytes([...])
    r"\bbytearray\s*\(\s*\[",                  # bytearray([...])

    # struct.pack to construct path bytes
    r"\bstruct\.pack\b",

    # binascii (alternative base64)
    r"\bbinascii\.",

    # importlib.util to load arbitrary files as modules
    r"\bimportlib\.util\b",

    # bash printf for path construction
    r"\bprintf\s+['\"]%s['\"]",               # printf '%s' ... (path concat)
    r"\$\(printf\b",                           # $(printf ...) subshell

    # bash variable expansion tricks
    r"\$\{[a-z]\}\$\{[a-z]\}",               # ${x}${y}${z} variable concat
    r"\bsource\s+\/dev\/stdin\b",             # source /dev/stdin (code injection)
    r"\beval\s+\"\$",                          # eval "$..." (variable eval)
    r"\bIFS\s*=",                              # IFS manipulation

    # Python string manipulation to morph allowed paths into forbidden
    r"\.replace\s*\(\s*['\"]public['\"]",     # .replace('public', 'private')
    r"\.replace\s*\(\s*['\"]workspace['\"]",  # .replace('workspace', ...)

    # os.path.join with unpacked list (hides path segments)
    r"os\.path\.join\s*\(\s*\*",              # os.path.join(*list)

    # map(chr, ...) to construct path from int list
    r"\bmap\s*\(\s*chr\b",                    # map(chr, [47,...])
    # chr(c) comprehension
    r"chr\s*\(\s*[a-z_]\w*\s*\)",             # chr(c), chr(x), chr(i) — variable arg

    # bytes.fromhex
    r"\bbytes\.fromhex\b",

    # os.popen (can execute shell commands with constructed paths)
    r"\bos\.popen\b",

    # unicode named characters for path separators
    r"\\N\{SOLIDUS\}",                        # \N{SOLIDUS} = '/'
    r"\\N\{REVERSE SOLIDUS\}",                # \N{REVERSE SOLIDUS} = '\'

    # pathlib division operator to build paths
    r"Path\s*\(\s*['\"]['\"]?\s*\)\s*\/",     # Path('') / ...
    r"Path\s*\(\s*\)\s*\/",                   # Path() / ...

    # Step-by-step directory traversal
    r"os\.chdir\s*\(\s*['\"]\/['\"]",         # os.chdir('/')
    r"os\.chdir\s*\(\s*['\"]\.\.?['\"]",      # os.chdir('..') or os.chdir('.')
    r"\bos\.fchdir\b",                         # os.fchdir(fd)
    r"\bcd\s+/workspace\s*&&\s*cd\s+\.\.",    # cd /workspace && cd ..
    r"\bcd\s+\.\.\s",                          # cd .. (bash)
    r"\bcd\s+/\s",                             # cd / (bash, to root)
    r"\bcd\s+/$",                              # cd / (bash, end of string)

    # ast/compile/exec for code generation evasion
    r"\bast\.parse\b",                         # ast.parse()
    r"\bcompile\s*\(",                         # compile()
    r"\bexec\s*\(",                            # exec()

    # importlib.import_module (alternative to importlib.util)
    r"\bimportlib\.import_module\b",

    # os.altsep and extracting slash from runtime
    r"\bos\.altsep\b",
    r"\bsys\.exec_prefix\b",
]

_ESCAPE_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE)
              for p in BLOCKED_ESCAPE_REGEXES]


# ── Layer 3: Runtime audit hook (Python only) ─────────────────────────
# Injected as a preamble into every Python script. Uses sys.addaudithook()
# to catch ALL file access at the C level, regardless of how the path was
# constructed (os.sep, bytes(), join(), replace(), etc.).
#
# The hook is irremovable once installed (Python design guarantee).
SANDBOX_PREAMBLE = textwrap.dedent("""\
    import sys as _sys, os as _os

    # Force all downloads/caches into workspace
    _ws = '/workspace'
    _os.environ.setdefault('HF_HOME', _os.path.join(_ws, '.cache', 'huggingface'))
    _os.environ.setdefault('TORCH_HOME', _os.path.join(_ws, '.cache', 'torch'))
    _os.environ.setdefault('XDG_CACHE_HOME', _os.path.join(_ws, '.cache'))
    _os.environ.setdefault('PIP_CACHE_DIR', _os.path.join(_ws, '.cache', 'pip'))
    _os.environ.setdefault('MONAI_DATA_DIRECTORY', _os.path.join(_ws, '.cache', 'monai'))

    # Fix torch.load weights_only default (PyTorch 2.6+) so agent never
    # needs to patch it — preserves original map_location (keeps GPU).
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
        _FORBIDDEN_PREFIXES = ('/data/private', '/eval/', '/results/',
                               '/app/agent_loop', '/app/agent_code_executor',
                               '/app/entrypoint_agent')
        _ALLOWED_PREFIXES = ('/data/public/', '/workspace/', '/tmp/',
                             '/usr/', '/lib/', '/lib64/', '/opt/conda/',
                             '/opt/nvidia/', '/dev/null', '/dev/urandom',
                             '/dev/shm/')
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
            if _resolved.startswith('/') and not any(
                    _resolved.startswith(_p) for _p in _ALLOWED_PREFIXES):
                pass  # allow — may be Python stdlib/site-packages

    _sys.addaudithook(_sandbox_hook)
    del _sandbox_hook
    # ── end sandbox preamble ──
""")


# ── Bash sandbox wrapper ──────────────────────────────────────────────
# For bash scripts, we prepend a trap and readonly functions that block
# common evasion patterns at the shell level.
BASH_PREAMBLE = textwrap.dedent("""\
    # ── sandbox preamble ──
    _sandbox_check() {
        local resolved
        for arg in "$@"; do
            resolved=$(readlink -f "$arg" 2>/dev/null || echo "$arg")
            case "$resolved" in
                /data/private*|/eval/*|/results/*|/app/agent_loop*|/app/agent_code_executor*|/app/entrypoint_agent*)
                    echo "SANDBOX VIOLATION: access to $resolved is FORBIDDEN." >&2
                    echo "All remaining scores will be zeroed. Rating = F." >&2
                    exit 99
                    ;;
            esac
        done
    }
    # Wrap dangerous commands
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
    """Return error string if code references blocked patterns, else empty.

    Enforces static sandboxing (layers 1+2). Layer 3 (runtime audit hook)
    is enforced at execution time by the preamble injected into scripts.
    """
    # --- Layer 1: explicit path substrings ---
    all_blocked = BLOCKED_PATTERNS + (extra_blocked or [])
    for bp in all_blocked:
        if bp in code:
            return f"BLOCKED: code references restricted pattern '{bp}'"

    # --- Layer 2: regex escape patterns ---
    for rx in _ESCAPE_RE:
        m = rx.search(code)
        if m:
            return (f"BLOCKED: sandbox escape detected — matched '{m.group()}'. "
                    f"You may ONLY access /data/public/ and /workspace/.")

    # --- Layer 2b: ground_truth keyword ---
    if "ground_truth" in code:
        return "BLOCKED: code references ground_truth (private data)"

    # --- Layer 2c: anti-tampering — block attempts to disable the audit hook ---
    anti_tamper_patterns = [
        r"_sandbox_hook",            # reference to our hook
        r"_sandbox_check",           # reference to our bash wrapper
        r"sys\.addaudithook",        # trying to add another hook
        r"addaudithook",             # same, without sys prefix
        r"PySys_AddAuditHook",       # C-level hook manipulation
        r"PySys_Audit",              # C-level audit call
        r"ctypes.*pythonapi",        # ctypes access to Python C API
        r"sandbox.*preamble",        # reference to preamble itself
        r"SANDBOX_PREAMBLE",         # reference to our constant
        r"# ── end sandbox preamble",  # trying to detect/skip preamble
    ]
    for pat in anti_tamper_patterns:
        if re.search(pat, code, re.IGNORECASE):
            return (f"BLOCKED: anti-tampering — code attempts to reference "
                    f"or disable the sandbox enforcement layer.")

    return ""


def execute_code(language: str, code: str, cwd: str,
                 timeout: int = None,
                 extra_blocked: list = None) -> dict:
    """Run code in the container, return stdout + stderr + exit_code.

    Python scripts get a sys.addaudithook() preamble injected.
    Bash scripts get function wrappers for cat/head/tail/etc.
    """
    violation = check_isolation(code, extra_blocked=extra_blocked)
    if violation:
        return {"exit_code": -1, "stdout": "", "stderr": violation}

    # Inject runtime sandbox preamble
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

        # Check if the runtime sandbox killed the process (exit code 99)
        if proc.returncode == 99 and "SANDBOX VIOLATION" in stderr:
            return {
                "exit_code": -1,
                "stdout": stdout,
                "stderr": f"BLOCKED: {stderr.strip()}",
            }

        return {
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"TIMEOUT: execution exceeded {timeout}s",
        }
    except Exception as e:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Execution error: {e}",
        }
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
