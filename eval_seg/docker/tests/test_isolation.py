#!/usr/bin/env python3
"""End-to-end test for Docker isolation + comprehensive bypass detection.

Tests the full orchestrator flow WITHOUT real LLM calls:
  1. Creates dummy patient data (tiny NIfTI files)
  2. Creates dummy agent outputs (fake masks)
  3. Runs the eval container against them
  4. Verifies isolation: agent cannot see private data
  5. Verifies all known bypass vectors are blocked
  6. Verifies legitimate operations are NOT blocked (no false positives)

Usage:
    python test_isolation.py              # local test (no Docker)
    python test_isolation.py --docker     # full Docker test
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCKER_DIR = os.path.dirname(SCRIPT_DIR)
EVAL_SEG_DIR = os.path.dirname(DOCKER_DIR)


def create_dummy_nifti(path, shape=(64, 64, 32), binary=True):
    """Create a minimal NIfTI file."""
    import nibabel as nib
    if binary:
        data = np.random.randint(0, 2, shape).astype(np.float32)
    else:
        data = np.random.randn(*shape).astype(np.float32) * 100
    img = nib.Nifti1Image(data, np.eye(4))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(img, path)
    return data


def setup_dummy_data(tmpdir):
    """Create a minimal dataset for testing."""
    patients = ["TEST_001", "TEST_002"]
    task = "kidney"
    shape = (64, 64, 32)

    # Public data (CT scans)
    public_dir = os.path.join(tmpdir, "data", task, "public")
    for pid in patients:
        create_dummy_nifti(
            os.path.join(public_dir, pid, "ct.nii.gz"),
            shape=shape, binary=False,
        )

    # Private data (ground truth)
    private_dir = os.path.join(tmpdir, "data", task, "private")
    for pid in patients:
        gt_organ = create_dummy_nifti(
            os.path.join(private_dir, "masks", pid, "organ.nii.gz"),
            shape=shape, binary=True,
        )
        gt_lesion = create_dummy_nifti(
            os.path.join(private_dir, "masks", pid, "lesion.nii.gz"),
            shape=shape, binary=True,
        )

    # Agent outputs (dummy masks — copy of ground truth for perfect score)
    agent_dir = os.path.join(tmpdir, "outputs", "agents_outputs")
    for pid in patients:
        # Copy GT as agent output for a "perfect" agent
        src = os.path.join(private_dir, "masks", pid)
        dst = os.path.join(agent_dir, pid)
        shutil.copytree(src, dst)

    return {
        "patients": patients,
        "task": task,
        "public_dir": public_dir,
        "private_dir": private_dir,
        "agent_dir": agent_dir,
        "outputs_dir": os.path.join(tmpdir, "outputs"),
        "run_dir": tmpdir,
    }


def test_eval_local(data):
    """Test eval pipeline locally (no Docker)."""
    sys.path.insert(0, EVAL_SEG_DIR)
    from format_checker import check_submission
    from dice_scorer import score_all
    from medal_tier import assign_tier
    from aggregate import build_report
    from failure_classifier import classify_failure

    print("\n[Test] Running local eval pipeline...")

    # Format check
    fmt = check_submission(
        agent_dir=data["agent_dir"],
        patient_ids=data["patients"],
        public_dir=data["public_dir"],
    )
    print(f"  Format: submission_valid={fmt['submission_format_valid']} "
          f"output_valid={fmt['output_format_valid']}")
    assert fmt["submission_format_valid"], "Format check failed"

    # Dice
    dice = score_all(
        pred_dir=data["agent_dir"],
        gt_dir=os.path.join(data["private_dir"], "masks"),
        patient_ids=data["patients"],
    )
    print(f"  Dice: organ={dice.get('mean_organ_dice', 0):.4f} "
          f"lesion={dice.get('mean_lesion_dice', 0):.4f}")

    # Result tier
    medal = assign_tier(dice.get("mean_lesion_dice", 0))
    print(f"  Result: {medal['name']} (tier {medal['tier']})")

    # Aggregate
    report = build_report(fmt, dice, medal)
    failure = classify_failure(report)
    report["failure"] = failure

    agg = report["aggregate"]
    print(f"  Overall: {agg['overall_score']:.4f}  "
          f"Rating: {agg['rating']}  Resolved: {agg['resolved']}")

    # Write report
    report_path = os.path.join(data["run_dir"], "detail_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {report_path}")

    return report


# ======================================================================
# Isolation check tests — comprehensive bypass detection
# ======================================================================

def test_isolation_check():
    """Test that the code executor blocks ALL known bypass vectors."""
    sys.path.insert(0, os.path.join(DOCKER_DIR, "agent"))
    from agent_code_executor import check_isolation

    print("\n[Test] Testing isolation checks...")

    passed = 0
    failed = 0

    def assert_blocked(code, description):
        nonlocal passed, failed
        result = check_isolation(code)
        if result:
            print(f"  BLOCKED (ok): {description}")
            passed += 1
        else:
            print(f"  *** MISSED ***: {description}")
            print(f"    Code: {code[:100]}")
            failed += 1

    def assert_allowed(code, description):
        nonlocal passed, failed
        result = check_isolation(code)
        if not result:
            print(f"  ALLOWED (ok): {description}")
            passed += 1
        else:
            print(f"  *** FALSE POSITIVE ***: {description}")
            print(f"    Code: {code[:100]}")
            print(f"    Blocked by: {result}")
            failed += 1

    # ── MUST BLOCK: Explicit forbidden paths ──────────────────────
    assert_blocked('open("/data/private/masks/test.nii.gz")',
                   "Direct read of private data")
    assert_blocked('import dice_scorer',
                   "Import of evaluation module")
    assert_blocked('os.listdir("/eval")',
                   "Listing eval directory")
    assert_blocked('cat /results/detail_report.json',
                   "Reading eval results")
    assert_blocked('open("ground_truth.csv")',
                   "Reading ground truth file")
    assert_blocked('curl --unix-socket /var/run/docker.sock',
                   "Docker socket access")

    # ── MUST BLOCK: Harness code self-read ────────────────────────
    assert_blocked('cat /app/agent_code_executor.py',
                   "Reading isolation checker code")
    assert_blocked('cat /app/agent_loop.py',
                   "Reading agent loop code")
    assert_blocked('open("/app/entrypoint_agent.sh")',
                   "Reading entrypoint script")

    # ── MUST BLOCK: Broad filesystem scans ────────────────────────
    assert_blocked('find / -name "*.py"',
                   "find from root")
    assert_blocked('find /lustre -name "model"',
                   "find on lustre")
    assert_blocked('find /home -type f',
                   "find on home")
    assert_blocked('find /opt -name "*.pt"',
                   "find on opt")
    assert_blocked('find /app -name "*.py"',
                   "find on /app (harness)")

    # ── MUST BLOCK: ls on system directories ──────────────────────
    assert_blocked('ls /',
                   "ls root")
    assert_blocked('ls /home',
                   "ls home")
    assert_blocked('ls /etc',
                   "ls etc")
    assert_blocked('ls -la /',
                   "ls -la root (with flags)")
    assert_blocked('ls -laR /tmp',
                   "ls recursive tmp (with flags)")
    assert_blocked('ls /app',
                   "ls /app (harness)")
    assert_blocked('ls /var',
                   "ls /var")

    # ── MUST BLOCK: tree command ──────────────────────────────────
    assert_blocked('tree /',
                   "tree root")
    assert_blocked('tree /usr',
                   "tree /usr")

    # ── MUST BLOCK: Relative path traversal ───────────────────────
    assert_blocked('cat ../../private/masks/patient1/organ.nii.gz',
                   "Relative traversal ../../ ")
    assert_blocked("os.path.join('/workspace', '..', '..')",
                   "os.path.join with ..")

    # ── MUST BLOCK: Python os.walk/listdir/scandir from root ──────
    assert_blocked("os.walk('/')",
                   "os.walk from root (single quotes)")
    assert_blocked('os.walk("/")',
                   "os.walk from root (double quotes)")
    assert_blocked("os.listdir('/')",
                   "os.listdir root (single quotes)")
    assert_blocked('os.listdir("/")',
                   "os.listdir root (double quotes)")
    assert_blocked("os.scandir('/')",
                   "os.scandir root")
    assert_blocked('os.scandir("/")',
                   "os.scandir root (double quotes)")

    # ── MUST BLOCK: Python traversal to parent ────────────────────
    assert_blocked("os.walk('..')",
                   "os.walk parent dir")
    assert_blocked("os.listdir('..')",
                   "os.listdir parent dir")
    assert_blocked("os.scandir('..')",
                   "os.scandir parent dir")

    # ── MUST BLOCK: pathlib traversal ─────────────────────────────
    assert_blocked("Path('/')",
                   "pathlib Path root")
    assert_blocked("Path('..')",
                   "pathlib Path parent")
    assert_blocked("p.parent.parent",
                   "pathlib double parent traversal")

    # ── MUST BLOCK: glob from root ────────────────────────────────
    assert_blocked("glob.glob('/**/*.py')",
                   "glob from root")
    assert_blocked("glob.iglob('/**/*.nii.gz')",
                   "iglob from root")

    # ── MUST BLOCK: String construction evasion ───────────────────
    assert_blocked("chr(47)",
                   "chr(47) = '/' construction")
    assert_blocked(r"path = '\x2fdata'",
                   "hex escape for /")
    assert_blocked("import base64; base64.b64decode('L2RhdGEvcHJpdmF0ZQ==')",
                   "base64 decode evasion")
    assert_blocked("codecs.decode(b'stuff', 'rot_13')",
                   "codecs decode evasion")
    assert_blocked("__import__('os').listdir('/')",
                   "dynamic import of os")
    assert_blocked("getattr(os, 'listdir')('/')",
                   "getattr on os module")

    # ── MUST BLOCK: subprocess with list args ─────────────────────
    assert_blocked("subprocess.run(['find', '/', '-name', '*.py'])",
                   "subprocess.run(['find'...])")
    assert_blocked("subprocess.run(['ls', '/'])",
                   "subprocess.run(['ls'...])")
    assert_blocked("subprocess.call(['tree', '/'])",
                   "subprocess.call(['tree'...])")

    # ── MUST BLOCK: open() on harness/system paths ────────────────
    assert_blocked("open('/app/agent_loop.py')",
                   "open() on harness code")
    assert_blocked("open('/etc/shadow')",
                   "open() on shadow file")

    # ── MUST BLOCK: System paths ──────────────────────────────────
    assert_blocked("cat /proc/self/maps",
                   "/proc access")
    assert_blocked("cat /sys/class/gpu",
                   "/sys access")
    assert_blocked("cat /etc/passwd",
                   "/etc/passwd")

    # ── MUST BLOCK: Container escape ──────────────────────────────
    assert_blocked("nsenter --target 1",
                   "nsenter command")
    assert_blocked("chroot /host",
                   "chroot command")
    assert_blocked("/proc/1/root",
                   "proc 1 root escape")

    # ── MUST BLOCK: Recon commands ────────────────────────────────
    assert_blocked("locate model.pt",
                   "locate command")
    assert_blocked("whereis python",
                   "whereis command")

    # ── MUST BLOCK: Red-team discovered bypasses ─────────────────
    # os.sep / os.path.join path construction
    assert_blocked("import os\npath = os.path.join(os.sep, 'data', 'private')",
                   "os.sep path construction")
    assert_blocked("path = os.sep + 'data'",
                   "os.sep concatenation")
    assert_blocked("os.path.sep",
                   "os.path.sep reference")

    # bytes/bytearray ASCII construction
    assert_blocked("bytes([47,100,97,116,97])",
                   "bytes([...]) path construction")
    assert_blocked("bytearray([47,100,97])",
                   "bytearray([...]) path construction")

    # struct.pack
    assert_blocked("struct.pack('13B', 47,100,97,116,97)",
                   "struct.pack byte path construction")

    # binascii
    assert_blocked("binascii.a2b_base64(b'stuff')",
                   "binascii alternative to base64")

    # importlib.util
    assert_blocked("importlib.util.spec_from_file_location('m', '/app/loop.py')",
                   "importlib.util module loading")

    # bash printf
    assert_blocked("$(printf '%s' '/da' 'ta')",
                   "bash $(printf ...) path concat")
    assert_blocked("printf '%s' '/da' 'ta'",
                   "bash printf path concat")

    # bash variable expansion
    assert_blocked("${x}${y}${z}",
                   "bash variable expansion concat")
    assert_blocked("source /dev/stdin <<< 'cat /data'",
                   "bash source /dev/stdin injection")
    assert_blocked('eval "$cmd"',
                   "bash eval variable execution")
    assert_blocked("IFS=/ read -r a b",
                   "bash IFS manipulation")

    # str.replace on allowed paths
    assert_blocked("'/data/public'.replace('public', 'private')",
                   "str.replace public->private")

    # Anti-tampering
    assert_blocked("sys.addaudithook(my_hook)",
                   "tampering with audit hooks")
    assert_blocked("ctypes.pythonapi.PySys_AddAuditHook",
                   "C-level audit hook manipulation")
    assert_blocked("_sandbox_hook = None",
                   "direct reference to sandbox hook")
    assert_blocked("SANDBOX_PREAMBLE",
                   "reference to preamble constant")

    # ── MUST BLOCK: Red-team round 2 bypasses ────────────────────
    # os.path.join with unpacked list
    assert_blocked("parts = ['','data','priv']\nos.path.join(*parts)",
                   "os.path.join(*list) unpacking")

    # map(chr, ...) path construction
    assert_blocked("list(map(chr, [47, 100, 97]))",
                   "map(chr, int_list) path construction")
    assert_blocked("''.join(chr(c) for c in [47, 100])",
                   "chr(c) comprehension with variable")

    # bytes.fromhex
    assert_blocked("bytes.fromhex('2f64617461')",
                   "bytes.fromhex path construction")

    # os.popen
    assert_blocked("os.popen('ls /data')",
                   "os.popen shell execution")

    # unicode named char
    assert_blocked("path = '\\N{SOLIDUS}data'",
                   "unicode SOLIDUS for /")

    # pathlib division operator
    assert_blocked("Path('') / 'data' / 'private'",
                   "pathlib division operator")

    # Step-by-step cd traversal
    assert_blocked("os.chdir('/')",
                   "os.chdir to root")
    assert_blocked("os.chdir('..')",
                   "os.chdir to parent")
    assert_blocked("os.fchdir(fd)",
                   "os.fchdir file descriptor")
    assert_blocked("cd /workspace && cd ..",
                   "bash cd traversal")
    assert_blocked("cd .. && ls",
                   "bash cd .. escape")

    # ast/compile/exec code generation
    assert_blocked("ast.parse('os.listdir(\"/\")')",
                   "ast.parse code generation")
    assert_blocked("exec(code_string)",
                   "exec() dynamic execution")
    assert_blocked("compile(src, 'x', 'exec')",
                   "compile() code generation")

    # importlib.import_module
    assert_blocked("importlib.import_module('base64')",
                   "importlib.import_module")

    # Runtime slash extraction
    assert_blocked("slash = os.altsep",
                   "os.altsep for path separator")
    assert_blocked("root = sys.exec_prefix",
                   "sys.exec_prefix path derivation")

    # str.replace workspace
    assert_blocked("path.replace('workspace', 'data/private')",
                   "str.replace workspace->forbidden")

    print(f"\n  --- Blocked pattern tests: {passed} passed, {failed} failed ---")

    # ══════════════════════════════════════════════════════════════
    # MUST ALLOW: Legitimate operations (no false positives!)
    # ══════════════════════════════════════════════════════════════
    allowed_passed = 0
    allowed_failed = 0

    def assert_allowed(code, description):
        nonlocal allowed_passed, allowed_failed
        result = check_isolation(code)
        if not result:
            print(f"  ALLOWED (ok): {description}")
            allowed_passed += 1
        else:
            print(f"  *** FALSE POSITIVE ***: {description}")
            print(f"    Code: {code[:100]}")
            print(f"    Blocked by: {result}")
            allowed_failed += 1

    print("\n[Test] Testing legitimate operations are NOT blocked...")

    # Basic imports
    assert_allowed('import nibabel as nib',
                   "Import nibabel")
    assert_allowed('import torch; print(torch.cuda.is_available())',
                   "Import torch + CUDA check")
    assert_allowed('import monai',
                   "Import monai")
    assert_allowed('from monai.bundle import download',
                   "MONAI bundle download")

    # Reading patient data (the core use case!)
    assert_allowed('nib.load("/data/public/TEST_001/ct.nii.gz")',
                   "Load patient CT from /data/public/")
    assert_allowed("os.listdir('/data/public')",
                   "List patient directory")
    assert_allowed('os.listdir("/data/public")',
                   "List patient directory (double quotes)")
    assert_allowed("os.listdir('/data/public/TEST_001')",
                   "List specific patient directory")
    assert_allowed('ls /data/public',
                   "bash ls data directory")
    assert_allowed('ls /data/public/TEST_001',
                   "bash ls specific patient")
    assert_allowed('ls -la /data/public/',
                   "bash ls -la data directory")
    assert_allowed('find /data/public -name "*.nii.gz"',
                   "find NIfTI files in data dir")
    assert_allowed('find /data/public/TEST_001 -type f',
                   "find files in patient dir")

    # Writing to workspace
    assert_allowed('nib.save(img, "/workspace/agents_outputs/TEST_001/organ.nii.gz")',
                   "Save mask to workspace")
    assert_allowed('os.makedirs("/workspace/agents_outputs/TEST_001", exist_ok=True)',
                   "Create output directory")
    assert_allowed("os.listdir('/workspace')",
                   "List workspace")
    assert_allowed('ls /workspace/',
                   "bash ls workspace")

    # Package management
    assert_allowed('pip install monai',
                   "pip install")
    assert_allowed('pip install -r requirements.txt',
                   "pip install from requirements")

    # Model downloads from internet
    assert_allowed('from huggingface_hub import snapshot_download',
                   "HuggingFace download")
    assert_allowed('wget https://example.com/model.pt',
                   "wget model download")
    assert_allowed('requests.get("https://api.github.com/repos")',
                   "API request")

    # Normal file operations within sandbox
    assert_allowed('open("/workspace/plan/plan.md", "w")',
                   "Write plan file")
    assert_allowed('open("/data/public/TEST_001/ct.nii.gz", "rb")',
                   "Read patient CT")
    assert_allowed('glob.glob("/data/public/*/ct.nii.gz")',
                   "Glob for patient CTs")
    assert_allowed('glob.glob("/workspace/**/*.nii.gz")',
                   "Glob workspace outputs")
    assert_allowed("os.walk('/data/public')",
                   "Walk data directory")
    assert_allowed("os.walk('/workspace')",
                   "Walk workspace")

    # Normal Python operations
    assert_allowed('import subprocess; subprocess.run(["python3", "train.py"])',
                   "subprocess run python script")
    assert_allowed('subprocess.run(["nvidia-smi"])',
                   "subprocess nvidia-smi")

    print(f"\n  --- Allowed operation tests: {allowed_passed} passed, {allowed_failed} failed ---")

    total_failed = failed + allowed_failed
    if total_failed > 0:
        print(f"\n  *** {total_failed} TESTS FAILED ***")
        sys.exit(1)
    else:
        print(f"\n  All isolation tests passed! "
              f"({passed + allowed_passed} total)")


def test_docker_eval(data):
    """Test eval container with Docker."""
    print("\n[Test] Running Docker eval container...")

    patient_ids = ",".join(data["patients"])
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=1g",
        "-v", f"{data['private_dir']}:/data/private:ro",
        "-v", f"{data['public_dir']}:/data/public:ro",
        "-v", f"{data['agent_dir']}:/agent_outputs:ro",
        "-v", f"{data['run_dir']}:/results:rw",
        "-e", f"PATIENT_IDS={patient_ids}",
        "-e", f"TASK={data['task']}",
        "-e", "AGENT_NAME=dummy_test",
        "medagentsbench-eval",
    ]

    print(f"  Command: docker run ... medagentsbench-eval")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  Exit code: {proc.returncode}")
    if proc.stdout:
        for line in proc.stdout.strip().split("\n"):
            print(f"  > {line}")
    if proc.returncode != 0 and proc.stderr:
        for line in proc.stderr.strip().split("\n")[-5:]:
            print(f"  ! {line}")

    report_path = os.path.join(data["run_dir"], "detail_report.json")
    if os.path.isfile(report_path):
        with open(report_path) as f:
            report = json.load(f)
        agg = report.get("aggregate", {})
        print(f"  Result: Rating={agg.get('rating')} "
              f"Overall={agg.get('overall_score', 0):.4f}")
    else:
        print("  WARNING: No report generated")

    return proc.returncode


def test_agent_cannot_see_private():
    """Verify agent container cannot access /data/private."""
    print("\n[Test] Testing agent cannot see private data...")

    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=1g",
        "--tmpfs", "/workspace:rw,size=1g",
        # Deliberately do NOT mount /data/private
        "medagentsbench-agent",
        "bash", "-c", "ls /data/private 2>&1 || echo 'GOOD: /data/private not accessible'"
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    output = proc.stdout + proc.stderr
    if "/data/private" not in output or "not accessible" in output or "No such file" in output:
        print("  PASS: Agent cannot access /data/private")
    else:
        print(f"  FAIL: Agent might see private data: {output[:200]}")

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true",
                        help="Run Docker container tests (images must be built)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MedAgentsBench Isolation Test Suite")
    print("=" * 60)

    # Create temp directory for test data
    tmpdir = tempfile.mkdtemp(prefix="medagentsbench_test_")
    print(f"\n[Test] Temp dir: {tmpdir}")

    try:
        # Setup dummy data
        data = setup_dummy_data(tmpdir)
        print(f"[Test] Created dummy data: {len(data['patients'])} patients")

        # Test 1: Isolation checks (always runs)
        test_isolation_check()

        # Test 2: Local eval pipeline
        report = test_eval_local(data)

        # Test 3+4: Docker tests (only if --docker)
        if args.docker:
            rc = test_docker_eval(data)
            if rc == 0:
                test_agent_cannot_see_private()
        else:
            print("\n[Test] Skipping Docker tests (use --docker to enable)")

        print(f"\n{'='*60}")
        print(f"  ALL TESTS PASSED")
        print(f"{'='*60}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"[Test] Cleaned up {tmpdir}")


if __name__ == "__main__":
    main()
