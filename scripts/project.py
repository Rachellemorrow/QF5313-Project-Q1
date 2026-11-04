"""Repository-local Windows bootstrap/runner; no PATH or policy changes.

Bootstrap with: py -3.11 -B scripts/project.py setup
The host Python runs only this standard-library launcher, not project code.
"""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
UV_VERSION = "0.13.7"
UV_URL = (
    "https://mirrors.aliyun.com/pypi/packages/8e/4f/"
    "041b59d2a38807a3e31ef41064cf3e45d2752fe0409d89040d52778049ae/"
    "uv-0.13.7-py3-none-win_amd64.whl"
)
UV_SHA256 = "2835e2ab397d0db86323b9975dce75ecd76aa9cfa0d6a2aea84734aae7f8c9b3"
UV = ROOT / ".tools/uv/uv.exe"
LOCAL_PATHS = {
    "UV_CACHE_DIR": ".uv-cache",
    "UV_PYTHON_CACHE_DIR": ".cache/python-downloads",
    "UV_PYTHON_INSTALL_DIR": ".tools/python",
    "UV_PYTHON_BIN_DIR": ".tools/bin",
    "UV_TOOL_DIR": ".tools/uv-tools",
    "UV_TOOL_BIN_DIR": ".tools/bin",
    "UV_CREDENTIALS_DIR": ".cache/uv-credentials",
    "UV_PROJECT_ENVIRONMENT": ".venv",
    "TEMP": ".tmp",
    "TMP": ".tmp",
    "TMPDIR": ".tmp",
    "XDG_CACHE_HOME": ".cache",
    "PYTHONPYCACHEPREFIX": ".cache/pycache",
    "PYTHONUSERBASE": ".cache/python-user",
    "PIP_CACHE_DIR": ".cache/pip",
    "TORCH_HOME": ".cache/torch",
    "TORCHINDUCTOR_CACHE_DIR": ".cache/torchinductor",
    "TRITON_CACHE_DIR": ".cache/triton",
    "HF_HOME": ".cache/huggingface",
    "MPLCONFIGDIR": ".cache/matplotlib",
    "NUMBA_CACHE_DIR": ".cache/numba",
}


def local_environment():
    """Return child-only configuration, preserving the host's PATH and profile."""
    env = os.environ.copy()
    # Ignore host-specific uv overrides (indexes, Python paths, active envs).
    # Network proxy/TLS variables are left intact; never log their values.
    for name in list(env):
        if name.startswith("UV_"):
            env.pop(name)
    for name, relative in LOCAL_PATHS.items():
        directory = ROOT / relative
        if name != "UV_PROJECT_ENVIRONMENT":
            directory.mkdir(parents=True, exist_ok=True)
        env[name] = str(directory)
    env.update({
        "UV_PYTHON_INSTALL_REGISTRY": "0",
        "UV_PYTHON_NO_REGISTRY": "1",
        "UV_PYTHON_INSTALL_BIN": "0",
        "UV_NO_MODIFY_PATH": "1",
        "UV_PYTHON_PREFERENCE": "only-managed",
        "UV_NO_SYSTEM_CONFIG": "1",
        "UV_NO_ENV_FILE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
    # External Python-path overrides must not inject another environment.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)
    return env


def run(arguments, env):
    command = [str(part) for part in arguments]
    display = [os.path.relpath(part, ROOT) if Path(part).is_absolute() and
               Path(part).is_relative_to(ROOT) else part for part in command]
    print("> " + subprocess.list2cmdline(display), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def bootstrap(env):
    if sys.platform != "win32" or platform.machine().upper() != "AMD64":
        raise RuntimeError("The pinned uv bootstrap supports Windows x86-64 only.")
    # The mirror is byte-verified against the official PyPI SHA-256 above.
    # Keep any older interrupted upstream download untouched.
    wheel = ROOT / f".tmp/uv-{UV_VERSION}-win_amd64-aliyun.whl"
    if not wheel.exists():
        print(f"Downloading pinned uv wheel to {wheel}", flush=True)
        # Use an explicit destination; no installer, pip, or global writes.
        with urllib.request.urlopen(UV_URL, timeout=120) as response:
            with wheel.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != UV_SHA256:
        raise RuntimeError("uv wheel SHA-256 mismatch; refusing to execute it.")
    with zipfile.ZipFile(wheel) as archive:
        entries = [name for name in archive.namelist() if name.endswith("/uv.exe")]
        if len(entries) != 1:
            raise RuntimeError("Expected exactly one uv.exe in the verified wheel.")
        executable = archive.read(entries[0])
        if UV.exists():
            if hashlib.sha256(UV.read_bytes()).digest() != hashlib.sha256(executable).digest():
                raise RuntimeError("Existing uv differs from the verified wheel; inspect manually.")
        else:
            UV.parent.mkdir(parents=True, exist_ok=True)
            with UV.open("xb") as output:
                output.write(executable)
    run([UV, "--version"], env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["bootstrap", "setup", "verify", "uv"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from the repository root.")
    env = local_environment()
    if args.action in {"bootstrap", "setup"}:
        bootstrap(env)
    elif not UV.is_file():
        parser.error("Local uv is missing; run bootstrap or setup first.")
    if args.action == "setup":
        python_version = (ROOT / ".python-version").read_text().strip()
        run([UV, "python", "install", python_version, "--no-bin", "--no-registry"], env)
        run([UV, "sync", "--locked"], env)
    elif args.action == "verify":
        run([UV, "--version"], env)
        run([UV, "run", "--locked", "--offline", "python", "--version"], env)
        run([UV, "lock", "--check", "--offline"], env)
        run([UV, "sync", "--locked", "--check", "--offline"], env)
        run([UV, "pip", "check", "--python", ROOT / ".venv/Scripts/python.exe"], env)
        run([UV, "run", "--locked", "--offline", "python", "-m", "pytest",
             "tests/q1/test_environment.py", "-v"], env)
    elif args.action == "uv":
        run([UV, *args.arguments], env)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
