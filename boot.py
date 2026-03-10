#!/usr/bin/env python3
"""LLM Vector Space Explorer — one-command launcher.  Run: python boot.py"""

import sys, os, platform, subprocess, shutil
from pathlib import Path

VENV = Path.home() / ".llm_explorer_venv"
IS_WIN = platform.system() == "Windows"
BIN = VENV / ("Scripts" if IS_WIN else "bin")
PY = BIN / ("python.exe" if IS_WIN else "python")
PIP = BIN / ("pip.exe" if IS_WIN else "pip")
APP = Path(__file__).parent / "app.py"

DEPS = [
    "dash", "plotly", "numpy", "scikit-learn",
    "torch --index-url https://download.pytorch.org/whl/cpu",
    "transformers", "umap-learn",
]

# ── tiny helpers ──────────────────────────────────────────────

def in_venv():
    """Are we already running inside our managed venv?"""
    return sys.prefix == str(VENV)

def run(cmd, **kw):
    """Run a command, exit on failure."""
    print(f"  → {' '.join(str(c) for c in cmd)}")
    subprocess.check_call(cmd, **kw)

def pip_install(spec):
    """Install one pip spec (may contain flags like --index-url)."""
    run([str(PIP), "install", "-q"] + spec.split())

def create_venv():
    """Create fresh venv with pip."""
    import venv as _venv
    print(f"Creating venv at {VENV}")
    _venv.create(str(VENV), with_pip=True)
    run([str(PY), "-m", "pip", "install", "-q", "--upgrade", "pip"])

def install_deps():
    """Install all project dependencies."""
    print("Installing dependencies...")
    for dep in DEPS:
        try:
            pip_install(dep)
        except subprocess.CalledProcessError:
            print(f"  ⚠ Failed: {dep} (continuing)")

def ensure_venv():
    """Create venv + install deps if anything is missing."""
    if not PY.exists():
        if VENV.exists():
            shutil.rmtree(VENV)
        create_venv()
        install_deps()
    else:
        # Quick check: try importing a key dep inside the venv
        rc = subprocess.call(
            [str(PY), "-c", "import dash, torch, transformers, umap"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if rc != 0:
            install_deps()

def relaunch():
    """Re-exec app.py under the venv python, then exit."""
    print(f"Launching app via {PY}\n")
    try:
        result = subprocess.run([str(PY), str(APP)], env={**os.environ})
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)

# ── main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    if in_venv():
        # We're inside the venv already — just run the app
        exec(open(APP).read(), {"__name__": "__main__", "__file__": str(APP)})
    else:
        ensure_venv()
        relaunch()

