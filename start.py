"""
start.py - CreditIQ launcher
All setup logic lives here so we avoid Windows batch file quirks entirely.
"""

import sys
import os
import subprocess
import time
import webbrowser

# ── Make sure output shows immediately ────────────────────────────────────────
os.environ["PYTHONUNBUFFERED"] = "1"

ROOT    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")

def banner():
    print()
    print("  ============================================================")
    print("   CreditIQ -- Credit Score Platform")
    print("  ============================================================")
    print()

def step(n, msg):
    print(f"  [{n}] {msg}")

def ok(msg):
    print(f"      OK: {msg}")

def err(msg):
    print()
    print(f"  ERROR: {msg}")
    print()

def run(cmd, **kwargs):
    return subprocess.run(cmd, **kwargs)

# ─────────────────────────────────────────────────────────────────────────────
banner()

# ── Python version ────────────────────────────────────────────────────────────
ver = sys.version_info
ok(f"Python {ver.major}.{ver.minor}.{ver.micro} at {sys.executable}")

if ver.major < 3 or (ver.major == 3 and ver.minor < 9):
    err("Python 3.9 or newer required. Download from https://python.org")
    input("Press Enter to exit...")
    sys.exit(1)

# ── Upgrade pip ───────────────────────────────────────────────────────────────
step(1, "Upgrading pip...")
run([sys.executable, "-m", "pip", "install", "--upgrade", "pip",
     "--quiet", "--disable-pip-version-check"])
ok("pip ready")

# ── Install packages ──────────────────────────────────────────────────────────
step(2, "Installing packages (first run: ~1-2 minutes)...")

packages = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.20.0",
    "joblib>=1.2.0",
    "python-multipart>=0.0.9",
]

heavy = [
    "numpy>=1.26.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
]

# Install light packages first
run([sys.executable, "-m", "pip", "install"] + packages +
    ["--quiet", "--disable-pip-version-check"])

# Install heavy packages - prefer binary wheels to avoid compilation
print("      Installing numpy/pandas/scikit-learn (binary wheels)...")
r = run([sys.executable, "-m", "pip", "install"] + heavy +
        ["--only-binary=:all:", "--quiet", "--disable-pip-version-check"])

if r.returncode != 0:
    print("      Binary install failed, trying standard install...")
    run([sys.executable, "-m", "pip", "install"] + heavy +
        ["--quiet", "--disable-pip-version-check"])

# Verify
try:
    import importlib
    for pkg in ["fastapi", "uvicorn", "sklearn", "numpy", "pandas", "joblib"]:
        importlib.import_module(pkg)
    ok("All packages installed")
except ImportError as e:
    err(f"Package missing: {e}")
    print("  Try installing Python 3.12 from:")
    print("  https://www.python.org/downloads/release/python-3120/")
    print()
    input("Press Enter to exit...")
    sys.exit(1)

# ── Check / train models ──────────────────────────────────────────────────────
step(3, "Checking ML models...")

os.chdir(BACKEND)
model_path = os.path.join(BACKEND, "models", "risk_model.pkl")

if not os.path.exists(model_path):
    print("      Models not found - training now (~30-60 seconds)...")
    r = run([sys.executable, "train_model.py"])
    if r.returncode != 0:
        err("Model training failed. See error above.")
        input("Press Enter to exit...")
        sys.exit(1)

ok("Models ready")

# ── Start uvicorn server ──────────────────────────────────────────────────────
print()
print("  Starting server...")
print()

server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app",
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
    cwd=BACKEND,
)

# Wait and check it started
time.sleep(3)
if server.poll() is not None:
    err("Server crashed immediately. Check error above.")
    input("Press Enter to exit...")
    sys.exit(1)

# ── Open browser ──────────────────────────────────────────────────────────────
url = "http://localhost:8000"
webbrowser.open(url)

print("  ============================================================")
print(f"   App:      {url}")
print(f"   API docs: {url}/docs")
print("  ============================================================")
print()
print("  Server is running. Press Ctrl+C to stop.")
print()

try:
    server.wait()
except KeyboardInterrupt:
    print()
    print("  Stopping server...")
    server.terminate()
    server.wait()
    print("  Stopped. Goodbye.")
