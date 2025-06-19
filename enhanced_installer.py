# ────────────────────────────────────────────────────────────────────────────────
#  ENHANCED FULL STACK INSTALLER v2.0 (Kaggle/Colab, Py 3.10-3.12) — video dubbing
# ────────────────────────────────────────────────────────────────────────────────
#  NEW: GPU detection • retry logic • better error recovery • updated versions
#  IMPROVED: Kaggle compatibility • package conflict resolution • disk space check
# ────────────────────────────────────────────────────────────────────────────────

import subprocess, sys, importlib, inspect, pathlib, re, types, os, time, shutil
from datetime import datetime

# Environment detection
IS_KAGGLE = any("/kaggle" in p for p in sys.path) or os.path.exists('/kaggle')
IS_COLAB = 'google.colab' in sys.modules
IS_LOCAL = not (IS_KAGGLE or IS_COLAB)
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

print(f"🔧 Enhanced Video Dubbing Installer v2.0")
print(f"📍 Environment: {'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'Local'}")
print(f"🐍 Python: {PYTHON_VERSION}")
print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------------------------------------------ Helper functions
def sh(cmd, check=True, timeout=300):
    """Execute shell command with better error handling"""
    print(f"\n$ {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True, timeout=timeout)
        if result.stderr and "warning" not in result.stderr.lower():
            print(f"⚠️  {result.stderr.strip()}")
        return result
    except subprocess.TimeoutExpired:
        print(f"⏱️  Command timed out after {timeout}s")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed (exit {e.returncode}): {e.stderr}")
        if check:
            raise
        return e

def check_disk_space(required_gb=5):
    """Check available disk space"""
    try:
        total, used, free = shutil.disk_usage('/')
        free_gb = free / (1024**3)
        print(f"💾 Disk space: {free_gb:.1f}GB free")
        if free_gb < required_gb:
            print(f"⚠️  Low disk space! {required_gb}GB recommended")
            return False
        return True
    except:
        return True  # Assume OK if check fails

def detect_gpu():
    """Detect GPU and CUDA availability"""
    gpu_info = {"has_gpu": False, "cuda_version": None, "gpu_name": None}
    
    try:
        result = sh("nvidia-smi --query-gpu=name --format=csv,noheader", check=False)
        if result and result.returncode == 0:
            gpu_info["has_gpu"] = True
            gpu_info["gpu_name"] = result.stdout.strip().split('\n')[0]
            
            # Get CUDA version
            cuda_result = sh("nvcc --version", check=False)
            if cuda_result and "release" in cuda_result.stdout:
                cuda_match = re.search(r'release (\d+\.\d+)', cuda_result.stdout)
                if cuda_match:
                    gpu_info["cuda_version"] = cuda_match.group(1)
    except:
        pass
    
    if gpu_info["has_gpu"]:
        print(f"🖥️  GPU: {gpu_info['gpu_name']}")
        print(f"🔧 CUDA: {gpu_info['cuda_version'] or 'Unknown'}")
    else:
        print("💻 CPU-only environment detected")
    
    return gpu_info

def get_optimal_torch_version(gpu_info):
    """Get optimal PyTorch version based on CUDA"""
    if not gpu_info["has_gpu"]:
        return "torch>=2.0.0", "torchaudio>=2.0.0"
    
    cuda_ver = gpu_info.get("cuda_version", "")
    if cuda_ver and cuda_ver.startswith("12"):
        return "torch>=2.1.0", "torchaudio>=2.1.0"
    elif cuda_ver and cuda_ver.startswith("11"):
        return "torch>=2.0.0,<2.2.0", "torchaudio>=2.0.0,<2.2.0"
    else:
        return "torch>=2.0.0", "torchaudio>=2.0.0"

# ------------------------------------------------------------------ Pre-installation checks
print(f"\n🔍 Pre-installation checks...")
check_disk_space(10)  # Require 10GB free space
gpu_info = detect_gpu()

# 0️⃣  System dependencies
print(f"\n📦 Installing system dependencies...")
if not IS_LOCAL:
    sh("apt-get -qq update")
    sh("apt-get -qq install -y ffmpeg git")
    
    # Additional dependencies for audio/video processing
    sh("apt-get -qq install -y libsndfile1 libsox-fmt-all")

# 1️⃣  Updated compatibility pins for 2025
print(f"\n🔧 Setting up compatibility pins...")

# More conservative numpy pinning for better stability
BASE_PINS = [
    "numpy>=1.24.0,<2.0.0",     # Conservative for max compatibility
    "scipy>=1.10.0,<1.14.0",    # Updated range
    "setuptools>=65.0.0",       # Fix for newer pip
    "wheel>=0.38.0",            # Ensure wheel support
]

# Environment-specific pins
if PYTHON_VERSION >= "3.11":
    BASE_PINS.extend([
        "matplotlib>=3.6.0,<3.10.0",  # Updated range for Py 3.11+
        "pillow>=9.0.0,<11.0.0",      # Updated pillow range
    ])
else:
    BASE_PINS.extend([
        "matplotlib>=3.5.0,<3.9.0",
        "pillow>=8.0.0,<10.0.0",
    ])

# Install base pins with retries
for pin in BASE_PINS:
    for attempt in range(3):
        try:
            flags = ["--user", "--no-warn-script-location"] if not IS_LOCAL else []
            cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *flags, pin]
            subprocess.run(cmd, check=True, timeout=120)
            print(f"✅ {pin}")
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 2:
                print(f"⚠️  Failed to install {pin} after 3 attempts")
            else:
                print(f"🔄 Retry {attempt + 1}/3 for {pin}")
                time.sleep(2)

# Handle numba compatibility
try:
    import numba
    print(f"🔧 Adjusting numba for compatibility...")
    sh("python -m pip install -q 'numba>=0.58.0,<0.61.0'", check=False)
except ImportError:
    pass

# 2️⃣  Enhanced purging with better error handling
print(f"\n🧹 Cleaning conflicting packages...")
PURGE_PACKAGES = [
    "speechbrain", "whisper", "openai-whisper", "dtw", "dtw-python", "fastdtw",
    "noisereduce", "hyperpyyaml", "ruamel.yaml", "gitpython", "opencv-python"
]

for pkg in PURGE_PACKAGES:
    sh(f"python -m pip uninstall -y -q {pkg}", check=False)

# Clear pip cache
sh("python -m pip cache purge", check=False)

# 3️⃣  Enhanced mappings and dependencies
IMPORT_ALIAS = {
    "openai-whisper": "whisper",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "dtw-python": "dtw",
    "gitpython": "git",
    "ruamel-yaml": "ruamel.yaml",
}

EXTRA_DEPS = {
    "speechbrain": ["hyperpyyaml", "ruamel.yaml"],
    "hyperpyyaml": ["ruamel.yaml"],
    "moviepy": ["imageio>=2.25.0", "imageio-ffmpeg>=0.4.8"],
    "noisereduce": ["scipy>=1.8.0"],
}

# Alternative packages if primary fails
ALTERNATIVES = {
    "speechbrain": ["speechbrain-nightly", "speech-recognition"],
    "dtw-python": ["fastdtw", "dtaidistance"],
    "noisereduce": ["spectral-subtract", "pyrubberband"],
    "moviepy": ["ffmpeg-python", "opencv-python"],
}

# 4️⃣  Updated package specifications for 2025
torch_spec, torchaudio_spec = get_optimal_torch_version(gpu_info)

ESSENTIAL = [
    torch_spec,
    torchaudio_spec,
    "transformers>=4.35.0",         # Updated for better performance
    "accelerate>=0.25.0",           # Updated version
    "librosa>=0.10.1",              # Latest stable
    "soundfile>=0.12.1",            # Updated
    "moviepy>=1.0.3",
    "openai-whisper>=20231117",
    "psutil>=5.9.0",
    "tqdm>=4.66.0",                 # Updated
    "requests>=2.31.0",             # Security updates
    "packaging>=23.0",              # Dependency resolution
]

# Add numpy/scipy pins to ensure they stick
ESSENTIAL.extend([
    "numpy>=1.24.0,<2.0.0",
    "scipy>=1.10.0,<1.14.0",
])

OPTIONAL = [
    "noisereduce>=3.0.0",
    "speechbrain>=0.5.16",          # Latest stable
    "dtw-python>=1.3.0",
    "gitpython>=3.1.40",            # Security updates
    "seaborn>=0.13.0",              # Updated
    "plotly>=5.17.0",               # For better visualization
]

# ---------------------------------------------------------------- Enhanced helpers
def base_pkg(spec: str) -> str:
    """Return 'package' from 'package<=1.2.3' / 'package>=...' / 'package'."""
    return re.split(r"[<>=!~]", spec, maxsplit=1)[0]

def import_name(pkg: str) -> str:
    return IMPORT_ALIAS.get(pkg, pkg.replace("-", "_"))

def unshadow(mod: str):
    """Remove dataset folders that shadow real packages."""
    removed_paths = []
    for p in list(sys.path):
        d = pathlib.Path(p) / mod
        if d.is_dir() and not (d / "__init__.py").exists():
            sys.path.remove(p)
            removed_paths.append(p)
    
    if removed_paths:
        print(f"⚠️  Removed {len(removed_paths)} shadowing paths for {mod}")

def pip_install_with_retry(spec: str, max_retries=3):
    """Install package with retry logic and fallbacks"""
    flags = ["--user", "--no-warn-script-location"] if not IS_LOCAL else []
    base_cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade"]
    
    for attempt in range(max_retries):
        try:
            cmd = base_cmd + flags + [spec]
            subprocess.run(cmd, check=True, timeout=300)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            if attempt == max_retries - 1:
                print(f"❌ Failed after {max_retries} attempts: {e}")
                return False
            else:
                print(f"🔄 Retry {attempt + 1}/{max_retries} for {spec}")
                time.sleep(5)  # Wait longer between retries
    
    return False

def try_alternatives(pkg: str) -> bool:
    """Try alternative packages if primary fails"""
    if pkg not in ALTERNATIVES:
        return False
    
    print(f"🔄 Trying alternatives for {pkg}...")
    for alt in ALTERNATIVES[pkg]:
        print(f"   📦 Attempting {alt}...")
        if pip_install_with_retry(alt):
            print(f"   ✅ {alt} installed as fallback")
            return True
    return False

def ensure_enhanced(spec: str, is_essential=True) -> bool:
    """Enhanced package installation with better error handling"""
    pkg = base_pkg(spec)
    mod = import_name(pkg)
    
    # Clear import cache and unshadow
    if mod in sys.modules:
        del sys.modules[mod]
    unshadow(mod)
    
    try:
        importlib.import_module(mod)
        print(f"✅ {pkg:20s} → {mod} (already ok)")
        return True
    except ImportError:
        pass

    print(f"📦 Installing {pkg}...")
    
    # Install main package
    if not pip_install_with_retry(spec):
        if is_essential:
            return False
        else:
            return try_alternatives(pkg)
    
    # Install extra dependencies
    for dep in EXTRA_DEPS.get(pkg, []):
        print(f"   ➕ Installing dependency: {dep}")
        pip_install_with_retry(dep)
    
    # Clear cache and test import
    unshadow(mod)
    if mod in sys.modules:
        del sys.modules[mod]
    
    try:
        importlib.import_module(mod)
        print(f"✅ {pkg:20s} → {mod} (installed)")
        return True
    except ImportError as e:
        print(f"⚠️  Import failed for {pkg}: {e}")
        if not is_essential:
            return try_alternatives(pkg)
        return False

# 5️⃣  Enhanced installation with progress tracking
print(f"\n🚀 Installing packages...")
failed_essential, failed_optional = [], []

print(f"\n🔧 Installing {len(ESSENTIAL)} essential packages...")
for i, spec in enumerate(ESSENTIAL, 1):
    print(f"[{i}/{len(ESSENTIAL)}] {base_pkg(spec)}")
    if not ensure_enhanced(spec, is_essential=True):
        failed_essential.append(spec)
    
    # Memory cleanup for large packages
    if base_pkg(spec) in ["torch", "transformers", "speechbrain"]:
        import gc
        gc.collect()

print(f"\n🔧 Installing {len(OPTIONAL)} optional packages...")
for i, spec in enumerate(OPTIONAL, 1):
    print(f"[{i}/{len(OPTIONAL)}] {base_pkg(spec)}")
    if not ensure_enhanced(spec, is_essential=False):
        failed_optional.append(spec)

# 6️⃣  Enhanced DTW fallback with multiple options
print(f"\n🔧 Ensuring DTW functionality...")
dtw_available = False

try:
    import dtw
    dtw_available = True
    print("✅ dtw-python available")
except ImportError:
    # Try fastdtw
    try:
        if pip_install_with_retry("fastdtw>=0.3.0"):
            import fastdtw
            # Create alias
            alias = types.ModuleType("dtw")
            alias.distance = fastdtw.fastdtw
            alias.dtw = fastdtw.fastdtw
            sys.modules["dtw"] = alias
            dtw_available = True
            print("✅ fastdtw installed and aliased as dtw")
    except:
        pass

if not dtw_available:
    print("⚠️  No DTW library available - synchronization features limited")

# 7️⃣  Enhanced verification with GPU testing
CRITICAL_TESTS = {
    "whisper": lambda: __import__("whisper").load_model("base"),
    "torch": lambda: __import__("torch").cuda.is_available() if gpu_info["has_gpu"] else True,
    "transformers": lambda: __import__("transformers").__version__,
    "librosa": lambda: __import__("librosa").stft([1,2,3,4]),
    "moviepy": lambda: __import__("moviepy.editor").__version__,
}

print(f"\n🧪 Running enhanced verification tests...")
verification_passed = True

for module, test_func in CRITICAL_TESTS.items():
    try:
        unshadow(module)
        result = test_func()
        if module == "torch" and gpu_info["has_gpu"]:
            print(f"   ✅ {module} (CUDA: {result})")
        else:
            print(f"   ✅ {module}")
    except Exception as e:
        print(f"   ❌ {module}: {e}")
        verification_passed = False

# 8️⃣  Enhanced summary with recommendations
print(f"\n📊 INSTALLATION SUMMARY")
print(f"═" * 50)
print(f"🟢 Essential: {len(ESSENTIAL)-len(failed_essential)}/{len(ESSENTIAL)} installed")
print(f"🟡 Optional:  {len(OPTIONAL)-len(failed_optional)}/{len(OPTIONAL)} installed")
print(f"🧪 Critical tests: {'✅ PASSED' if verification_passed else '❌ FAILED'}")

if failed_essential:
    print(f"\n❌ FAILED ESSENTIAL PACKAGES:")
    for spec in failed_essential:
        print(f"   • {base_pkg(spec)}")

if failed_optional:
    print(f"\n⚠️  FAILED OPTIONAL PACKAGES:")
    for spec in failed_optional:
        print(f"   • {base_pkg(spec)}")

# Memory usage info
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"\n💾 Memory usage: {mem.percent}% ({mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB)")
except:
    pass

print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Final decision
if failed_essential or not verification_passed:
    print(f"\n🛑 ENVIRONMENT NOT READY")
    print(f"   Please check the failed packages above and retry.")
    if IS_KAGGLE:
        print(f"   💡 Try restarting the Kaggle kernel and running again.")
    raise RuntimeError("Critical packages failed to install or verify.")
else:
    print(f"\n🎉 ENVIRONMENT READY!")
    print(f"   All critical packages installed and verified.")
    print(f"   🚀 Ready for enhanced video dubbing!")
