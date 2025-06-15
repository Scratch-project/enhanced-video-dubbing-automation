"""
Enhanced Video Dubbing Automation - Installation and Setup
Kaggle-optimized package installation and model setup
"""

import subprocess
import sys
import os
import json
from pathlib import Path
import torch
import requests
from tqdm import tqdm

from config import config
from utils import Logger, clear_gpu_memory

class EnvironmentSetup:
    """Handle environment setup and package installation for Kaggle"""
    
    def __init__(self):
        self.logger = Logger("setup")
        self.is_kaggle = os.path.exists("/kaggle")
        
    def install_packages(self):
        """Install all required packages with Kaggle-compatible approach"""
        packages = [
            # Core ML packages
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            
            # Audio processing
            "librosa>=0.10.0",
            "noisereduce>=3.0.0",
            "pydub>=0.25.0",
            "scipy>=1.10.0",
            "soundfile>=0.12.0",
            
            # Speech and NLP
            "openai-whisper>=20230314",
            "speechbrain>=0.5.0",
            "sentencepiece>=0.1.99",
            
            # Video processing
            "moviepy>=1.0.3",
            "opencv-python>=4.7.0",
            
            # Synchronization
            "dtw-python>=1.3.0",
            
            # Utilities
            "tqdm>=4.65.0",
            "psutil>=5.9.0",
            "requests>=2.28.0",
            "numpy>=1.24.0",
            
            # HuggingFace Hub
            "huggingface-hub>=0.15.0",
            
            # Git for cloning repositories
            "gitpython>=3.1.0"
        ]
        
        self.logger.info("Starting package installation...")
        
        for package in tqdm(packages, desc="Installing packages"):
            try:
                if self.is_kaggle:
                    # Use --user flag for Kaggle
                    cmd = [sys.executable, "-m", "pip", "install", "--user", package]
                else:
                    cmd = [sys.executable, "-m", "pip", "install", package]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.logger.info(f"Successfully installed {package}")
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to install {package}: {e.stderr}")
                # Try alternative installation methods
                self._try_alternative_installation(package)
    
    def _try_alternative_installation(self, package):
        """Try alternative installation methods for failed packages"""
        try:
            # Try conda if available
            if self.is_kaggle:
                cmd = ["conda", "install", "-c", "conda-forge", package.split(">=")[0], "-y"]
                subprocess.run(cmd, check=True)
                self.logger.info(f"Successfully installed {package} via conda")
        except:
            self.logger.warning(f"Could not install {package} via alternative methods")
    
    def setup_cuda_environment(self):
        """Setup CUDA environment and check GPU availability"""
        self.logger.info("Setting up CUDA environment...")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            self.logger.info(f"CUDA available with {device_count} GPU(s)")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Set memory allocation strategy
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            torch.cuda.empty_cache()
            
        else:
            self.logger.warning("CUDA not available - falling back to CPU processing")
            return False
        
        return True
    
    def download_and_cache_models(self):
        """Download and cache all required models"""
        self.logger.info("Downloading and caching models...")
        
        models_to_cache = [
            {
                "name": "Whisper Large-v3",
                "id": "openai/whisper-large-v3",
                "type": "whisper"
            },
            {
                "name": "SeamlessM4T Large",
                "id": "facebook/hf-seamless-m4t-large",
                "type": "seamless"
            },
            {
                "name": "SpeechBrain Speaker Diarization",
                "id": "speechbrain/spkrec-ecapa-voxceleb",
                "type": "speechbrain"
            }
        ]
        
        for model_info in models_to_cache:
            try:
                self._cache_model(model_info)
            except Exception as e:
                self.logger.error(f"Failed to cache {model_info['name']}: {str(e)}")
    
    def _cache_model(self, model_info):
        """Cache individual model"""
        model_name = model_info["name"]
        model_id = model_info["id"]
        model_type = model_info["type"]
        
        self.logger.info(f"Caching {model_name}...")
        
        if model_type == "whisper":
            import whisper
            # Download Whisper model
            model = whisper.load_model(config.WHISPER_MODEL, 
                                     download_root=str(config.MODELS_DIR))
            del model
            
        elif model_type == "seamless":
            from transformers import SeamlessM4TModel, SeamlessM4TProcessor
            # Cache SeamlessM4T model
            model = SeamlessM4TModel.from_pretrained(
                model_id, 
                cache_dir=str(config.MODELS_DIR),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            processor = SeamlessM4TProcessor.from_pretrained(
                model_id,
                cache_dir=str(config.MODELS_DIR)
            )
            del model, processor
            
        elif model_type == "speechbrain":
            # SpeechBrain models are downloaded on first use
            pass
        
        clear_gpu_memory()
        self.logger.info(f"Successfully cached {model_name}")
    
    def clone_openvoice_repository(self):
        """Clone OpenVoice v2 repository"""
        self.logger.info("Cloning OpenVoice v2 repository...")
        
        openvoice_dir = config.WORKING_DIR / "OpenVoice"
        
        if openvoice_dir.exists():
            self.logger.info("OpenVoice repository already exists")
            return
        
        try:
            import git
            repo = git.Repo.clone_from(
                "https://github.com/myshell-ai/OpenVoice.git",
                openvoice_dir
            )
            self.logger.info("Successfully cloned OpenVoice repository")
            
            # Install OpenVoice requirements
            requirements_file = openvoice_dir / "requirements.txt"
            if requirements_file.exists():
                cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
                if self.is_kaggle:
                    cmd.insert(4, "--user")
                subprocess.run(cmd, check=True)
                self.logger.info("Installed OpenVoice requirements")
                
        except Exception as e:
            self.logger.error(f"Failed to clone OpenVoice repository: {str(e)}")
    
    def verify_installation(self):
        """Verify that all components are properly installed"""
        self.logger.info("Verifying installation...")
        
        checks = []
        
        # Check PyTorch and CUDA
        try:
            import torch
            checks.append(("PyTorch", torch.__version__, True))
            checks.append(("CUDA Available", torch.cuda.is_available(), torch.cuda.is_available()))
        except ImportError:
            checks.append(("PyTorch", "Not installed", False))
        
        # Check core packages
        packages_to_check = [
            ("transformers", "transformers"),
            ("whisper", "openai-whisper"),
            ("librosa", "librosa"),
            ("moviepy", "moviepy.editor"),
            ("speechbrain", "speechbrain"),
            ("noisereduce", "noisereduce"),
            ("dtw", "dtw"),
        ]
        
        for package_name, import_name in packages_to_check:
            try:
                __import__(import_name)
                checks.append((package_name, "Installed", True))
            except ImportError:
                checks.append((package_name, "Not installed", False))
        
        # Check ffmpeg
        try:
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, text=True, check=True)
            checks.append(("ffmpeg", "Available", True))
        except:
            checks.append(("ffmpeg", "Not available", False))
        
        # Print verification results
        self.logger.info("Installation Verification Results:")
        for name, status, success in checks:
            status_symbol = "✓" if success else "✗"
            self.logger.info(f"{status_symbol} {name}: {status}")
        
        # Check model cache
        if config.MODELS_DIR.exists():
            cached_files = list(config.MODELS_DIR.glob("**/*"))
            self.logger.info(f"Model cache directory contains {len(cached_files)} files")
        
        return all(check[2] for check in checks)
    
    def run_full_setup(self):
        """Run complete environment setup"""
        self.logger.info("Starting enhanced video dubbing environment setup...")
        
        # Install packages
        self.install_packages()
        
        # Setup CUDA
        cuda_available = self.setup_cuda_environment()
        
        # Download models
        if cuda_available:
            self.download_and_cache_models()
        
        # Clone OpenVoice
        self.clone_openvoice_repository()
        
        # Verify installation
        success = self.verify_installation()
        
        if success:
            self.logger.info("Environment setup completed successfully!")
        else:
            self.logger.error("Environment setup completed with some issues")
        
        return success

# Main setup function
def setup_environment():
    """Main function to setup the environment"""
    setup = EnvironmentSetup()
    return setup.run_full_setup()

if __name__ == "__main__":
    setup_environment()
