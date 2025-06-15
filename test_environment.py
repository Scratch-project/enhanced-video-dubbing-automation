"""
Environment Validation and Testing Module
Enhanced Video Dubbing Automation Project
"""

import os
import sys
import subprocess
import importlib
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import time

from config import config
from utils import Logger

class EnvironmentValidator:
    """Validate the complete environment setup for video dubbing"""
    
    def __init__(self, local_mode: bool = False):
        config.__init__(local_mode)
        self.logger = Logger("environment_validator")
        self.validation_results = {}
        
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete environment validation"""
        print("🔍 Starting Environment Validation...")
        print("=" * 60)
        
        # System checks
        self.check_system_info()
        self.check_gpu_availability()
        self.check_disk_space()
        
        # Package checks
        self.check_required_packages()
        self.check_model_availability()
        
        # Directory structure
        self.check_directory_structure()
        
        # Tool availability
        self.check_external_tools()
        
        # Generate report
        self.generate_validation_report()
        
        return self.validation_results
    
    def check_system_info(self):
        """Check basic system information"""
        print("\n📋 System Information")
        print("-" * 30)
        
        try:
            import platform
            import psutil
            
            system_info = {
                "platform": platform.platform(),
                "python_version": sys.version.split()[0],
                "cpu_count": os.cpu_count(),
                "total_memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
                "available_memory_gb": round(psutil.virtual_memory().available / 1024**3, 2)
            }
            
            for key, value in system_info.items():
                print(f"  {key}: {value}")
            
            self.validation_results["system_info"] = system_info
            self.validation_results["system_check"] = "✅ PASSED"
            
        except Exception as e:
            print(f"  ❌ System check failed: {e}")
            self.validation_results["system_check"] = f"❌ FAILED: {e}"
    
    def check_gpu_availability(self):
        """Check GPU availability and CUDA setup"""
        print("\n🖥️  GPU & CUDA Information")
        print("-" * 30)
        
        try:
            if torch.cuda.is_available():
                gpu_info = {
                    "cuda_available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(),
                    "memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
                    "memory_cached_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
                    "cuda_version": torch.version.cuda
                }
                
                print(f"  ✅ CUDA Available: {gpu_info['device_count']} device(s)")
                print(f"  Device: {gpu_info['device_name']}")
                print(f"  CUDA Version: {gpu_info['cuda_version']}")
                print(f"  Memory: {gpu_info['memory_allocated_mb']}MB allocated")
                
                self.validation_results["gpu_info"] = gpu_info
                self.validation_results["gpu_check"] = "✅ PASSED"
                
            else:
                print("  ⚠️  CUDA not available - will use CPU (slower)")
                self.validation_results["gpu_info"] = {"cuda_available": False}
                self.validation_results["gpu_check"] = "⚠️  CPU ONLY"
                
        except Exception as e:
            print(f"  ❌ GPU check failed: {e}")
            self.validation_results["gpu_check"] = f"❌ FAILED: {e}"
    
    def check_disk_space(self):
        """Check available disk space"""
        print("\n💾 Disk Space Information")
        print("-" * 30)
        
        try:
            import shutil
            
            total, used, free = shutil.disk_usage(config.BASE_DIR)
            free_gb = free / 1024**3
            
            print(f"  Free space: {free_gb:.1f} GB")
            
            if free_gb > 50:
                print("  ✅ Sufficient disk space available")
                self.validation_results["disk_check"] = "✅ PASSED"
            elif free_gb > 20:
                print("  ⚠️  Limited disk space - may affect large videos")
                self.validation_results["disk_check"] = "⚠️  LIMITED"
            else:
                print("  ❌ Insufficient disk space for video processing")
                self.validation_results["disk_check"] = "❌ INSUFFICIENT"
            
            self.validation_results["disk_info"] = {"free_gb": free_gb}
            
        except Exception as e:
            print(f"  ❌ Disk check failed: {e}")
            self.validation_results["disk_check"] = f"❌ FAILED: {e}"
    
    def check_required_packages(self):
        """Check if all required packages are installed"""
        print("\n📦 Package Dependencies")
        print("-" * 30)
        
        required_packages = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("whisper", "OpenAI Whisper"),
            ("librosa", "Librosa"),
            ("noisereduce", "NoiseReduce"),
            ("speechbrain", "SpeechBrain"),
            ("moviepy", "MoviePy"),
            ("soundfile", "SoundFile"),
            ("psutil", "PSUtil"),
            ("git", "GitPython")
        ]
        
        package_status = {}
        all_passed = True
        
        for package, display_name in required_packages:
            try:
                if package == "git":
                    import git
                else:
                    importlib.import_module(package)
                print(f"  ✅ {display_name}")
                package_status[package] = "✅ INSTALLED"
            except ImportError:
                print(f"  ❌ {display_name} - NOT INSTALLED")
                package_status[package] = "❌ MISSING"
                all_passed = False
        
        self.validation_results["package_status"] = package_status
        self.validation_results["package_check"] = "✅ ALL PASSED" if all_passed else "❌ MISSING PACKAGES"
    
    def check_model_availability(self):
        """Check if required models can be loaded"""
        print("\n🤖 Model Availability")
        print("-" * 30)
        
        model_checks = {}
        
        # Test Whisper model loading
        try:
            import whisper
            print("  📝 Testing Whisper model loading...")
            model = whisper.load_model("base")
            print("  ✅ Whisper: Base model loaded successfully")
            model_checks["whisper"] = "✅ AVAILABLE"
            del model  # Free memory
        except Exception as e:
            print(f"  ❌ Whisper: {e}")
            model_checks["whisper"] = f"❌ FAILED: {e}"
        
        # Test Transformers model access
        try:
            from transformers import AutoTokenizer
            print("  🔄 Testing Transformers model access...")
            tokenizer = AutoTokenizer.from_pretrained("facebook/seamless-m4t-medium", use_auth_token=False)
            print("  ✅ SeamlessM4T: Model access successful")
            model_checks["seamless"] = "✅ AVAILABLE"
        except Exception as e:
            print(f"  ⚠️  SeamlessM4T: {e} (will download on first use)")
            model_checks["seamless"] = f"⚠️  WILL_DOWNLOAD: {e}"
        
        self.validation_results["model_checks"] = model_checks
    
    def check_directory_structure(self):
        """Validate directory structure"""
        print("\n📁 Directory Structure")
        print("-" * 30)
        
        required_dirs = [
            config.MODELS_DIR,
            config.TEMP_DIR,
            config.OUTPUT_DIR,
            config.LOGS_DIR,
            config.CHECKPOINTS_DIR
        ]
        
        all_created = True
        for directory in required_dirs:
            if directory.exists():
                print(f"  ✅ {directory.name}/")
            else:
                print(f"  📁 Creating {directory.name}/")
                directory.mkdir(parents=True, exist_ok=True)
                all_created = True
        
        self.validation_results["directory_check"] = "✅ ALL CREATED" if all_created else "❌ CREATION FAILED"
    
    def check_external_tools(self):
        """Check external tool availability"""
        print("\n🔧 External Tools")
        print("-" * 30)
        
        tools = {
            "ffmpeg": ["ffmpeg", "-version"],
            "git": ["git", "--version"]
        }
        
        tool_status = {}
        
        for tool, cmd in tools.items():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    version = result.stdout.split('\n')[0]
                    print(f"  ✅ {tool}: {version}")
                    tool_status[tool] = "✅ AVAILABLE"
                else:
                    print(f"  ❌ {tool}: Command failed")
                    tool_status[tool] = "❌ FAILED"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print(f"  ❌ {tool}: Not found or timeout")
                tool_status[tool] = "❌ NOT_FOUND"
        
        self.validation_results["tool_status"] = tool_status
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n📊 Validation Summary")
        print("=" * 60)
        
        # Count statuses
        total_checks = 0
        passed_checks = 0
        
        for key, value in self.validation_results.items():
            if key.endswith("_check"):
                total_checks += 1
                if "✅" in str(value):
                    passed_checks += 1
        
        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        print(f"Overall Status: {passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("🎉 Environment is ready for video dubbing!")
            status = "READY"
        elif success_rate >= 70:
            print("⚠️  Environment has some issues but may work with limitations")
            status = "PARTIAL"
        else:
            print("❌ Environment needs significant fixes before use")
            status = "NOT_READY"
        
        self.validation_results["overall_status"] = status
        self.validation_results["success_rate"] = success_rate
        
        # Save report
        report_path = config.LOGS_DIR / f"environment_validation_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: {report_path}")
        
        return status

def run_quick_test():
    """Run a quick environment validation"""
    validator = EnvironmentValidator(local_mode=True)
    return validator.run_full_validation()

if __name__ == "__main__":
    run_quick_test()
