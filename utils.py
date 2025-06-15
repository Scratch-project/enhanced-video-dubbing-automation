"""
Utility functions for the Enhanced Video Dubbing Automation Project
"""

import os
import json
import logging
import time
import psutil
import torch
import gc
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import subprocess
import shutil

from config import config

class Logger:
    """Enhanced logging system with resource monitoring"""
    
    def __init__(self, video_name: str, log_level: str = "INFO"):
        self.video_name = video_name
        self.log_file = config.get_log_path(video_name)
        self.setup_logger(log_level)
    
    def setup_logger(self, log_level: str):
        """Setup logging configuration"""
        self.logger = logging.getLogger(f"dubbing_{self.video_name}")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def log_step_start(self, step_name: str):
        """Log the start of a processing step"""
        self.logger.info(f"Starting {step_name}")
        self.logger.info(f"GPU Memory: {get_gpu_memory_usage()}")
        self.logger.info(f"System Memory: {get_system_memory_usage()}")
        self.logger.info(f"Disk Space: {get_disk_space()}")
    
    def log_step_end(self, step_name: str, success: bool = True):
        """Log the end of a processing step"""
        status = "completed successfully" if success else "failed"
        self.logger.info(f"{step_name} {status}")
        self.logger.info(f"GPU Memory after step: {get_gpu_memory_usage()}")
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)

class CheckpointManager:
    """Manage processing checkpoints for session recovery"""
    
    def __init__(self, video_name: str):
        self.video_name = video_name
        self.checkpoint_file = config.CHECKPOINTS_DIR / f"{video_name}_checkpoint.json"
    
    def save_checkpoint(self, step: str, data: Dict[str, Any]):
        """Save checkpoint data"""
        checkpoint_data = {
            "video_name": self.video_name,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "system_info": {
                "gpu_memory": get_gpu_memory_usage(),
                "system_memory": get_system_memory_usage(),
                "disk_space": get_disk_space()
            }
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint data if exists"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_completed_steps(self) -> List[str]:
        """Get list of completed steps"""
        checkpoint = self.load_checkpoint()
        if checkpoint:
            return checkpoint.get("completed_steps", [])
        return []
    
    def mark_step_completed(self, step: str):
        """Mark a step as completed"""
        checkpoint = self.load_checkpoint() or {}
        completed_steps = checkpoint.get("completed_steps", [])
        if step not in completed_steps:
            completed_steps.append(step)
        checkpoint["completed_steps"] = completed_steps
        checkpoint["last_completed_step"] = step
        checkpoint["last_update"] = datetime.now().isoformat()
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

def get_gpu_memory_usage() -> str:
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return f"{allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.2f}GB total"
    return "CUDA not available"

def get_system_memory_usage() -> str:
    """Get current system memory usage"""
    memory = psutil.virtual_memory()
    used_gb = memory.used / 1024**3
    total_gb = memory.total / 1024**3
    percent = memory.percent
    return f"{used_gb:.2f}GB / {total_gb:.2f}GB ({percent:.1f}%)"

def get_disk_space() -> str:
    """Get current disk space usage"""
    try:
        if os.path.exists("/kaggle/working"):
            disk_usage = shutil.disk_usage("/kaggle/working")
        else:
            disk_usage = shutil.disk_usage(".")
        
        free_gb = disk_usage.free / 1024**3
        total_gb = disk_usage.total / 1024**3
        used_gb = (disk_usage.total - disk_usage.free) / 1024**3
        
        return f"{used_gb:.2f}GB used, {free_gb:.2f}GB free, {total_gb:.2f}GB total"
    except Exception as e:
        return f"Error getting disk space: {str(e)}"

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def check_system_resources() -> Dict[str, bool]:
    """Check if system resources are within acceptable limits"""
    warnings = {}
    
    # Check GPU memory
    if torch.cuda.is_available():
        allocated_ratio = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        warnings["gpu_memory_high"] = allocated_ratio > config.GPU_MEMORY_FRACTION
    
    # Check system memory
    memory = psutil.virtual_memory()
    warnings["system_memory_high"] = memory.percent > (config.MEMORY_WARNING_THRESHOLD * 100)
    
    # Check disk space
    try:
        if os.path.exists("/kaggle/working"):
            disk_usage = shutil.disk_usage("/kaggle/working")
        else:
            disk_usage = shutil.disk_usage(".")
        
        free_gb = disk_usage.free / 1024**3
        warnings["disk_space_low"] = free_gb < config.DISK_SPACE_WARNING_THRESHOLD
    except:
        warnings["disk_space_check_failed"] = True
    
    return warnings

def validate_video_file(file_path: Path) -> bool:
    """Validate that the video file exists and is supported"""
    if not file_path.exists():
        return False
    
    if file_path.suffix.lower() not in config.SUPPORTED_VIDEO_FORMATS:
        return False
    
    # Check file size
    file_size_gb = file_path.stat().st_size / 1024**3
    if file_size_gb > config.MAX_FILE_SIZE_GB:
        return False
    
    return True

def get_video_info(file_path: Path) -> Dict[str, Any]:
    """Get video file information using ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}

def estimate_processing_time(video_duration_minutes: float) -> Dict[str, float]:
    """Estimate processing time for each step (in minutes)"""
    # These are rough estimates based on typical processing speeds
    estimates = {
        "audio_extraction": video_duration_minutes * 0.1,
        "transcription": video_duration_minutes * 0.5,
        "translation": video_duration_minutes * 0.3,
        "voice_synthesis": video_duration_minutes * 2.0,
        "synchronization": video_duration_minutes * 0.5,
        "subtitle_generation": video_duration_minutes * 0.2,
        "final_assembly": video_duration_minutes * 0.8,
    }
    
    estimates["total"] = sum(estimates.values())
    return estimates

def create_processing_report(video_name: str, start_time: datetime, 
                           end_time: datetime, status: str, 
                           step_times: Dict[str, float], 
                           errors: List[str] = None) -> Dict[str, Any]:
    """Create a comprehensive processing report"""
    duration = (end_time - start_time).total_seconds() / 60  # minutes
    
    report = {
        "video_name": video_name,
        "processing_start": start_time.isoformat(),
        "processing_end": end_time.isoformat(),
        "total_duration_minutes": duration,
        "status": status,
        "step_times": step_times,
        "errors": errors or [],
        "output_files": [],
        "quality_metrics": {},
        "system_info": {
            "final_gpu_memory": get_gpu_memory_usage(),
            "final_system_memory": get_system_memory_usage(),
            "final_disk_space": get_disk_space()
        }
    }
    
    # Save report
    report_path = config.OUTPUT_DIR / video_name / "processing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def retry_on_failure(max_retries: int = 3, delay: int = 60):
    """Decorator for retrying failed operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    clear_gpu_memory()  # Clear memory before retry
            return None
        return wrapper
    return decorator

def safe_model_loading(model_load_func):
    """Safely load models with memory management"""
    def wrapper(*args, **kwargs):
        clear_gpu_memory()
        try:
            model = model_load_func(*args, **kwargs)
            return model
        except Exception as e:
            clear_gpu_memory()
            raise e
    return wrapper
