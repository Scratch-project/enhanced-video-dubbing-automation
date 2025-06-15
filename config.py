"""
Enhanced Video Dubbing Automation Project - Configuration
Kaggle-optimized settings for Arabic to English/German video dubbing
"""

import os
from pathlib import Path

class Config:
    """Configuration class for the video dubbing automation project"""
    
    # Kaggle Environment Settings
    KAGGLE_WORKING_DIR = Path("/kaggle/working")
    KAGGLE_INPUT_DIR = Path("/kaggle/input")
    
    # Local Development Override (for testing outside Kaggle)
    LOCAL_WORKING_DIR = Path("./working")
    LOCAL_INPUT_DIR = Path("./input")
    
    def __init__(self, local_mode=False):
        self.local_mode = local_mode
        self.setup_directories()
    
    def setup_directories(self):
        """Setup directory paths based on environment"""
        if self.local_mode or not os.path.exists("/kaggle"):
            self.WORKING_DIR = self.LOCAL_WORKING_DIR
            self.INPUT_DIR = self.LOCAL_INPUT_DIR
        else:
            self.WORKING_DIR = self.KAGGLE_WORKING_DIR
            self.INPUT_DIR = self.KAGGLE_INPUT_DIR
        
        # Project directories
        self.MODELS_DIR = self.WORKING_DIR / "models"
        self.TEMP_DIR = self.WORKING_DIR / "temp"
        self.OUTPUT_DIR = self.WORKING_DIR / "output"
        self.LOGS_DIR = self.WORKING_DIR / "logs"
        self.CHECKPOINTS_DIR = self.WORKING_DIR / "checkpoints"
        self.SCRIPTS_DIR = self.WORKING_DIR / "scripts"
        
        # Create directories if they don't exist
        for directory in [self.MODELS_DIR, self.TEMP_DIR, self.OUTPUT_DIR, 
                         self.LOGS_DIR, self.CHECKPOINTS_DIR, self.SCRIPTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    # Model Configuration
    WHISPER_MODEL = "large-v3"
    SEAMLESS_MODEL = "facebook/hf-seamless-m4t-large"
    OPENVOICE_REPO = "myshell-ai/OpenVoice"
    
    # Language Settings
    SOURCE_LANGUAGE = "ar"  # Arabic
    TARGET_LANGUAGES = ["en", "de"]  # English, German
    
    # Processing Settings
    AUDIO_SAMPLE_RATE = 48000
    AUDIO_FORMAT = "wav"
    VIDEO_RESOLUTION = "1080p"
    
    # GPU Memory Management
    GPU_MEMORY_FRACTION = 0.8
    BATCH_SIZE = 16
    MAX_CHUNK_LENGTH = 30.0  # seconds
    
    # Quality Settings
    NOISE_REDUCTION_STRENGTH = 0.5
    VOICE_SIMILARITY_THRESHOLD = 0.85
    TRANSLATION_CONFIDENCE_THRESHOLD = 0.7
    
    # Session Management
    MAX_SESSION_TIME = 11.5 * 3600  # 11.5 hours in seconds
    CHECKPOINT_INTERVAL = 1800  # 30 minutes
    
    # File Processing
    MAX_FILE_SIZE_GB = 8
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mkv", ".mov", ".wmv"]
    
    # Output Settings
    OUTPUT_VIDEO_FORMATS = ["mp4"]
    OUTPUT_RESOLUTIONS = ["1080p", "720p"]
    SUBTITLE_FORMATS = ["srt", "vtt"]
    
    # Error Handling
    MAX_RETRIES = 3
    RETRY_DELAY = 60  # seconds
    
    # Resource Monitoring
    MEMORY_WARNING_THRESHOLD = 0.9  # 90% memory usage
    DISK_SPACE_WARNING_THRESHOLD = 10  # GB
    
    def get_video_output_path(self, video_name, language, resolution="1080p", format="mp4"):
        """Get output path for processed video"""
        return self.OUTPUT_DIR / video_name / f"{video_name}_{language}_{resolution}.{format}"
    
    def get_subtitle_output_path(self, video_name, language, format="srt"):
        """Get output path for subtitle files"""
        return self.OUTPUT_DIR / video_name / f"{video_name}_{language}.{format}"
    
    def get_checkpoint_path(self, video_name, step):
        """Get checkpoint file path"""
        return self.CHECKPOINTS_DIR / f"{video_name}_step_{step}.json"
    
    def get_log_path(self, video_name):
        """Get log file path"""
        return self.LOGS_DIR / f"{video_name}_processing.log"

# Global configuration instance
config = Config()
