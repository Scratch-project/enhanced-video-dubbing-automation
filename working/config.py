
"""Enhanced Video Dubbing Configuration"""
import os
from pathlib import Path

class Config:
    def __init__(self, local_mode=False):
        self.local_mode = local_mode
        self.setup_directories()

    def setup_directories(self):
        if self.local_mode or not os.path.exists("/kaggle"):
            self.WORKING_DIR = Path("./working")
            self.INPUT_DIR = Path("./input")
        else:
            self.WORKING_DIR = Path("/kaggle/working")
            self.INPUT_DIR = Path("/kaggle/input")

        self.MODELS_DIR = self.WORKING_DIR / "models"
        self.TEMP_DIR = self.WORKING_DIR / "temp"
        self.OUTPUT_DIR = self.WORKING_DIR / "output"
        self.LOGS_DIR = self.WORKING_DIR / "logs"
        self.CHECKPOINTS_DIR = self.WORKING_DIR / "checkpoints"

        for directory in [self.MODELS_DIR, self.TEMP_DIR, self.OUTPUT_DIR, 
                         self.LOGS_DIR, self.CHECKPOINTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    # Model Configuration
    WHISPER_MODEL = "large-v3"
    SEAMLESS_MODEL = "facebook/hf-seamless-m4t-large"

    # Language Settings
    SOURCE_LANGUAGE = "ar"
    TARGET_LANGUAGES = ["en", "de"]

    # Processing Settings
    AUDIO_SAMPLE_RATE = 48000
    GPU_MEMORY_FRACTION = 0.8
    BATCH_SIZE = 16
    MAX_CHUNK_LENGTH = 30.0

    # Quality Settings
    NOISE_REDUCTION_STRENGTH = 0.5
    VOICE_SIMILARITY_THRESHOLD = 0.85

    # File Processing
    MAX_FILE_SIZE_GB = 8
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mkv", ".mov"]

    # Error Handling
    MAX_RETRIES = 3
    RETRY_DELAY = 60

    def get_video_output_path(self, video_name, language, resolution="1080p"):
        return self.OUTPUT_DIR / video_name / f"{video_name}_{language}_{resolution}.mp4"

    def get_log_path(self, video_name):
        return self.LOGS_DIR / f"{video_name}_processing.log"

config = Config(local_mode=not os.path.exists("/kaggle"))
