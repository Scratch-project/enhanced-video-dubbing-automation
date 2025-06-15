"""
Step 4: Voice Cloning & Synthesis with OpenVoice v2 Module
Enhanced Video Dubbing Automation Project
"""

import torch
import librosa
import soundfile as sf
import numpy as np
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import gc
from tqdm import tqdm
import os

from config import config
from utils import Logger, CheckpointManager, retry_on_failure, clear_gpu_memory

class VoiceCloningProcessor:
    """Handle voice cloning and synthesis using OpenVoice v2"""
    
    def __init__(self, video_name: str):
        self.video_name = video_name
        self.logger = Logger(video_name)
        self.checkpoint_manager = CheckpointManager(video_name)
        self.openvoice_dir = config.WORKING_DIR / "OpenVoice"
        self.voice_samples_dir = config.TEMP_DIR / f"{self.video_name}_voice_samples"
        self.voice_samples_dir.mkdir(exist_ok=True)
        
    def setup_openvoice_environment(self):
        """Setup OpenVoice v2 environment and dependencies"""
        self.logger.log_step_start("OpenVoice Setup")
        
        try:
            # Ensure OpenVoice repository exists
            if not self.openvoice_dir.exists():
                self.logger.info("Cloning OpenVoice repository...")
                subprocess.run([
                    "git", "clone", "https://github.com/myshell-ai/OpenVoice.git",
                    str(self.openvoice_dir)
                ], check=True)
            
            # Add OpenVoice to Python path
            if str(self.openvoice_dir) not in sys.path:
                sys.path.insert(0, str(self.openvoice_dir))
            
            # Install additional requirements if needed
            requirements_file = self.openvoice_dir / "requirements.txt"
            if requirements_file.exists():
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "--user",
                    "-r", str(requirements_file)
                ], check=True)
            
            self.logger.info("OpenVoice environment setup completed")
            self.logger.log_step_end("OpenVoice Setup", True)
            
        except Exception as e:
            self.logger.error(f"OpenVoice setup failed: {str(e)}")
            raise
    
    def extract_voice_samples(self, clean_audio_path: Path, 
                            vad_results: Dict[str, Any]) -> List[Path]:
        """Extract clean voice samples from the original speaker"""
        self.logger.log_step_start("Voice Sample Extraction")
        
        try:
            # Load clean audio
            audio, sr = librosa.load(str(clean_audio_path), sr=config.AUDIO_SAMPLE_RATE)
            
            # Select best voice segments for reference
            segments = vad_results["segments"]
            
            # Filter segments for quality voice samples
            quality_segments = []
            for segment in segments:
                duration = segment["duration"]
                # Select segments between 3-10 seconds for good voice samples
                if 3.0 <= duration <= 10.0:
                    start_sample = int(segment["start"] * sr)
                    end_sample = int(segment["end"] * sr)
                    
                    if end_sample <= len(audio):
                        segment_audio = audio[start_sample:end_sample]
                        
                        # Calculate quality metrics
                        rms_energy = np.sqrt(np.mean(segment_audio**2))
                        zero_crossing_rate = librosa.feature.zero_crossing_rate(segment_audio)[0].mean()
                        
                        # Filter for clear speech
                        if rms_energy > 0.01 and zero_crossing_rate < 0.15:
                            quality_segments.append({
                                "segment": segment,
                                "audio": segment_audio,
                                "rms_energy": rms_energy,
                                "quality_score": rms_energy / zero_crossing_rate
                            })
            
            # Sort by quality and select top samples
            quality_segments.sort(key=lambda x: x["quality_score"], reverse=True)
            top_samples = quality_segments[:min(5, len(quality_segments))]
            
            # Save voice samples
            sample_files = []
            for i, sample in enumerate(top_samples):
                sample_file = self.voice_samples_dir / f"reference_voice_{i:02d}.wav"
                
                # Normalize and save
                normalized_audio = sample["audio"] / np.max(np.abs(sample["audio"]))
                sf.write(str(sample_file), normalized_audio, sr)
                
                sample_files.append(sample_file)
                
                self.logger.info(f"Saved voice sample {i+1}: {sample['segment']['duration']:.1f}s, "
                               f"quality={sample['quality_score']:.3f}")
            
            self.logger.info(f"Extracted {len(sample_files)} high-quality voice samples")
            
            # Save sample metadata
            samples_metadata = {
                "video_name": self.video_name,
                "total_samples": len(sample_files),
                "sample_files": [str(f) for f in sample_files],
                "quality_metrics": [
                    {
                        "file": str(sample_files[i]),
                        "duration": s["segment"]["duration"],
                        "quality_score": s["quality_score"],
                        "rms_energy": s["rms_energy"]
                    }
                    for i, s in enumerate(top_samples)
                ]
            }
            
            metadata_file = self.voice_samples_dir / "samples_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(samples_metadata, f, indent=2)
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint("voice_sample_extraction", {
                "samples_dir": str(self.voice_samples_dir),
                "sample_count": len(sample_files),
                "metadata_file": str(metadata_file)
            })
            
            self.logger.log_step_end("Voice Sample Extraction", True)
            return sample_files
            
        except Exception as e:
            self.logger.error(f"Voice sample extraction failed: {str(e)}")
            raise
    
    @retry_on_failure(max_retries=3)
    def initialize_voice_cloning_model(self):
        """Initialize OpenVoice v2 model for cross-lingual cloning"""
        self.logger.log_step_start("Voice Cloning Model Initialization")
        
        try:
            # Import OpenVoice modules
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
            
            # Initialize tone color converter
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tone_color_converter = ToneColorConverter(
                f'{self.openvoice_dir}/checkpoints/converter/config.json',
                device=device
            )
            self.tone_color_converter.load_ckpt(
                f'{self.openvoice_dir}/checkpoints/converter/checkpoint.pth'
            )
            
            # Initialize speaker encoder
            self.se_extractor = se_extractor
            
            self.logger.info(f"Voice cloning model initialized on {device}")
            self.logger.log_step_end("Voice Cloning Model Initialization", True)
            
        except Exception as e:
            self.logger.error(f"Voice cloning model initialization failed: {str(e)}")
            raise
    
    def extract_tone_color_embedding(self, voice_samples: List[Path]) -> np.ndarray:
        """Extract tone color embedding from voice samples"""
        self.logger.log_step_start("Tone Color Embedding Extraction")
        
        try:
            # Load and process voice samples
            embeddings = []
            
            for sample_file in voice_samples:
                # Extract speaker embedding
                se, audio_name = self.se_extractor.get_se(
                    str(sample_file), 
                    self.tone_color_converter, 
                    target_dir=str(self.voice_samples_dir),
                    vad=True
                )
                embeddings.append(se)
            
            # Average embeddings for more robust representation
            if embeddings:
                averaged_embedding = np.mean(embeddings, axis=0)
                
                # Save embedding
                embedding_file = self.voice_samples_dir / "tone_color_embedding.npy"
                np.save(str(embedding_file), averaged_embedding)
                
                self.logger.info(f"Tone color embedding extracted from {len(embeddings)} samples")
                self.logger.log_step_end("Tone Color Embedding Extraction", True)
                
                return averaged_embedding
            else:
                raise ValueError("No valid embeddings extracted")
                
        except Exception as e:
            self.logger.error(f"Tone color embedding extraction failed: {str(e)}")
            raise
    
    def synthesize_dubbed_audio(self, translation_data: Dict[str, Any], 
                              target_language: str, 
                              tone_embedding: np.ndarray) -> Path:
        """Synthesize dubbed audio for target language"""
        self.logger.log_step_start(f"Audio Synthesis ({target_language})")
        
        try:
            # Import OpenVoice TTS modules
            from openvoice.api import BaseSpeakerTTS
            
            # Initialize base TTS model for target language
            if target_language == "en":
                base_model_path = f"{self.openvoice_dir}/checkpoints/base_speakers/EN"
            elif target_language == "de":
                base_model_path = f"{self.openvoice_dir}/checkpoints/base_speakers/DE"
            else:
                raise ValueError(f"Unsupported target language: {target_language}")
            
            # Load base TTS model
            base_speaker_tts = BaseSpeakerTTS(
                f'{base_model_path}/config.json',
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            base_speaker_tts.load_ckpt(f'{base_model_path}/checkpoint.pth')
            
            # Create output directory for synthesized audio
            synth_dir = config.TEMP_DIR / f"{self.video_name}_synthesized_{target_language}"
            synth_dir.mkdir(exist_ok=True)
            
            # Synthesize each segment
            segments = translation_data["segments"]
            synthesized_segments = []
            
            for i, segment in enumerate(tqdm(segments, desc=f"Synthesizing {target_language}")):
                try:
                    text = segment["translated_text"]
                    
                    # Skip empty or very short texts
                    if len(text.strip()) < 3:
                        continue
                    
                    # Generate base audio
                    base_audio_file = synth_dir / f"base_segment_{i:04d}.wav"
                    base_speaker_tts.tts(
                        text, 
                        str(base_audio_file), 
                        speaker='default',
                        language=target_language
                    )
                    
                    # Apply tone color conversion
                    final_audio_file = synth_dir / f"final_segment_{i:04d}.wav"
                    self.tone_color_converter.convert(
                        audio_src_path=str(base_audio_file),
                        src_se=tone_embedding,
                        tgt_se=tone_embedding,
                        output_path=str(final_audio_file),
                        message="Converting tone color"
                    )
                    
                    # Load and validate synthesized audio
                    synth_audio, synth_sr = librosa.load(str(final_audio_file))
                    synth_duration = len(synth_audio) / synth_sr
                    
                    synthesized_segments.append({
                        "segment_id": segment["id"],
                        "original_start": segment["start"],
                        "original_end": segment["end"],
                        "original_duration": segment["end"] - segment["start"],
                        "synthesized_duration": synth_duration,
                        "audio_file": str(final_audio_file),
                        "text": text,
                        "timing_ratio": synth_duration / (segment["end"] - segment["start"])
                    })
                    
                    # Clean up base audio file
                    if base_audio_file.exists():
                        base_audio_file.unlink()
                    
                except Exception as e:
                    self.logger.warning(f"Failed to synthesize segment {i}: {str(e)}")
                    continue
            
            # Save synthesis metadata
            synthesis_metadata = {
                "video_name": self.video_name,
                "target_language": target_language,
                "total_segments": len(synthesized_segments),
                "synthesis_directory": str(synth_dir),
                "segments": synthesized_segments,
                "timing_statistics": self._calculate_timing_stats(synthesized_segments)
            }
            
            metadata_file = synth_dir / "synthesis_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(synthesis_metadata, f, indent=2)
            
            self.logger.info(f"Synthesized {len(synthesized_segments)} segments for {target_language}")
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(f"audio_synthesis_{target_language}", {
                "synthesis_dir": str(synth_dir),
                "segments_count": len(synthesized_segments),
                "metadata_file": str(metadata_file)
            })
            
            self.logger.log_step_end(f"Audio Synthesis ({target_language})", True)
            return synth_dir
            
        except Exception as e:
            self.logger.error(f"Audio synthesis for {target_language} failed: {str(e)}")
            raise
        finally:
            clear_gpu_memory()
    
    def _calculate_timing_stats(self, segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate timing statistics for synthesized segments"""
        if not segments:
            return {}
        
        timing_ratios = [seg["timing_ratio"] for seg in segments]
        
        return {
            "average_timing_ratio": np.mean(timing_ratios),
            "min_timing_ratio": np.min(timing_ratios),
            "max_timing_ratio": np.max(timing_ratios),
            "std_timing_ratio": np.std(timing_ratios),
            "total_original_duration": sum(seg["original_duration"] for seg in segments),
            "total_synthesized_duration": sum(seg["synthesized_duration"] for seg in segments)
        }
    
    def create_continuous_audio(self, synthesis_dir: Path, 
                              target_language: str) -> Path:
        """Create continuous dubbed audio from synthesized segments"""
        self.logger.log_step_start(f"Continuous Audio Creation ({target_language})")
        
        try:
            # Load synthesis metadata
            metadata_file = synthesis_dir / "synthesis_metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            segments = metadata["segments"]
            
            # Sort segments by original start time
            segments.sort(key=lambda x: x["original_start"])
            
            # Create continuous audio with proper timing
            continuous_audio = []
            current_time = 0.0
            sample_rate = config.AUDIO_SAMPLE_RATE
            
            for segment in segments:
                # Calculate silence needed before this segment
                silence_duration = segment["original_start"] - current_time
                
                if silence_duration > 0:
                    silence_samples = int(silence_duration * sample_rate)
                    continuous_audio.extend([0.0] * silence_samples)
                    current_time += silence_duration
                
                # Load and add segment audio
                if os.path.exists(segment["audio_file"]):
                    seg_audio, seg_sr = librosa.load(segment["audio_file"], sr=sample_rate)
                    continuous_audio.extend(seg_audio.tolist())
                    current_time += len(seg_audio) / sample_rate
            
            # Convert to numpy array and save
            continuous_audio = np.array(continuous_audio, dtype=np.float32)
            
            output_file = config.TEMP_DIR / f"{self.video_name}_dubbed_{target_language}.wav"
            sf.write(str(output_file), continuous_audio, sample_rate)
            
            duration_minutes = len(continuous_audio) / sample_rate / 60
            self.logger.info(f"Created continuous dubbed audio: {duration_minutes:.1f} minutes")
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(f"continuous_audio_{target_language}", {
                "output_file": str(output_file),
                "duration_minutes": duration_minutes,
                "segments_processed": len(segments)
            })
            
            self.logger.log_step_end(f"Continuous Audio Creation ({target_language})", True)
            return output_file
            
        except Exception as e:
            self.logger.error(f"Continuous audio creation failed: {str(e)}")
            raise
    
    def process_voice_cloning(self, clean_audio_path: Path, 
                            vad_results: Dict[str, Any],
                            translation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Path]:
        """Complete voice cloning and synthesis pipeline"""
        self.logger.info(f"Starting voice cloning processing for {self.video_name}")
        
        results = {}
        
        try:
            # Setup OpenVoice environment
            self.setup_openvoice_environment()
            
            # Extract voice samples
            voice_samples = self.extract_voice_samples(clean_audio_path, vad_results)
            
            # Initialize voice cloning model
            self.initialize_voice_cloning_model()
            
            # Extract tone color embedding
            tone_embedding = self.extract_tone_color_embedding(voice_samples)
            
            # Process each target language
            for target_language in config.TARGET_LANGUAGES:
                if target_language in translation_results:
                    self.logger.info(f"Processing voice synthesis for {target_language}")
                    
                    # Synthesize dubbed audio
                    synthesis_dir = self.synthesize_dubbed_audio(
                        translation_results[target_language],
                        target_language,
                        tone_embedding
                    )
                    
                    # Create continuous audio
                    continuous_audio_file = self.create_continuous_audio(
                        synthesis_dir, target_language
                    )
                    
                    results[target_language] = continuous_audio_file
            
            # Mark step as completed
            self.checkpoint_manager.mark_step_completed("voice_cloning_processing")
            
            self.logger.info("Voice cloning processing completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Voice cloning processing failed: {str(e)}")
            raise
        finally:
            # Clean up models to free memory
            if hasattr(self, 'tone_color_converter'):
                del self.tone_color_converter
            clear_gpu_memory()
