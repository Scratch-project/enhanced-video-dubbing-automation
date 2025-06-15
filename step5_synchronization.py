"""
Step 5: Intelligent Audio-Video Synchronization Module
Enhanced Video Dubbing Automation Project
"""

import librosa
import numpy as np
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dtw import dtw
from scipy import signal
import soundfile as sf

from config import config
from utils import Logger, CheckpointManager, retry_on_failure

class SynchronizationProcessor:
    """Handle intelligent audio-video synchronization using DTW"""
    
    def __init__(self, video_name: str):
        self.video_name = video_name
        self.logger = Logger(video_name)
        self.checkpoint_manager = CheckpointManager(video_name)
        
    def analyze_original_timing(self, original_audio_path: Path) -> Dict[str, Any]:
        """Analyze original audio timing patterns and speech rate"""
        self.logger.log_step_start("Original Audio Timing Analysis")
        
        try:
            # Load original audio
            audio, sr = librosa.load(str(original_audio_path), sr=config.AUDIO_SAMPLE_RATE)
            
            # Calculate speech rate metrics
            # Detect onset times
            onset_frames = librosa.onset.onset_detect(
                y=audio, sr=sr, units='time', hop_length=512
            )
            
            # Calculate speaking rate (onsets per minute)
            duration_minutes = len(audio) / sr / 60
            speaking_rate = len(onset_frames) / duration_minutes if duration_minutes > 0 else 0
            
            # Calculate spectral features for timing analysis
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            
            # Calculate energy envelope
            hop_length = 512
            frame_length = 2048
            energy = librosa.feature.rms(
                y=audio, frame_length=frame_length, hop_length=hop_length
            )[0]
            
            # Timing statistics
            timing_analysis = {
                "duration_seconds": float(len(audio) / sr),
                "duration_minutes": float(duration_minutes),
                "speaking_rate_onsets_per_minute": float(speaking_rate),
                "tempo_bpm": float(tempo),
                "total_onsets": len(onset_frames),
                "onset_times": onset_frames.tolist(),
                "average_spectral_centroid": float(np.mean(spectral_centroid)),
                "energy_statistics": {
                    "mean": float(np.mean(energy)),
                    "std": float(np.std(energy)),
                    "max": float(np.max(energy)),
                    "min": float(np.min(energy))
                },
                "mfcc_statistics": {
                    "mean": np.mean(mfcc, axis=1).tolist(),
                    "std": np.std(mfcc, axis=1).tolist()
                }
            }
            
            # Save timing analysis
            analysis_file = config.TEMP_DIR / f"{self.video_name}_timing_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(timing_analysis, f, indent=2)
            
            self.logger.info(f"Original audio analysis: {duration_minutes:.1f}min, "
                           f"speaking rate: {speaking_rate:.1f} onsets/min")
            
            self.logger.log_step_end("Original Audio Timing Analysis", True)
            return timing_analysis
            
        except Exception as e:
            self.logger.error(f"Original timing analysis failed: {str(e)}")
            raise
    
    def calculate_length_differences(self, original_audio_path: Path, 
                                   dubbed_audio_path: Path) -> Dict[str, float]:
        """Calculate length differences between original and dubbed audio"""
        self.logger.log_step_start("Length Difference Calculation")
        
        try:
            # Load both audio files
            original_audio, sr1 = librosa.load(str(original_audio_path), sr=config.AUDIO_SAMPLE_RATE)
            dubbed_audio, sr2 = librosa.load(str(dubbed_audio_path), sr=config.AUDIO_SAMPLE_RATE)
            
            # Calculate durations
            original_duration = len(original_audio) / sr1
            dubbed_duration = len(dubbed_audio) / sr2
            
            # Calculate differences
            length_difference = dubbed_duration - original_duration
            length_ratio = dubbed_duration / original_duration if original_duration > 0 else 1.0
            
            differences = {
                "original_duration": float(original_duration),
                "dubbed_duration": float(dubbed_duration),
                "length_difference_seconds": float(length_difference),
                "length_ratio": float(length_ratio),
                "speed_adjustment_needed": float(1.0 / length_ratio) if length_ratio != 0 else 1.0
            }
            
            self.logger.info(f"Length analysis: Original={original_duration:.1f}s, "
                           f"Dubbed={dubbed_duration:.1f}s, Ratio={length_ratio:.3f}")
            
            self.logger.log_step_end("Length Difference Calculation", True)
            return differences
            
        except Exception as e:
            self.logger.error(f"Length difference calculation failed: {str(e)}")
            raise
    
    def perform_dynamic_time_warping(self, original_audio_path: Path, 
                                   dubbed_audio_path: Path) -> Dict[str, Any]:
        """Use Dynamic Time Warping for intelligent alignment"""
        self.logger.log_step_start("Dynamic Time Warping Alignment")
        
        try:
            # Load audio files
            original_audio, sr = librosa.load(str(original_audio_path), sr=config.AUDIO_SAMPLE_RATE)
            dubbed_audio, _ = librosa.load(str(dubbed_audio_path), sr=config.AUDIO_SAMPLE_RATE)
            
            # Extract features for DTW alignment
            # Use MFCC features for better alignment
            original_mfcc = librosa.feature.mfcc(y=original_audio, sr=sr, n_mfcc=13)
            dubbed_mfcc = librosa.feature.mfcc(y=dubbed_audio, sr=sr, n_mfcc=13)
            
            # Normalize features
            original_mfcc = (original_mfcc - np.mean(original_mfcc, axis=1, keepdims=True)) / np.std(original_mfcc, axis=1, keepdims=True)
            dubbed_mfcc = (dubbed_mfcc - np.mean(dubbed_mfcc, axis=1, keepdims=True)) / np.std(dubbed_mfcc, axis=1, keepdims=True)
            
            # Compute DTW alignment
            distance_matrix = np.linalg.norm(
                original_mfcc[:, :, np.newaxis] - dubbed_mfcc[:, np.newaxis, :], 
                axis=0
            )
            
            # Perform DTW
            distance, cost_matrix, acc_cost_matrix, path = dtw(
                distance_matrix.T, 
                dist_method='euclidean'
            )
            
            # Extract alignment path
            alignment_path = list(zip(*path))
            
            # Convert frame indices to time
            hop_length = 512
            original_times = librosa.frames_to_time(
                np.array([p[1] for p in alignment_path]), sr=sr, hop_length=hop_length
            )
            dubbed_times = librosa.frames_to_time(
                np.array([p[0] for p in alignment_path]), sr=sr, hop_length=hop_length
            )
            
            # Calculate alignment statistics
            time_differences = dubbed_times - original_times
            
            dtw_result = {
                "dtw_distance": float(distance),
                "alignment_points": len(alignment_path),
                "original_times": original_times.tolist(),
                "dubbed_times": dubbed_times.tolist(),
                "time_differences": time_differences.tolist(),
                "alignment_statistics": {
                    "mean_time_difference": float(np.mean(time_differences)),
                    "std_time_difference": float(np.std(time_differences)),
                    "max_time_difference": float(np.max(time_differences)),
                    "min_time_difference": float(np.min(time_differences))
                }
            }
            
            # Save DTW results
            dtw_file = config.TEMP_DIR / f"{self.video_name}_dtw_alignment.json"
            with open(dtw_file, 'w') as f:
                json.dump(dtw_result, f, indent=2)
            
            self.logger.info(f"DTW alignment completed: {len(alignment_path)} alignment points, "
                           f"distance={distance:.3f}")
            
            self.logger.log_step_end("Dynamic Time Warping Alignment", True)
            return dtw_result
            
        except Exception as e:
            self.logger.error(f"Dynamic time warping failed: {str(e)}")
            raise
    
    def apply_time_stretching(self, dubbed_audio_path: Path, 
                            stretch_factor: float) -> Path:
        """Apply time-stretching using librosa"""
        self.logger.log_step_start("Time Stretching")
        
        try:
            # Load dubbed audio
            audio, sr = librosa.load(str(dubbed_audio_path), sr=config.AUDIO_SAMPLE_RATE)
            
            # Apply time stretching
            stretched_audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
            
            # Save stretched audio
            output_path = config.TEMP_DIR / f"{self.video_name}_stretched_audio.wav"
            sf.write(str(output_path), stretched_audio, sr)
            
            # Calculate new duration
            new_duration = len(stretched_audio) / sr
            original_duration = len(audio) / sr
            
            self.logger.info(f"Time stretching applied: factor={stretch_factor:.3f}, "
                           f"duration {original_duration:.1f}s → {new_duration:.1f}s")
            
            self.logger.log_step_end("Time Stretching", True)
            return output_path
            
        except Exception as e:
            self.logger.error(f"Time stretching failed: {str(e)}")
            raise
    
    def insert_intelligent_pauses(self, dubbed_audio_path: Path, 
                                dtw_result: Dict[str, Any],
                                timing_analysis: Dict[str, Any]) -> Path:
        """Insert intelligent pauses to preserve visual cues"""
        self.logger.log_step_start("Intelligent Pause Insertion")
        
        try:
            # Load dubbed audio
            audio, sr = librosa.load(str(dubbed_audio_path), sr=config.AUDIO_SAMPLE_RATE)
            
            # Analyze where pauses should be inserted based on DTW and original timing
            original_onsets = np.array(timing_analysis["onset_times"])
            time_differences = np.array(dtw_result["time_differences"])
            
            # Find locations where we need to add pauses (negative time differences)
            pause_locations = []
            
            for i, time_diff in enumerate(time_differences):
                if time_diff < -0.1:  # Need to add pause (dubbed is ahead)
                    pause_duration = min(abs(time_diff), 2.0)  # Max 2 second pause
                    pause_time = dtw_result["dubbed_times"][i]
                    
                    pause_locations.append({
                        "time": pause_time,
                        "duration": pause_duration
                    })
            
            # Sort pause locations by time
            pause_locations.sort(key=lambda x: x["time"])
            
            # Insert pauses
            adjusted_audio = []
            current_time = 0.0
            
            for pause in pause_locations:
                pause_time = pause["time"]
                pause_duration = pause["duration"]
                
                # Add audio up to pause point
                pause_sample = int(pause_time * sr)
                current_sample = int(current_time * sr)
                
                if pause_sample > current_sample:
                    adjusted_audio.extend(audio[current_sample:pause_sample])
                
                # Add silence (pause)
                silence_samples = int(pause_duration * sr)
                adjusted_audio.extend([0.0] * silence_samples)
                
                current_time = pause_time
            
            # Add remaining audio
            current_sample = int(current_time * sr)
            if current_sample < len(audio):
                adjusted_audio.extend(audio[current_sample:])
            
            # Convert to numpy array and save
            adjusted_audio = np.array(adjusted_audio, dtype=np.float32)
            
            output_path = config.TEMP_DIR / f"{self.video_name}_pause_adjusted_audio.wav"
            sf.write(str(output_path), adjusted_audio, sr)
            
            new_duration = len(adjusted_audio) / sr
            self.logger.info(f"Intelligent pauses inserted: {len(pause_locations)} pauses, "
                           f"new duration: {new_duration:.1f}s")
            
            self.logger.log_step_end("Intelligent Pause Insertion", True)
            return output_path
            
        except Exception as e:
            self.logger.error(f"Intelligent pause insertion failed: {str(e)}")
            raise
    
    def generate_alignment_report(self, original_audio_path: Path, 
                                final_audio_path: Path,
                                timing_analysis: Dict[str, Any],
                                dtw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alignment report for quality validation"""
        self.logger.log_step_start("Alignment Report Generation")
        
        try:
            # Load final audio for comparison
            final_audio, sr = librosa.load(str(final_audio_path), sr=config.AUDIO_SAMPLE_RATE)
            final_duration = len(final_audio) / sr
            
            # Calculate final alignment quality
            original_duration = timing_analysis["duration_seconds"]
            duration_difference = abs(final_duration - original_duration)
            duration_accuracy = 1.0 - (duration_difference / original_duration)
            
            # DTW alignment quality
            dtw_stats = dtw_result["alignment_statistics"]
            alignment_quality = 1.0 / (1.0 + abs(dtw_stats["mean_time_difference"]))
            
            # Overall synchronization score
            sync_score = (duration_accuracy + alignment_quality) / 2.0
            
            alignment_report = {
                "video_name": self.video_name,
                "original_duration": original_duration,
                "final_duration": final_duration,
                "duration_difference": duration_difference,
                "duration_accuracy": duration_accuracy,
                "alignment_quality": alignment_quality,
                "synchronization_score": sync_score,
                "dtw_statistics": dtw_stats,
                "timing_preservation": {
                    "original_speaking_rate": timing_analysis["speaking_rate_onsets_per_minute"],
                    "tempo_preserved": timing_analysis["tempo_bpm"]
                },
                "quality_assessment": {
                    "excellent": sync_score >= 0.9,
                    "good": 0.8 <= sync_score < 0.9,
                    "acceptable": 0.7 <= sync_score < 0.8,
                    "needs_improvement": sync_score < 0.7
                }
            }
            
            # Save alignment report
            report_file = config.TEMP_DIR / f"{self.video_name}_alignment_report.json"
            with open(report_file, 'w') as f:
                json.dump(alignment_report, f, indent=2)
            
            quality_level = "excellent" if sync_score >= 0.9 else \
                          "good" if sync_score >= 0.8 else \
                          "acceptable" if sync_score >= 0.7 else "needs improvement"
            
            self.logger.info(f"Alignment report generated: sync_score={sync_score:.3f} ({quality_level})")
            
            self.logger.log_step_end("Alignment Report Generation", True)
            return alignment_report
            
        except Exception as e:
            self.logger.error(f"Alignment report generation failed: {str(e)}")
            raise
    
    def process_synchronization(self, original_audio_path: Path,
                              dubbed_audio_files: Dict[str, Path]) -> Dict[str, Path]:
        """Complete synchronization processing pipeline"""
        self.logger.info(f"Starting synchronization processing for {self.video_name}")
        
        results = {}
        
        try:
            # Analyze original timing
            timing_analysis = self.analyze_original_timing(original_audio_path)
            
            # Process each dubbed language
            for language, dubbed_audio_path in dubbed_audio_files.items():
                self.logger.info(f"Processing synchronization for {language}")
                
                # Calculate length differences
                length_diff = self.calculate_length_differences(
                    original_audio_path, dubbed_audio_path
                )
                
                # Perform DTW alignment
                dtw_result = self.perform_dynamic_time_warping(
                    original_audio_path, dubbed_audio_path
                )
                
                # Apply time stretching if needed
                current_audio = dubbed_audio_path
                if abs(length_diff["length_ratio"] - 1.0) > 0.05:  # 5% difference threshold
                    stretch_factor = length_diff["speed_adjustment_needed"]
                    current_audio = self.apply_time_stretching(current_audio, stretch_factor)
                
                # Insert intelligent pauses
                final_audio = self.insert_intelligent_pauses(
                    current_audio, dtw_result, timing_analysis
                )
                
                # Generate alignment report
                alignment_report = self.generate_alignment_report(
                    original_audio_path, final_audio, timing_analysis, dtw_result
                )
                
                # Save final synchronized audio
                synchronized_audio_path = config.TEMP_DIR / f"{self.video_name}_synchronized_{language}.wav"
                
                # Copy final audio to synchronized path
                import shutil
                shutil.copy2(str(final_audio), str(synchronized_audio_path))
                
                results[language] = synchronized_audio_path
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(f"synchronization_{language}", {
                    "synchronized_audio": str(synchronized_audio_path),
                    "alignment_report": alignment_report,
                    "sync_score": alignment_report["synchronization_score"]
                })
                
                self.logger.info(f"Synchronization for {language} completed: "
                               f"score={alignment_report['synchronization_score']:.3f}")
            
            # Mark step as completed
            self.checkpoint_manager.mark_step_completed("synchronization_processing")
            
            self.logger.info("Synchronization processing completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Synchronization processing failed: {str(e)}")
            raise
