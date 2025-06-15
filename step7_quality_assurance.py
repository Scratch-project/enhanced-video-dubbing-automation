"""
Step 7: Quality Assurance & Final Assembly Module
Enhanced Video Dubbing Automation Project
"""

import subprocess
import json
import numpy as np
import librosa
import cv2
from pathlib import Path
from typing import Dict, List, Any, Tuple
import shutil
import tempfile

from config import config
from utils import Logger, CheckpointManager, retry_on_failure

class QualityAssuranceProcessor:
    """Handle quality assurance and final video assembly"""
    
    def __init__(self, video_name: str):
        self.video_name = video_name
        self.logger = Logger(video_name)
        self.checkpoint_manager = CheckpointManager(video_name)
        
    def analyze_audio_quality(self, audio_path: Path) -> Dict[str, Any]:
        """Automated quality checks using audio analysis"""
        self.logger.log_step_start("Audio Quality Analysis")
        
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=config.AUDIO_SAMPLE_RATE)
            
            # Calculate quality metrics
            quality_metrics = {}
            
            # 1. Audio levels
            rms_level = np.sqrt(np.mean(audio**2))
            peak_level = np.max(np.abs(audio))
            
            quality_metrics["audio_levels"] = {
                "rms_db": float(20 * np.log10(rms_level)) if rms_level > 0 else -60,
                "peak_db": float(20 * np.log10(peak_level)) if peak_level > 0 else -60,
                "dynamic_range_db": float(20 * np.log10(peak_level / rms_level)) if rms_level > 0 else 0
            }
            
            # 2. Distortion analysis
            # Calculate THD (Total Harmonic Distortion) approximation
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Find fundamental frequencies and harmonics
            freqs = librosa.fft_frequencies(sr=sr)
            harmonic_content = np.mean(magnitude, axis=1)
            
            quality_metrics["distortion"] = {
                "spectral_flatness": float(np.mean(librosa.feature.spectral_flatness(y=audio))),
                "spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio)))
            }
            
            # 3. Frequency response
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            
            quality_metrics["frequency_response"] = {
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "mfcc_statistics": {
                    "mean": np.mean(mfcc, axis=1).tolist(),
                    "std": np.std(mfcc, axis=1).tolist()
                }
            }
            
            # 4. Silence and gaps analysis
            # Detect silence periods
            silence_threshold = np.max(np.abs(audio)) * 0.01  # 1% of peak
            silence_frames = np.abs(audio) < silence_threshold
            
            # Find silence segments
            silence_segments = []
            in_silence = False
            silence_start = 0
            
            for i, is_silent in enumerate(silence_frames):
                if is_silent and not in_silence:
                    silence_start = i
                    in_silence = True
                elif not is_silent and in_silence:
                    silence_duration = (i - silence_start) / sr
                    if silence_duration > 0.1:  # Only count silences > 100ms
                        silence_segments.append(silence_duration)
                    in_silence = False
            
            quality_metrics["silence_analysis"] = {
                "total_silence_segments": len(silence_segments),
                "average_silence_duration": float(np.mean(silence_segments)) if silence_segments else 0.0,
                "max_silence_duration": float(np.max(silence_segments)) if silence_segments else 0.0,
                "silence_ratio": float(np.sum(silence_frames) / len(audio))
            }
            
            # 5. Overall quality score
            quality_score = self._calculate_overall_audio_quality(quality_metrics)
            quality_metrics["overall_quality_score"] = quality_score
            
            self.logger.info(f"Audio quality analysis completed: score={quality_score:.3f}")
            self.logger.log_step_end("Audio Quality Analysis", True)
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Audio quality analysis failed: {str(e)}")
            raise
    
    def _calculate_overall_audio_quality(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall audio quality score (0-1)"""
        score = 1.0
        
        # Penalize poor audio levels
        rms_db = metrics["audio_levels"]["rms_db"]
        if rms_db < -30 or rms_db > -6:  # Too quiet or too loud
            score -= 0.2
        
        # Penalize high distortion
        spectral_flatness = metrics["distortion"]["spectral_flatness"]
        if spectral_flatness > 0.5:  # Too flat (noisy)
            score -= 0.2
        
        # Penalize excessive silence
        silence_ratio = metrics["silence_analysis"]["silence_ratio"]
        if silence_ratio > 0.3:  # More than 30% silence
            score -= 0.1
        
        # Penalize very long silence gaps
        max_silence = metrics["silence_analysis"]["max_silence_duration"]
        if max_silence > 5.0:  # Silence longer than 5 seconds
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def validate_synchronization(self, original_video_path: Path, 
                               dubbed_audio_path: Path) -> Dict[str, Any]:
        """Validate audio-video synchronization accuracy"""
        self.logger.log_step_start("Synchronization Validation")
        
        try:
            # Extract original audio from video for comparison
            temp_original_audio = config.TEMP_DIR / f"{self.video_name}_temp_original.wav"
            
            cmd = [
                "ffmpeg", "-i", str(original_video_path),
                "-ar", str(config.AUDIO_SAMPLE_RATE),
                "-ac", "1", "-y", str(temp_original_audio)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Load both audio files
            original_audio, sr1 = librosa.load(str(temp_original_audio), sr=config.AUDIO_SAMPLE_RATE)
            dubbed_audio, sr2 = librosa.load(str(dubbed_audio_path), sr=config.AUDIO_SAMPLE_RATE)
            
            # Calculate cross-correlation for sync detection
            if len(dubbed_audio) > len(original_audio):
                correlation = np.correlate(dubbed_audio[:len(original_audio)], original_audio, mode='full')
            else:
                correlation = np.correlate(original_audio[:len(dubbed_audio)], dubbed_audio, mode='full')
            
            # Find peak correlation
            peak_index = np.argmax(correlation)
            max_correlation = correlation[peak_index] / (np.linalg.norm(original_audio) * np.linalg.norm(dubbed_audio))
            
            # Calculate timing offset
            offset_samples = peak_index - (len(correlation) // 2)
            offset_seconds = offset_samples / sr1
            
            sync_validation = {
                "correlation_coefficient": float(max_correlation),
                "timing_offset_seconds": float(offset_seconds),
                "sync_quality": "excellent" if abs(offset_seconds) < 0.1 else 
                              "good" if abs(offset_seconds) < 0.5 else
                              "acceptable" if abs(offset_seconds) < 1.0 else "poor",
                "duration_difference": float(abs(len(original_audio) - len(dubbed_audio)) / sr1)
            }
            
            # Clean up temp file
            if temp_original_audio.exists():
                temp_original_audio.unlink()
            
            self.logger.info(f"Sync validation: offset={offset_seconds:.3f}s, "
                           f"correlation={max_correlation:.3f}")
            
            self.logger.log_step_end("Synchronization Validation", True)
            return sync_validation
            
        except Exception as e:
            self.logger.error(f"Synchronization validation failed: {str(e)}")
            raise
    
    def generate_preview_clips(self, video_path: Path, 
                             dubbed_audio_files: Dict[str, Path],
                             num_clips: int = 3) -> Dict[str, List[Path]]:
        """Generate preview clips from different video sections"""
        self.logger.log_step_start("Preview Clip Generation")
        
        try:
            # Get video duration
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            video_duration = float(result.stdout.strip())
            
            # Calculate clip positions (beginning, middle, end)
            clip_positions = []
            if num_clips == 3:
                clip_positions = [
                    30,  # 30 seconds from start
                    video_duration / 2,  # Middle
                    video_duration - 60  # 60 seconds from end
                ]
            else:
                # Distribute clips evenly
                for i in range(num_clips):
                    position = (video_duration / (num_clips + 1)) * (i + 1)
                    clip_positions.append(max(30, position))
            
            preview_clips = {}
            
            # Generate clips for original video
            original_clips = []
            for i, start_time in enumerate(clip_positions):
                clip_path = config.TEMP_DIR / f"{self.video_name}_preview_original_{i+1}.mp4"
                
                cmd = [
                    "ffmpeg", "-i", str(video_path),
                    "-ss", str(start_time),
                    "-t", "30",  # 30-second clips
                    "-c:v", "libx264", "-c:a", "aac",
                    "-y", str(clip_path)
                ]
                
                subprocess.run(cmd, capture_output=True, check=True)
                original_clips.append(clip_path)
            
            preview_clips["original"] = original_clips
            
            # Generate clips for each dubbed version
            for language, dubbed_audio_path in dubbed_audio_files.items():
                dubbed_clips = []
                
                for i, start_time in enumerate(clip_positions):
                    clip_path = config.TEMP_DIR / f"{self.video_name}_preview_{language}_{i+1}.mp4"
                    
                    # Create clip with dubbed audio
                    cmd = [
                        "ffmpeg", "-i", str(video_path), "-i", str(dubbed_audio_path),
                        "-ss", str(start_time),
                        "-t", "30",
                        "-map", "0:v", "-map", "1:a",
                        "-c:v", "libx264", "-c:a", "aac",
                        "-y", str(clip_path)
                    ]
                    
                    subprocess.run(cmd, capture_output=True, check=True)
                    dubbed_clips.append(clip_path)
                
                preview_clips[language] = dubbed_clips
            
            self.logger.info(f"Generated {num_clips} preview clips for each version")
            self.logger.log_step_end("Preview Clip Generation", True)
            
            return preview_clips
            
        except Exception as e:
            self.logger.error(f"Preview clip generation failed: {str(e)}")
            raise
    
    def check_subtitle_alignment(self, subtitle_files: Dict[str, Path],
                                audio_files: Dict[str, Path]) -> Dict[str, Any]:
        """Check subtitle alignment and readability"""
        self.logger.log_step_start("Subtitle Alignment Check")
        
        try:
            alignment_results = {}
            
            for language in subtitle_files.keys():
                if language in audio_files:
                    subtitle_file = subtitle_files[language]
                    audio_file = audio_files[language]
                    
                    # Load audio duration
                    audio, sr = librosa.load(str(audio_file), sr=config.AUDIO_SAMPLE_RATE)
                    audio_duration = len(audio) / sr
                    
                    # Parse subtitle file
                    subtitles = self._parse_srt_file(subtitle_file)
                    
                    # Check alignment
                    alignment_issues = []
                    
                    for subtitle in subtitles:
                        # Check if subtitle extends beyond audio
                        if subtitle["end"] > audio_duration:
                            alignment_issues.append({
                                "segment": subtitle["index"],
                                "issue": "Subtitle extends beyond audio",
                                "subtitle_end": subtitle["end"],
                                "audio_duration": audio_duration
                            })
                        
                        # Check reading speed
                        words = len(subtitle["text"].split())
                        reading_speed = words / subtitle["duration"]
                        
                        if reading_speed > 3.0:  # Too fast
                            alignment_issues.append({
                                "segment": subtitle["index"],
                                "issue": "Reading speed too fast",
                                "words_per_second": reading_speed
                            })
                    
                    alignment_results[language] = {
                        "total_subtitles": len(subtitles),
                        "alignment_issues": alignment_issues,
                        "alignment_score": 1.0 - (len(alignment_issues) / max(len(subtitles), 1))
                    }
            
            self.logger.info(f"Subtitle alignment checked for {len(alignment_results)} languages")
            self.logger.log_step_end("Subtitle Alignment Check", True)
            
            return alignment_results
            
        except Exception as e:
            self.logger.error(f"Subtitle alignment check failed: {str(e)}")
            raise
    
    def _parse_srt_file(self, srt_file: Path) -> List[Dict[str, Any]]:
        """Parse SRT file and return segments"""
        segments = []
        
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        subtitle_blocks = content.split('\n\n')
        
        for block in subtitle_blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                index = int(lines[0])
                timing_line = lines[1]
                text = '\n'.join(lines[2:])
                
                # Parse timing
                start_str, end_str = timing_line.split(' --> ')
                start_seconds = self._srt_timestamp_to_seconds(start_str)
                end_seconds = self._srt_timestamp_to_seconds(end_str)
                
                segments.append({
                    "index": index,
                    "start": start_seconds,
                    "end": end_seconds,
                    "duration": end_seconds - start_seconds,
                    "text": text
                })
        
        return segments
    
    def _srt_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert SRT timestamp to seconds"""
        time_part, millisec_part = timestamp.split(',')
        hours, minutes, seconds = map(int, time_part.split(':'))
        milliseconds = int(millisec_part)
        
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        return total_seconds
    
    def assemble_final_videos(self, original_video_path: Path,
                            dubbed_audio_files: Dict[str, Path],
                            subtitle_files: Dict[str, Path]) -> Dict[str, Path]:
        """Assemble final videos with multiple audio tracks and subtitles"""
        self.logger.log_step_start("Final Video Assembly")
        
        try:
            output_dir = config.OUTPUT_DIR / self.video_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            final_videos = {}
            
            for language, dubbed_audio_path in dubbed_audio_files.items():
                self.logger.info(f"Assembling final video for {language}")
                
                # Create final video with dubbed audio and subtitles
                for resolution in config.OUTPUT_RESOLUTIONS:
                    output_path = config.get_video_output_path(
                        self.video_name, language, resolution
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Build ffmpeg command
                    cmd = [
                        "ffmpeg",
                        "-i", str(original_video_path),  # Video input
                        "-i", str(dubbed_audio_path),    # Audio input
                    ]
                    
                    # Add subtitle input if available
                    subtitle_input_added = False
                    if language in subtitle_files:
                        cmd.extend(["-i", str(subtitle_files[language])])
                        subtitle_input_added = True
                    
                    # Video encoding options based on resolution
                    if resolution == "1080p":
                        cmd.extend([
                            "-map", "0:v", "-map", "1:a",
                            "-c:v", "libx264",
                            "-preset", "medium",
                            "-crf", "23",
                            "-vf", "scale=1920:1080",
                            "-c:a", "aac",
                            "-b:a", "128k"
                        ])
                    elif resolution == "720p":
                        cmd.extend([
                            "-map", "0:v", "-map", "1:a",
                            "-c:v", "libx264",
                            "-preset", "medium",
                            "-crf", "25",
                            "-vf", "scale=1280:720",
                            "-c:a", "aac",
                            "-b:a", "128k"
                        ])
                    
                    # Add subtitle mapping if available
                    if subtitle_input_added:
                        cmd.extend(["-map", "2:s", "-c:s", "mov_text"])
                        cmd.extend([f"-metadata:s:s:0", f"language={language}"])
                    
                    # Output options
                    cmd.extend([
                        "-movflags", "+faststart",  # Optimize for streaming
                        "-y", str(output_path)
                    ])
                    
                    self.logger.info(f"Encoding {resolution} video for {language}...")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        self.logger.error(f"Video encoding failed: {result.stderr}")
                        continue
                    
                    if output_path.exists():
                        file_size_mb = output_path.stat().st_size / 1024**2
                        self.logger.info(f"Final video created: {resolution} - {file_size_mb:.1f}MB")
                        
                        if resolution == "1080p":  # Use 1080p as primary output
                            final_videos[language] = output_path
            
            self.logger.info(f"Final video assembly completed for {len(final_videos)} languages")
            self.logger.log_step_end("Final Video Assembly", True)
            
            return final_videos
            
        except Exception as e:
            self.logger.error(f"Final video assembly failed: {str(e)}")
            raise
    
    def generate_quality_report(self, original_video_path: Path,
                              final_videos: Dict[str, Path],
                              audio_quality_results: Dict[str, Any],
                              sync_validation_results: Dict[str, Any],
                              subtitle_alignment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        self.logger.log_step_start("Quality Report Generation")
        
        try:
            # Calculate overall quality scores
            quality_scores = {}
            
            for language in final_videos.keys():
                audio_score = audio_quality_results.get(language, {}).get("overall_quality_score", 0.5)
                sync_score = 1.0 if sync_validation_results.get(language, {}).get("sync_quality") == "excellent" else \
                           0.8 if sync_validation_results.get(language, {}).get("sync_quality") == "good" else \
                           0.6 if sync_validation_results.get(language, {}).get("sync_quality") == "acceptable" else 0.4
                
                subtitle_score = subtitle_alignment_results.get(language, {}).get("alignment_score", 0.5)
                
                overall_score = (audio_score + sync_score + subtitle_score) / 3.0
                quality_scores[language] = {
                    "audio_quality": audio_score,
                    "sync_quality": sync_score,
                    "subtitle_quality": subtitle_score,
                    "overall_quality": overall_score
                }
            
            # Get video information
            video_info = self._get_video_info(original_video_path)
            
            quality_report = {
                "video_name": self.video_name,
                "processing_date": self.checkpoint_manager.load_checkpoint(),
                "original_video_info": video_info,
                "processed_languages": list(final_videos.keys()),
                "quality_scores": quality_scores,
                "detailed_results": {
                    "audio_quality": audio_quality_results,
                    "synchronization": sync_validation_results,
                    "subtitle_alignment": subtitle_alignment_results
                },
                "output_files": {
                    language: str(path) for language, path in final_videos.items()
                },
                "recommendations": self._generate_quality_recommendations(quality_scores)
            }
            
            # Save quality report
            report_path = config.OUTPUT_DIR / self.video_name / "quality_report.json"
            with open(report_path, 'w') as f:
                json.dump(quality_report, f, indent=2)
            
            self.logger.info("Quality report generated successfully")
            self.logger.log_step_end("Quality Report Generation", True)
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Quality report generation failed: {str(e)}")
            raise
    
    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video file information"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except:
            return {"error": "Could not get video info"}
    
    def _generate_quality_recommendations(self, quality_scores: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        for language, scores in quality_scores.items():
            if scores["audio_quality"] < 0.7:
                recommendations.append(f"Consider improving audio quality for {language} (score: {scores['audio_quality']:.2f})")
            
            if scores["sync_quality"] < 0.8:
                recommendations.append(f"Review synchronization for {language} (score: {scores['sync_quality']:.2f})")
            
            if scores["subtitle_quality"] < 0.8:
                recommendations.append(f"Check subtitle timing for {language} (score: {scores['subtitle_quality']:.2f})")
            
            if scores["overall_quality"] >= 0.9:
                recommendations.append(f"Excellent quality achieved for {language}")
        
        return recommendations
    
    def process_quality_assurance(self, original_video_path: Path,
                                dubbed_audio_files: Dict[str, Path],
                                subtitle_files: Dict[str, Path]) -> Dict[str, Any]:
        """Complete quality assurance and final assembly pipeline"""
        self.logger.info(f"Starting quality assurance processing for {self.video_name}")
        
        try:
            # Audio quality analysis
            audio_quality_results = {}
            for language, audio_path in dubbed_audio_files.items():
                audio_quality_results[language] = self.analyze_audio_quality(audio_path)
            
            # Synchronization validation
            sync_validation_results = {}
            for language, audio_path in dubbed_audio_files.items():
                sync_validation_results[language] = self.validate_synchronization(
                    original_video_path, audio_path
                )
            
            # Subtitle alignment check
            subtitle_alignment_results = self.check_subtitle_alignment(
                subtitle_files, dubbed_audio_files
            )
            
            # Generate preview clips
            preview_clips = self.generate_preview_clips(
                original_video_path, dubbed_audio_files
            )
            
            # Assemble final videos
            final_videos = self.assemble_final_videos(
                original_video_path, dubbed_audio_files, subtitle_files
            )
            
            # Generate quality report
            quality_report = self.generate_quality_report(
                original_video_path, final_videos,
                audio_quality_results, sync_validation_results,
                subtitle_alignment_results
            )
            
            # Mark step as completed
            self.checkpoint_manager.mark_step_completed("quality_assurance_processing")
            
            self.logger.info("Quality assurance processing completed successfully")
            
            return {
                "final_videos": final_videos,
                "quality_report": quality_report,
                "preview_clips": preview_clips,
                "audio_quality": audio_quality_results,
                "sync_validation": sync_validation_results,
                "subtitle_alignment": subtitle_alignment_results
            }
            
        except Exception as e:
            self.logger.error(f"Quality assurance processing failed: {str(e)}")
            raise
