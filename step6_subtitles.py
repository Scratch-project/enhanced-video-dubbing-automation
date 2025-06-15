"""
Step 6: Subtitle Generation & Integration Module
Enhanced Video Dubbing Automation Project
"""

import whisper
import torch
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
import librosa

from config import config
from utils import Logger, CheckpointManager, retry_on_failure, clear_gpu_memory, safe_model_loading

class SubtitleProcessor:
    """Handle subtitle generation and integration"""
    
    def __init__(self, video_name: str):
        self.video_name = video_name
        self.logger = Logger(video_name)
        self.checkpoint_manager = CheckpointManager(video_name)
        self.whisper_model = None
        
    @safe_model_loading
    def load_whisper_model(self):
        """Load Whisper model for subtitle generation"""
        if self.whisper_model is None:
            self.logger.info("Loading Whisper model for subtitle generation...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model(
                config.WHISPER_MODEL,
                device=device,
                download_root=str(config.MODELS_DIR)
            )
            
            self.logger.info(f"Whisper model loaded on {device}")
        
        return self.whisper_model
    
    @retry_on_failure(max_retries=3)
    def generate_subtitles_from_audio(self, audio_path: Path, 
                                    language: str) -> Dict[str, Any]:
        """Generate SRT files from dubbed audio using Whisper"""
        self.logger.log_step_start(f"Subtitle Generation ({language})")
        
        try:
            # Load Whisper model
            model = self.load_whisper_model()
            
            # Set transcription options
            options = {
                "language": language,
                "task": "transcribe",
                "word_timestamps": True,
                "fp16": torch.cuda.is_available(),
                "temperature": 0.0,
                "beam_size": 5,
                "best_of": 5
            }
            
            self.logger.info(f"Generating subtitles for {language} audio...")
            
            # Perform transcription
            result = model.transcribe(str(audio_path), **options)
            
            # Process segments for subtitle formatting
            subtitle_segments = []
            
            for i, segment in enumerate(result["segments"]):
                # Clean and format text
                text = self._clean_subtitle_text(segment["text"])
                
                if len(text.strip()) < 2:  # Skip very short texts
                    continue
                
                # Apply line length optimization
                formatted_lines = self._format_subtitle_lines(text)
                
                subtitle_segment = {
                    "index": i + 1,
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["end"] - segment["start"],
                    "text": formatted_lines,
                    "word_count": len(text.split()),
                    "confidence": segment.get("avg_logprob", 0.0)
                }
                
                subtitle_segments.append(subtitle_segment)
            
            subtitle_data = {
                "video_name": self.video_name,
                "language": language,
                "total_segments": len(subtitle_segments),
                "total_duration": max(seg["end"] for seg in subtitle_segments) if subtitle_segments else 0,
                "segments": subtitle_segments,
                "generation_options": options
            }
            
            self.logger.info(f"Generated {len(subtitle_segments)} subtitle segments for {language}")
            
            self.logger.log_step_end(f"Subtitle Generation ({language})", True)
            return subtitle_data
            
        except Exception as e:
            self.logger.error(f"Subtitle generation for {language} failed: {str(e)}")
            raise
        finally:
            clear_gpu_memory()
    
    def _clean_subtitle_text(self, text: str) -> str:
        """Clean and normalize subtitle text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove filler words and sounds for subtitles
        filler_patterns = [
            r'\b(um|uh|er|ah|hmm)\b',
            r'\[.*?\]',  # Remove bracketed content
            r'\(.*?\)',  # Remove parenthetical content
        ]
        
        for pattern in filler_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def _format_subtitle_lines(self, text: str, max_chars_per_line: int = 42) -> str:
        """Format text into proper subtitle lines"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + (1 if current_line else 0)  # +1 for space
            
            if current_length + word_length <= max_chars_per_line:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Limit to 2 lines maximum
        if len(lines) > 2:
            # Try to redistribute words
            all_words = ' '.join(lines).split()
            mid_point = len(all_words) // 2
            
            line1 = ' '.join(all_words[:mid_point])
            line2 = ' '.join(all_words[mid_point:])
            
            # Check if lines are reasonable length
            if len(line1) <= max_chars_per_line * 1.2 and len(line2) <= max_chars_per_line * 1.2:
                lines = [line1, line2]
            else:
                lines = lines[:2]  # Just take first two lines
        
        return '\n'.join(lines)
    
    def adjust_subtitle_timing(self, subtitle_data: Dict[str, Any], 
                             synchronized_audio_path: Path) -> Dict[str, Any]:
        """Adjust subtitle timing to match dubbed audio precisely"""
        self.logger.log_step_start("Subtitle Timing Adjustment")
        
        try:
            # Load synchronized audio to get actual timing
            audio, sr = librosa.load(str(synchronized_audio_path), sr=config.AUDIO_SAMPLE_RATE)
            actual_duration = len(audio) / sr
            
            segments = subtitle_data["segments"]
            original_duration = subtitle_data["total_duration"]
            
            # Calculate timing adjustment factor
            timing_factor = actual_duration / original_duration if original_duration > 0 else 1.0
            
            # Adjust each segment timing
            adjusted_segments = []
            
            for segment in segments:
                adjusted_segment = {
                    **segment,
                    "start": segment["start"] * timing_factor,
                    "end": segment["end"] * timing_factor,
                    "duration": (segment["end"] - segment["start"]) * timing_factor,
                    "timing_adjusted": True,
                    "timing_factor": timing_factor
                }
                
                adjusted_segments.append(adjusted_segment)
            
            # Update subtitle data
            adjusted_subtitle_data = {
                **subtitle_data,
                "segments": adjusted_segments,
                "total_duration": actual_duration,
                "timing_adjustment_applied": True,
                "timing_factor": timing_factor
            }
            
            self.logger.info(f"Subtitle timing adjusted: factor={timing_factor:.3f}, "
                           f"duration {original_duration:.1f}s → {actual_duration:.1f}s")
            
            self.logger.log_step_end("Subtitle Timing Adjustment", True)
            return adjusted_subtitle_data
            
        except Exception as e:
            self.logger.error(f"Subtitle timing adjustment failed: {str(e)}")
            raise
    
    def create_srt_file(self, subtitle_data: Dict[str, Any], 
                       output_path: Path) -> Path:
        """Create SRT subtitle file"""
        self.logger.log_step_start(f"SRT File Creation")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in subtitle_data["segments"]:
                    # Format timestamps
                    start_time = self._seconds_to_srt_timestamp(segment["start"])
                    end_time = self._seconds_to_srt_timestamp(segment["end"])
                    
                    # Write SRT entry
                    f.write(f"{segment['index']}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment['text']}\n\n")
            
            self.logger.info(f"SRT file created: {output_path}")
            self.logger.log_step_end("SRT File Creation", True)
            return output_path
            
        except Exception as e:
            self.logger.error(f"SRT file creation failed: {str(e)}")
            raise
    
    def create_vtt_file(self, subtitle_data: Dict[str, Any], 
                       output_path: Path) -> Path:
        """Create VTT subtitle file"""
        self.logger.log_step_start("VTT File Creation")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                for segment in subtitle_data["segments"]:
                    # Format timestamps
                    start_time = self._seconds_to_vtt_timestamp(segment["start"])
                    end_time = self._seconds_to_vtt_timestamp(segment["end"])
                    
                    # Write VTT entry
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment['text']}\n\n")
            
            self.logger.info(f"VTT file created: {output_path}")
            self.logger.log_step_end("VTT File Creation", True)
            return output_path
            
        except Exception as e:
            self.logger.error(f"VTT file creation failed: {str(e)}")
            raise
    
    def _seconds_to_srt_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _seconds_to_vtt_timestamp(self, seconds: float) -> str:
        """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
    
    def create_original_subtitles(self, original_transcript: Dict[str, Any]) -> Path:
        """Create subtitles for original Arabic audio"""
        self.logger.log_step_start("Original Arabic Subtitle Creation")
        
        try:
            # Convert transcript to subtitle format
            segments = original_transcript["segments"]
            
            subtitle_segments = []
            for i, segment in enumerate(segments):
                subtitle_segment = {
                    "index": i + 1,
                    "start": segment["start_seconds"],
                    "end": segment["end_seconds"],
                    "text": self._format_subtitle_lines(segment["text"]),
                    "duration": segment["end_seconds"] - segment["start_seconds"]
                }
                subtitle_segments.append(subtitle_segment)
            
            subtitle_data = {
                "video_name": self.video_name,
                "language": "ar",
                "segments": subtitle_segments,
                "total_segments": len(subtitle_segments),
                "total_duration": max(seg["end"] for seg in subtitle_segments) if subtitle_segments else 0
            }
            
            # Create SRT file for original
            original_srt_path = config.OUTPUT_DIR / self.video_name / f"{self.video_name}_arabic.srt"
            original_srt_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.create_srt_file(subtitle_data, original_srt_path)
            
            self.logger.info(f"Original Arabic subtitles created: {len(subtitle_segments)} segments")
            self.logger.log_step_end("Original Arabic Subtitle Creation", True)
            
            return original_srt_path
            
        except Exception as e:
            self.logger.error(f"Original subtitle creation failed: {str(e)}")
            raise
    
    def embed_subtitles_in_video(self, video_path: Path, 
                               subtitle_files: Dict[str, Path]) -> Path:
        """Embed subtitles as soft subtitles using ffmpeg"""
        self.logger.log_step_start("Subtitle Embedding")
        
        try:
            output_path = config.TEMP_DIR / f"{self.video_name}_with_subtitles.mp4"
            
            # Build ffmpeg command
            cmd = ["ffmpeg", "-i", str(video_path)]
            
            # Add subtitle inputs
            for lang, subtitle_file in subtitle_files.items():
                cmd.extend(["-i", str(subtitle_file)])
            
            # Map video stream
            cmd.extend(["-map", "0:v"])
            
            # Map audio streams (assuming original audio is present)
            cmd.extend(["-map", "0:a"])
            
            # Map subtitle streams
            for i, (lang, _) in enumerate(subtitle_files.items()):
                cmd.extend(["-map", f"{i+1}:s"])
            
            # Set subtitle metadata
            for i, (lang, _) in enumerate(subtitle_files.items()):
                cmd.extend([f"-metadata:s:s:{i}", f"language={lang}"])
                cmd.extend([f"-metadata:s:s:{i}", f"title={lang.upper()} Subtitles"])
            
            # Output options
            cmd.extend([
                "-c:v", "copy",  # Copy video without re-encoding
                "-c:a", "copy",  # Copy audio without re-encoding
                "-c:s", "mov_text",  # Use mov_text for MP4 subtitles
                "-y",  # Overwrite output file
                str(output_path)
            ])
            
            self.logger.info(f"Embedding subtitles with command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / 1024**2
                self.logger.info(f"Subtitles embedded successfully: {file_size_mb:.1f}MB")
                
                self.logger.log_step_end("Subtitle Embedding", True)
                return output_path
            else:
                raise FileNotFoundError("Subtitle embedding failed - output file not created")
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg subtitle embedding error: {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"Subtitle embedding failed: {str(e)}")
            raise
    
    def validate_subtitle_timing(self, subtitle_file: Path, 
                               audio_duration: float) -> Dict[str, Any]:
        """Validate subtitle timing and readability"""
        self.logger.log_step_start("Subtitle Validation")
        
        try:
            # Parse SRT file
            segments = self._parse_srt_file(subtitle_file)
            
            validation_results = {
                "total_segments": len(segments),
                "timing_issues": [],
                "readability_issues": [],
                "duration_match": True,
                "validation_passed": True
            }
            
            for segment in segments:
                # Check timing issues
                if segment["duration"] < 0.5:
                    validation_results["timing_issues"].append({
                        "segment": segment["index"],
                        "issue": "Too short",
                        "duration": segment["duration"]
                    })
                
                if segment["duration"] > 7.0:
                    validation_results["timing_issues"].append({
                        "segment": segment["index"],
                        "issue": "Too long",
                        "duration": segment["duration"]
                    })
                
                # Check readability
                words_per_second = len(segment["text"].split()) / segment["duration"]
                if words_per_second > 3.0:  # Reading speed too fast
                    validation_results["readability_issues"].append({
                        "segment": segment["index"],
                        "issue": "Reading speed too fast",
                        "words_per_second": words_per_second
                    })
            
            # Check if subtitle duration matches audio
            if segments:
                subtitle_duration = max(seg["end"] for seg in segments)
                duration_diff = abs(subtitle_duration - audio_duration)
                
                if duration_diff > 5.0:  # More than 5 seconds difference
                    validation_results["duration_match"] = False
                    validation_results["duration_difference"] = duration_diff
            
            # Overall validation
            if (validation_results["timing_issues"] or 
                validation_results["readability_issues"] or 
                not validation_results["duration_match"]):
                validation_results["validation_passed"] = False
            
            self.logger.info(f"Subtitle validation: {len(validation_results['timing_issues'])} timing issues, "
                           f"{len(validation_results['readability_issues'])} readability issues")
            
            self.logger.log_step_end("Subtitle Validation", True)
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Subtitle validation failed: {str(e)}")
            raise
    
    def _parse_srt_file(self, srt_file: Path) -> List[Dict[str, Any]]:
        """Parse SRT file and return segments"""
        segments = []
        
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Split by double newlines
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
    
    def process_subtitles(self, synchronized_audio_files: Dict[str, Path],
                        original_transcript: Dict[str, Any]) -> Dict[str, Path]:
        """Complete subtitle processing pipeline"""
        self.logger.info(f"Starting subtitle processing for {self.video_name}")
        
        results = {}
        
        try:
            # Create output directory
            output_dir = config.OUTPUT_DIR / self.video_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create original Arabic subtitles
            original_srt = self.create_original_subtitles(original_transcript)
            results["ar"] = original_srt
            
            # Process each dubbed language
            for language, audio_path in synchronized_audio_files.items():
                self.logger.info(f"Processing subtitles for {language}")
                
                # Generate subtitles from dubbed audio
                subtitle_data = self.generate_subtitles_from_audio(audio_path, language)
                
                # Adjust timing to match synchronized audio
                adjusted_subtitle_data = self.adjust_subtitle_timing(subtitle_data, audio_path)
                
                # Create SRT file
                srt_path = output_dir / f"{self.video_name}_{language}.srt"
                self.create_srt_file(adjusted_subtitle_data, srt_path)
                
                # Create VTT file
                vtt_path = output_dir / f"{self.video_name}_{language}.vtt"
                self.create_vtt_file(adjusted_subtitle_data, vtt_path)
                
                # Validate subtitles
                audio, sr = librosa.load(str(audio_path), sr=config.AUDIO_SAMPLE_RATE)
                audio_duration = len(audio) / sr
                validation_results = self.validate_subtitle_timing(srt_path, audio_duration)
                
                results[language] = srt_path
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(f"subtitles_{language}", {
                    "srt_file": str(srt_path),
                    "vtt_file": str(vtt_path),
                    "validation_passed": validation_results["validation_passed"],
                    "segments_count": adjusted_subtitle_data["total_segments"]
                })
                
                self.logger.info(f"Subtitles for {language} completed: "
                               f"{adjusted_subtitle_data['total_segments']} segments")
            
            # Mark step as completed
            self.checkpoint_manager.mark_step_completed("subtitle_processing")
            
            self.logger.info("Subtitle processing completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Subtitle processing failed: {str(e)}")
            raise
        finally:
            # Clean up models to free memory
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
            clear_gpu_memory()
