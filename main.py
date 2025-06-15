"""
Enhanced Video Dubbing Automation - Main Orchestration System
Complete pipeline for Arabic to English/German video dubbing
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import gc

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import config
from utils import Logger, CheckpointManager, create_processing_report, estimate_processing_time, clear_gpu_memory
from setup import setup_environment

# Import processing modules
from step1_audio_processing import AudioProcessor
from step2_transcription import TranscriptionProcessor
from step3_translation import TranslationProcessor
from step4_voice_cloning import VoiceCloningProcessor
from step5_synchronization import SynchronizationProcessor
from step6_subtitles import SubtitleProcessor
from step7_quality_assurance import QualityAssuranceProcessor

class VideoDubbingOrchestrator:
    """Main orchestrator for the complete video dubbing pipeline"""
    
    def __init__(self, local_mode: bool = False):
        """Initialize the orchestrator"""
        # Set up configuration
        config.__init__(local_mode)
        
        self.logger = Logger("orchestrator")
        self.start_time = datetime.now()
        self.processing_errors = []
        self.step_times = {}
        
        # Initialize processors (will be created per video)
        self.processors = {}
        
        # Track overall progress
        self.total_videos = 0
        self.completed_videos = 0
        self.failed_videos = []
        
    def setup_environment(self) -> bool:
        """Setup the complete environment"""
        self.logger.info("Setting up Enhanced Video Dubbing environment...")
        
        try:
            success = setup_environment()
            if success:
                self.logger.info("Environment setup completed successfully")
                return True
            else:
                self.logger.error("Environment setup failed")
                return False
        except Exception as e:
            self.logger.error(f"Environment setup error: {str(e)}")
            return False
    
    def discover_videos(self, input_directory: Optional[Path] = None) -> List[Path]:
        """Discover video files in input directory"""
        self.logger.info("Discovering video files...")
        
        search_dir = input_directory or config.INPUT_DIR
        
        if not search_dir.exists():
            self.logger.error(f"Input directory does not exist: {search_dir}")
            return []
        
        video_files = []
        
        for format_ext in config.SUPPORTED_VIDEO_FORMATS:
            pattern = f"*{format_ext}"
            found_files = list(search_dir.glob(pattern))
            video_files.extend(found_files)
        
        # Filter by file size
        valid_videos = []
        for video_file in video_files:
            file_size_gb = video_file.stat().st_size / (1024**3)
            if file_size_gb <= config.MAX_FILE_SIZE_GB:
                valid_videos.append(video_file)
                self.logger.info(f"Found video: {video_file.name} ({file_size_gb:.1f}GB)")
            else:
                self.logger.warning(f"Skipping oversized video: {video_file.name} ({file_size_gb:.1f}GB)")
        
        self.total_videos = len(valid_videos)
        self.logger.info(f"Discovered {len(valid_videos)} valid video files")
        
        return valid_videos
    
    def estimate_total_processing_time(self, video_files: List[Path]) -> Dict[str, float]:
        """Estimate total processing time for all videos"""
        self.logger.info("Estimating processing time...")
        
        total_estimates = {
            "audio_extraction": 0,
            "transcription": 0,
            "translation": 0,
            "voice_synthesis": 0,
            "synchronization": 0,
            "subtitle_generation": 0,
            "final_assembly": 0,
            "total": 0
        }
        
        for video_file in video_files:
            try:
                # Get video duration (rough estimate)
                import subprocess
                cmd = [
                    "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(video_file)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                duration_seconds = float(result.stdout.strip())
                duration_minutes = duration_seconds / 60
                
                # Get estimates for this video
                video_estimates = estimate_processing_time(duration_minutes)
                
                # Add to totals
                for step, time_est in video_estimates.items():
                    if step in total_estimates:
                        total_estimates[step] += time_est
                
            except Exception as e:
                self.logger.warning(f"Could not estimate time for {video_file.name}: {str(e)}")
                # Use default estimate (90 minutes average)
                default_estimates = estimate_processing_time(90)
                for step, time_est in default_estimates.items():
                    if step in total_estimates:
                        total_estimates[step] += time_est
        
        # Convert to hours for display
        total_estimates_hours = {k: v/60 for k, v in total_estimates.items()}
        
        self.logger.info(f"Estimated total processing time: {total_estimates_hours['total']:.1f} hours")
        
        return total_estimates_hours
    
    def process_single_video(self, video_path: Path, target_languages: List[str] = None) -> Dict[str, Any]:
        """Process a single video through the complete pipeline"""
        video_name = video_path.stem
        self.logger.info(f"Starting processing for video: {video_name}")
        
        # Initialize processing result
        processing_result = {
            "video_name": video_name,
            "video_path": str(video_path),
            "start_time": datetime.now().isoformat(),
            "target_languages": target_languages or config.TARGET_LANGUAGES,
            "status": "in_progress",
            "steps_completed": [],
            "outputs": {},
            "errors": [],
            "step_times": {}
        }
        
        try:
            # Create processors for this video
            processors = {
                "audio": AudioProcessor(video_name),
                "transcription": TranscriptionProcessor(video_name),
                "translation": TranslationProcessor(video_name),
                "voice_cloning": VoiceCloningProcessor(video_name),
                "synchronization": SynchronizationProcessor(video_name),
                "subtitles": SubtitleProcessor(video_name),
                "quality_assurance": QualityAssuranceProcessor(video_name)
            }
            
            # Step 1: Audio Processing
            step_start = time.time()
            self.logger.info("🎵 Step 1: Audio Processing")
            
            # Extract and clean audio
            raw_audio = processors["audio"].extract_audio_from_video(video_path)
            clean_audio = processors["audio"].apply_noise_reduction(raw_audio)
            speech_segments = processors["audio"].detect_speech_segments(clean_audio)
            
            processing_result["steps_completed"].append("audio_processing")
            processing_result["step_times"]["audio_processing"] = time.time() - step_start
            processing_result["outputs"]["clean_audio"] = str(clean_audio)
            processing_result["outputs"]["speech_segments"] = len(speech_segments)
            
            # Progress update
            self._update_progress("Audio processing completed", 1, 8)
            
            # Step 2: Transcription
            step_start = time.time()
            self.logger.info("📝 Step 2: Transcription")
            
            transcription_result = processors["transcription"].transcribe_with_diarization(
                clean_audio, speech_segments
            )
            
            processing_result["steps_completed"].append("transcription")
            processing_result["step_times"]["transcription"] = time.time() - step_start
            processing_result["outputs"]["transcription"] = transcription_result
            
            # Progress update
            self._update_progress("Transcription completed", 2, 8)
            
            # Step 3: Translation
            step_start = time.time()
            self.logger.info("🌐 Step 3: Translation")
            
            translations = {}
            for target_lang in target_languages or config.TARGET_LANGUAGES:
                translation_result = processors["translation"].translate_segments(
                    transcription_result, target_lang
                )
                translations[target_lang] = translation_result
            
            processing_result["steps_completed"].append("translation")
            processing_result["step_times"]["translation"] = time.time() - step_start
            processing_result["outputs"]["translations"] = translations
            
            # Progress update
            self._update_progress("Translation completed", 3, 8)
            
            # Step 4: Voice Cloning
            step_start = time.time()
            self.logger.info("🗣️ Step 4: Voice Cloning")
            
            dubbed_audio = {}
            for target_lang in target_languages or config.TARGET_LANGUAGES:
                cloned_audio = processors["voice_cloning"].synthesize_speech(
                    translations[target_lang], target_lang, clean_audio
                )
                dubbed_audio[target_lang] = cloned_audio
            
            processing_result["steps_completed"].append("voice_cloning")
            processing_result["step_times"]["voice_cloning"] = time.time() - step_start
            processing_result["outputs"]["dubbed_audio"] = dubbed_audio
            
            # Progress update
            self._update_progress("Voice cloning completed", 4, 8)
            
            # Step 5: Synchronization
            step_start = time.time()
            self.logger.info("⏱️ Step 5: Synchronization")
            
            synchronized_audio = {}
            for target_lang in target_languages or config.TARGET_LANGUAGES:
                sync_result = processors["synchronization"].synchronize_audio_video(
                    dubbed_audio[target_lang], video_path, translations[target_lang]
                )
                synchronized_audio[target_lang] = sync_result
            
            processing_result["steps_completed"].append("synchronization")
            processing_result["step_times"]["synchronization"] = time.time() - step_start
            processing_result["outputs"]["synchronized_audio"] = synchronized_audio
            
            # Progress update
            self._update_progress("Synchronization completed", 5, 8)
            
            # Step 6: Subtitles
            step_start = time.time()
            self.logger.info("📋 Step 6: Subtitle Generation")
            
            subtitle_files = {}
            for target_lang in target_languages or config.TARGET_LANGUAGES:
                subtitles = processors["subtitles"].generate_subtitles(
                    synchronized_audio[target_lang], translations[target_lang]
                )
                subtitle_files[target_lang] = subtitles
            
            processing_result["steps_completed"].append("subtitles")
            processing_result["step_times"]["subtitles"] = time.time() - step_start
            processing_result["outputs"]["subtitle_files"] = subtitle_files
            
            # Progress update
            self._update_progress("Subtitle generation completed", 6, 8)
            
            # Step 7: Quality Assurance
            step_start = time.time()
            self.logger.info("✅ Step 7: Quality Assurance")
            
            qa_results = {}
            for target_lang in target_languages or config.TARGET_LANGUAGES:
                qa_result = processors["quality_assurance"].validate_processing(
                    video_path, synchronized_audio[target_lang], 
                    subtitle_files[target_lang], translations[target_lang]
                )
                qa_results[target_lang] = qa_result
            
            processing_result["steps_completed"].append("quality_assurance")
            processing_result["step_times"]["quality_assurance"] = time.time() - step_start
            processing_result["outputs"]["qa_results"] = qa_results
            
            # Progress update
            self._update_progress("Quality assurance completed", 7, 8)
            
            # Step 8: Final Assembly
            step_start = time.time()
            self.logger.info("🎬 Step 8: Final Assembly")
            
            final_videos = {}
            for target_lang in target_languages or config.TARGET_LANGUAGES:
                final_video = processors["quality_assurance"].assemble_final_video(
                    video_path, synchronized_audio[target_lang], 
                    subtitle_files[target_lang], target_lang
                )
                final_videos[target_lang] = final_video
            
            processing_result["steps_completed"].append("final_assembly")
            processing_result["step_times"]["final_assembly"] = time.time() - step_start
            processing_result["outputs"]["final_videos"] = final_videos
            
            # Progress update
            self._update_progress("Final assembly completed", 8, 8)
            
            # Mark as completed
            processing_result["status"] = "completed"
            processing_result["end_time"] = datetime.now().isoformat()
            processing_result["total_time_seconds"] = sum(processing_result["step_times"].values())
            
            self.completed_videos += 1
            self.logger.info(f"✅ Video processing completed successfully: {video_name}")
            
            return processing_result
            
        except Exception as e:
            error_msg = f"Processing failed for {video_name}: {str(e)}"
            self.logger.error(error_msg)
            
            processing_result["status"] = "failed"
            processing_result["error"] = error_msg
            processing_result["end_time"] = datetime.now().isoformat()
            
            self.failed_videos.append(video_name)
            self.processing_errors.append(error_msg)
            
            return processing_result
        
        finally:
            # Cleanup GPU memory
            clear_gpu_memory()
    
    def _update_progress(self, message: str, current_step: int, total_steps: int):
        """Update processing progress"""
        progress_percent = (current_step / total_steps) * 100
        self.logger.info(f"Progress: {progress_percent:.1f}% - {message}")
        
        # Update step tracking
        if current_step not in self.step_times:
            self.step_times[current_step] = time.time()

    def process_batch(self, video_files: List[Path]) -> Dict[str, Any]:
        """Process a batch of videos sequentially"""
        self.logger.info(f"Starting batch processing of {len(video_files)} videos")
        
        batch_results = {}
        
        for i, video_file in enumerate(video_files, 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"PROCESSING VIDEO {i}/{len(video_files)}: {video_file.name}")
            self.logger.info(f"{'='*80}")
            
            try:
                video_results = self.process_single_video(video_file)
                batch_results[video_file.stem] = {
                    "status": "completed",
                    "results": video_results
                }
                
                # Save batch progress
                self._save_batch_progress(batch_results)
                
            except Exception as e:
                self.logger.error(f"Failed to process {video_file.name}: {str(e)}")
                batch_results[video_file.stem] = {
                    "status": "failed",
                    "error": str(e)
                }
                
                # Continue with next video despite failure
                continue
            
            # Check session time limit
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            if elapsed_time > config.MAX_SESSION_TIME:
                self.logger.warning("Approaching session time limit - stopping batch processing")
                break
        
        return batch_results
    
    def _save_batch_progress(self, batch_results: Dict[str, Any]):
        """Save batch processing progress"""
        progress_file = config.OUTPUT_DIR / "batch_processing_progress.json"
        
        progress_data = {
            "start_time": self.start_time.isoformat(),
            "current_time": datetime.now().isoformat(),
            "total_videos": self.total_videos,
            "completed_videos": self.completed_videos,
            "failed_videos": len(self.failed_videos),
            "step_times": self.step_times,
            "failed_video_details": self.failed_videos,
            "batch_results": batch_results
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def generate_final_report(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final processing report"""
        self.logger.info("Generating final processing report...")
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds() / 60  # minutes
        
        # Calculate success rate
        successful_videos = sum(1 for result in batch_results.values() 
                              if result["status"] == "completed")
        success_rate = successful_videos / len(batch_results) if batch_results else 0
        
        # Generate comprehensive report
        final_report = create_processing_report(
            "batch_processing",
            self.start_time,
            end_time,
            "completed" if success_rate > 0.5 else "partially_failed",
            self.step_times,
            self.processing_errors
        )
        
        # Add batch-specific information
        final_report.update({
            "total_videos_processed": len(batch_results),
            "successful_videos": successful_videos,
            "failed_videos": len(self.failed_videos),
            "success_rate": success_rate,
            "failed_video_details": self.failed_videos,
            "average_processing_time_per_video": total_duration / len(batch_results) if batch_results else 0,
            "batch_results_summary": {
                video_name: result["status"] for video_name, result in batch_results.items()
            }
        })
        
        # Save final report
        report_file = config.OUTPUT_DIR / "batch_processing_summary.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        self.logger.info(f"Final report saved to: {report_file}")
        self.logger.info(f"Processing completed: {successful_videos}/{len(batch_results)} videos successful")
        
        return final_report
    
    def run_complete_pipeline(self, input_directory: Optional[Path] = None) -> Dict[str, Any]:
        """Run the complete video dubbing pipeline"""
        self.logger.info("Starting Enhanced Video Dubbing Automation Pipeline")
        self.logger.info("=" * 80)
        
        try:
            # Step 0: Environment Setup
            if not self.setup_environment():
                raise RuntimeError("Environment setup failed")
            
            # Discover videos
            video_files = self.discover_videos(input_directory)
            if not video_files:
                raise RuntimeError("No valid video files found")
            
            # Estimate processing time
            time_estimates = self.estimate_total_processing_time(video_files)
            
            self.logger.info(f"Estimated processing time: {time_estimates['total']:.1f} hours")
            
            if time_estimates['total'] > 11:  # More than 11 hours
                self.logger.warning("Estimated processing time exceeds Kaggle session limit")
                self.logger.info("Consider processing videos in smaller batches")
            
            # Process batch
            batch_results = self.process_batch(video_files)
            
            # Generate final report
            final_report = self.generate_final_report(batch_results)
            
            self.logger.info("Enhanced Video Dubbing Pipeline completed successfully!")
            return final_report
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Video Dubbing Automation")
    parser.add_argument("--input-dir", type=Path, help="Input directory containing videos")
    parser.add_argument("--local", action="store_true", help="Run in local mode (not Kaggle)")
    parser.add_argument("--single-video", type=Path, help="Process a single video file")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = VideoDubbingOrchestrator(local_mode=args.local)
    
    try:
        if args.single_video:
            # Process single video
            if not args.single_video.exists():
                print(f"Error: Video file not found: {args.single_video}")
                return 1
            
            orchestrator.setup_environment()
            results = orchestrator.process_single_video(args.single_video)
            print(f"Single video processing completed: {args.single_video.name}")
            
        else:
            # Process batch
            results = orchestrator.run_complete_pipeline(args.input_dir)
            print(f"Batch processing completed: {results['successful_videos']}/{results['total_videos_processed']} videos successful")
        
        return 0
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
