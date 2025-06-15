"""
Step 2: Enhanced Transcription with Speaker Diarization Module
Enhanced Video Dubbing Automation Project
"""

import torch
import whisper
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import librosa
from speechbrain.pretrained import SpeakerRecognition
import gc

from config import config
from utils import Logger, CheckpointManager, retry_on_failure, clear_gpu_memory, safe_model_loading

class TranscriptionProcessor:
    """Handle transcription and speaker diarization"""
    
    def __init__(self, video_name: str):
        self.video_name = video_name
        self.logger = Logger(video_name)
        self.checkpoint_manager = CheckpointManager(video_name)
        self.whisper_model = None
        self.speaker_model = None
        
    @safe_model_loading
    def load_whisper_model(self):
        """Load Whisper model with GPU optimization"""
        if self.whisper_model is None:
            self.logger.info("Loading Whisper large-v3 model...")
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model with caching
            model_path = config.MODELS_DIR / "whisper"
            self.whisper_model = whisper.load_model(
                config.WHISPER_MODEL,
                device=device,
                download_root=str(model_path)
            )
            
            self.logger.info(f"Whisper model loaded on {device}")
        
        return self.whisper_model
    
    @safe_model_loading
    def load_speaker_diarization_model(self):
        """Load SpeechBrain speaker diarization model"""
        if self.speaker_model is None:
            self.logger.info("Loading SpeechBrain speaker recognition model...")
            
            try:
                self.speaker_model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(config.MODELS_DIR / "speechbrain"),
                    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                )
                self.logger.info("SpeechBrain model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load SpeechBrain model: {str(e)}")
                self.speaker_model = None
        
        return self.speaker_model
    
    @retry_on_failure(max_retries=3)
    def transcribe_with_whisper(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe audio using Whisper with precise timestamps"""
        self.logger.log_step_start("Whisper Transcription")
        
        try:
            # Load Whisper model
            model = self.load_whisper_model()
            
            # Transcription options for Arabic
            options = {
                "language": "ar",  # Arabic language
                "task": "transcribe",
                "word_timestamps": True,
                "fp16": torch.cuda.is_available(),  # Use FP16 if CUDA available
                "temperature": 0.0,  # Deterministic output
                "beam_size": 5,
                "best_of": 5,
                "patience": 1.0
            }
            
            self.logger.info(f"Starting transcription with options: {options}")
            
            # Perform transcription
            result = model.transcribe(str(audio_path), **options)
            
            # Process segments for better structure
            processed_segments = []
            
            for segment in result["segments"]:
                processed_segment = {
                    "id": segment["id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "confidence": segment.get("avg_logprob", 0.0),
                    "no_speech_prob": segment.get("no_speech_prob", 0.0),
                    "words": []
                }
                
                # Add word-level timestamps if available
                if "words" in segment:
                    for word in segment["words"]:
                        processed_segment["words"].append({
                            "word": word["word"],
                            "start": word["start"],
                            "end": word["end"],
                            "probability": word.get("probability", 1.0)
                        })
                
                processed_segments.append(processed_segment)
            
            # Create final transcription result
            transcription_result = {
                "language": result["language"],
                "text": result["text"],
                "segments": processed_segments,
                "duration": max(seg["end"] for seg in processed_segments) if processed_segments else 0,
                "word_count": len(result["text"].split()),
                "segment_count": len(processed_segments)
            }
            
            # Save transcription
            transcription_file = config.TEMP_DIR / f"{self.video_name}_transcription.json"
            with open(transcription_file, 'w', encoding='utf-8') as f:
                json.dump(transcription_result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Transcription completed: {len(processed_segments)} segments, "
                           f"{transcription_result['word_count']} words")
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint("transcription", {
                "transcription_file": str(transcription_file),
                "segment_count": len(processed_segments),
                "duration": transcription_result["duration"]
            })
            
            self.logger.log_step_end("Whisper Transcription", True)
            return transcription_result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise
        finally:
            clear_gpu_memory()
    
    def perform_speaker_diarization(self, audio_path: Path, 
                                  transcription: Dict[str, Any]) -> Dict[str, Any]:
        """Perform speaker diarization using SpeechBrain"""
        self.logger.log_step_start("Speaker Diarization")
        
        try:
            # Load speaker model
            speaker_model = self.load_speaker_diarization_model()
            
            if speaker_model is None:
                self.logger.warning("Speaker diarization model not available, using energy-based method")
                return self._energy_based_speaker_detection(audio_path, transcription)
            
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=16000)  # SpeechBrain expects 16kHz
            
            # Process each segment for speaker identification
            speaker_segments = []
            speaker_embeddings = {}
            current_speaker_id = 0
            
            for segment in transcription["segments"]:
                start_sample = int(segment["start"] * sr)
                end_sample = int(segment["end"] * sr)
                
                if end_sample > len(audio):
                    end_sample = len(audio)
                
                segment_audio = audio[start_sample:end_sample]
                
                # Skip very short segments
                if len(segment_audio) < sr * 0.5:  # Less than 0.5 seconds
                    speaker_segments.append({
                        **segment,
                        "speaker": "UNKNOWN",
                        "speaker_confidence": 0.0
                    })
                    continue
                
                try:
                    # Get speaker embedding
                    embedding = speaker_model.encode_batch(
                        torch.tensor(segment_audio).unsqueeze(0)
                    ).squeeze()
                    
                    # Find matching speaker or create new one
                    speaker_id, confidence = self._match_speaker(
                        embedding, speaker_embeddings, threshold=0.75
                    )
                    
                    if speaker_id is None:
                        # New speaker
                        speaker_id = f"SPEAKER_{current_speaker_id:02d}"
                        speaker_embeddings[speaker_id] = embedding
                        current_speaker_id += 1
                        confidence = 1.0
                    
                    speaker_segments.append({
                        **segment,
                        "speaker": speaker_id,
                        "speaker_confidence": float(confidence)
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Speaker diarization failed for segment {segment['id']}: {str(e)}")
                    speaker_segments.append({
                        **segment,
                        "speaker": "UNKNOWN",
                        "speaker_confidence": 0.0
                    })
            
            # Identify main speaker (speaker with most speaking time)
            speaker_times = {}
            for segment in speaker_segments:
                speaker = segment["speaker"]
                duration = segment["end"] - segment["start"]
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
            
            main_speaker = max(speaker_times, key=speaker_times.get) if speaker_times else "SPEAKER_00"
            
            # Create diarization result
            diarization_result = {
                "segments": speaker_segments,
                "speakers": list(speaker_embeddings.keys()),
                "main_speaker": main_speaker,
                "speaker_times": speaker_times,
                "total_speakers": len(speaker_embeddings)
            }
            
            self.logger.info(f"Speaker diarization completed: {len(speaker_embeddings)} speakers detected")
            self.logger.info(f"Main speaker: {main_speaker} ({speaker_times.get(main_speaker, 0):.1f}s)")
            
            # Save diarization results
            diarization_file = config.TEMP_DIR / f"{self.video_name}_diarization.json"
            with open(diarization_file, 'w', encoding='utf-8') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_result = self._make_json_serializable(diarization_result)
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint("speaker_diarization", {
                "diarization_file": str(diarization_file),
                "total_speakers": len(speaker_embeddings),
                "main_speaker": main_speaker
            })
            
            self.logger.log_step_end("Speaker Diarization", True)
            return diarization_result
            
        except Exception as e:
            self.logger.error(f"Speaker diarization failed: {str(e)}")
            # Fallback to energy-based detection
            return self._energy_based_speaker_detection(audio_path, transcription)
        finally:
            clear_gpu_memory()
    
    def _match_speaker(self, embedding: torch.Tensor, 
                      speaker_embeddings: Dict[str, torch.Tensor], 
                      threshold: float = 0.75) -> Tuple[str, float]:
        """Match embedding to existing speaker or identify as new speaker"""
        if not speaker_embeddings:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for speaker_id, stored_embedding in speaker_embeddings.items():
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0), 
                stored_embedding.unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id
        
        if best_similarity >= threshold:
            return best_match, best_similarity
        else:
            return None, 0.0
    
    def _energy_based_speaker_detection(self, audio_path: Path, 
                                      transcription: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback energy-based speaker detection"""
        self.logger.info("Using energy-based speaker detection fallback")
        
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=config.AUDIO_SAMPLE_RATE)
        
        # Calculate energy for each segment
        energy_segments = []
        for segment in transcription["segments"]:
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)
            
            if end_sample > len(audio):
                end_sample = len(audio)
            
            segment_audio = audio[start_sample:end_sample]
            energy = np.mean(segment_audio**2) if len(segment_audio) > 0 else 0.0
            
            energy_segments.append({
                **segment,
                "energy": float(energy),
                "speaker": "MAIN_SPEAKER",  # Assume main speaker for now
                "speaker_confidence": 0.5
            })
        
        return {
            "segments": energy_segments,
            "speakers": ["MAIN_SPEAKER"],
            "main_speaker": "MAIN_SPEAKER",
            "speaker_times": {"MAIN_SPEAKER": sum(seg["end"] - seg["start"] for seg in energy_segments)},
            "total_speakers": 1,
            "method": "energy_based"
        }
    
    def _make_json_serializable(self, obj):
        """Convert tensors and other non-serializable objects to JSON-serializable format"""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def filter_main_speaker_content(self, diarization_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter segments to include only main speaker content"""
        self.logger.log_step_start("Main Speaker Filtering")
        
        main_speaker = diarization_result["main_speaker"]
        main_speaker_segments = []
        
        for segment in diarization_result["segments"]:
            if segment["speaker"] == main_speaker:
                # Additional filtering criteria
                if (segment["speaker_confidence"] >= 0.5 and 
                    segment.get("confidence", 0) >= -0.5 and  # Whisper confidence
                    segment.get("no_speech_prob", 1.0) <= 0.3):  # Low no-speech probability
                    
                    main_speaker_segments.append(segment)
        
        self.logger.info(f"Filtered to {len(main_speaker_segments)} main speaker segments "
                        f"from {len(diarization_result['segments'])} total segments")
        
        # Save filtered segments
        filtered_file = config.TEMP_DIR / f"{self.video_name}_main_speaker_segments.json"
        with open(filtered_file, 'w', encoding='utf-8') as f:
            json.dump(main_speaker_segments, f, indent=2, ensure_ascii=False)
        
        self.logger.log_step_end("Main Speaker Filtering", True)
        return main_speaker_segments
    
    def create_timestamped_transcript(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create final timestamped transcript with speaker labels"""
        self.logger.log_step_start("Timestamped Transcript Creation")
        
        # Create formatted transcript
        transcript_lines = []
        full_text_parts = []
        
        for segment in segments:
            # Format timestamp
            start_time = self._format_timestamp(segment["start"])
            end_time = self._format_timestamp(segment["end"])
            
            # Create transcript line
            transcript_line = {
                "id": segment["id"],
                "timestamp": f"{start_time} --> {end_time}",
                "start_seconds": segment["start"],
                "end_seconds": segment["end"],
                "speaker": segment["speaker"],
                "text": segment["text"],
                "confidence": segment.get("confidence", 0.0),
                "speaker_confidence": segment.get("speaker_confidence", 0.0)
            }
            
            transcript_lines.append(transcript_line)
            full_text_parts.append(segment["text"])
        
        # Create final transcript
        final_transcript = {
            "video_name": self.video_name,
            "language": "ar",
            "total_duration": max(seg["end_seconds"] for seg in transcript_lines) if transcript_lines else 0,
            "total_segments": len(transcript_lines),
            "full_text": " ".join(full_text_parts),
            "segments": transcript_lines,
            "created_at": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
        }
        
        # Save timestamped transcript
        transcript_file = config.TEMP_DIR / f"{self.video_name}_timestamped_transcript.json"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            json.dump(final_transcript, f, indent=2, ensure_ascii=False)
        
        # Create backup
        backup_file = config.TEMP_DIR / f"{self.video_name}_transcript_backup.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(final_transcript, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Timestamped transcript created with {len(transcript_lines)} segments")
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint("timestamped_transcript", {
            "transcript_file": str(transcript_file),
            "backup_file": str(backup_file),
            "total_segments": len(transcript_lines),
            "total_duration": final_transcript["total_duration"]
        })
        
        self.logger.log_step_end("Timestamped Transcript Creation", True)
        return final_transcript
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in HH:MM:SS.mmm format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def process_transcription(self, audio_path: Path) -> Dict[str, Any]:
        """Complete transcription processing pipeline"""
        self.logger.info(f"Starting transcription processing for {self.video_name}")
        
        try:
            # Step 1: Whisper transcription
            transcription = self.transcribe_with_whisper(audio_path)
            
            # Step 2: Speaker diarization
            diarization = self.perform_speaker_diarization(audio_path, transcription)
            
            # Step 3: Filter main speaker content
            main_speaker_segments = self.filter_main_speaker_content(diarization)
            
            # Step 4: Create timestamped transcript
            final_transcript = self.create_timestamped_transcript(main_speaker_segments)
            
            # Mark step as completed
            self.checkpoint_manager.mark_step_completed("transcription_processing")
            
            self.logger.info("Transcription processing completed successfully")
            
            return {
                "transcription": transcription,
                "diarization": diarization,
                "main_speaker_segments": main_speaker_segments,
                "final_transcript": final_transcript
            }
            
        except Exception as e:
            self.logger.error(f"Transcription processing failed: {str(e)}")
            raise
        finally:
            # Clean up models to free memory
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
            if self.speaker_model is not None:
                del self.speaker_model
                self.speaker_model = None
            clear_gpu_memory()
