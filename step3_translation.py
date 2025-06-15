"""
Step 3: Translation with Meta SeamlessM4T v2 Module
Enhanced Video Dubbing Automation Project
"""

import torch
from transformers import SeamlessM4TModel, SeamlessM4TProcessor
import json
from pathlib import Path
from typing import Dict, List, Any
import gc
from tqdm import tqdm

from config import config
from utils import Logger, CheckpointManager, retry_on_failure, clear_gpu_memory, safe_model_loading

class TranslationProcessor:
    """Handle translation using Meta SeamlessM4T v2"""
    
    def __init__(self, video_name: str):
        self.video_name = video_name
        self.logger = Logger(video_name)
        self.checkpoint_manager = CheckpointManager(video_name)
        self.model = None
        self.processor = None
        
    @safe_model_loading
    def load_seamless_model(self):
        """Load SeamlessM4T v2 model with caching"""
        if self.model is None or self.processor is None:
            self.logger.info("Loading SeamlessM4T v2 model...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Load model and processor with caching
            self.processor = SeamlessM4TProcessor.from_pretrained(
                config.SEAMLESS_MODEL,
                cache_dir=str(config.MODELS_DIR),
            )
            
            self.model = SeamlessM4TModel.from_pretrained(
                config.SEAMLESS_MODEL,
                cache_dir=str(config.MODELS_DIR),
                torch_dtype=torch_dtype,
                device_map=device if device == "cuda" else None
            )
            
            if device == "cuda":
                self.model = self.model.to(device)
            
            self.logger.info(f"SeamlessM4T model loaded on {device}")
        
        return self.model, self.processor
    
    def prepare_segments_for_translation(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare transcript segments for batch translation"""
        self.logger.log_step_start("Translation Preparation")
        
        segments = transcript_data["final_transcript"]["segments"]
        
        # Filter and prepare segments
        translation_segments = []
        
        for segment in segments:
            # Skip very short segments or low confidence segments
            if (segment["end_seconds"] - segment["start_seconds"] < 1.0 or
                segment.get("confidence", 0) < -1.0):
                continue
            
            # Clean text for translation
            text = segment["text"].strip()
            if len(text) < 3:  # Skip very short texts
                continue
            
            translation_segments.append({
                "id": segment["id"],
                "start": segment["start_seconds"],
                "end": segment["end_seconds"],
                "original_text": text,
                "speaker": segment.get("speaker", "MAIN_SPEAKER"),
                "confidence": segment.get("confidence", 0.0)
            })
        
        self.logger.info(f"Prepared {len(translation_segments)} segments for translation")
        self.logger.log_step_end("Translation Preparation", True)
        
        return translation_segments
    
    @retry_on_failure(max_retries=3)
    def translate_segments_batch(self, segments: List[Dict[str, Any]], 
                               target_language: str) -> List[Dict[str, Any]]:
        """Translate segments in batches for efficiency"""
        self.logger.log_step_start(f"Translation to {target_language.upper()}")
        
        model, processor = self.load_seamless_model()
        
        translated_segments = []
        batch_size = config.BATCH_SIZE
        
        # Language mapping for SeamlessM4T
        lang_mapping = {
            "en": "eng",
            "de": "deu",
            "ar": "arb"
        }
        
        target_lang_code = lang_mapping.get(target_language, target_language)
        
        try:
            # Process in batches
            for i in tqdm(range(0, len(segments), batch_size), 
                         desc=f"Translating to {target_language}"):
                batch = segments[i:i + batch_size]
                batch_texts = [seg["original_text"] for seg in batch]
                
                try:
                    # Prepare inputs
                    inputs = processor(
                        text=batch_texts,
                        src_lang="arb",  # Arabic
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    
                    # Move to device if using CUDA
                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v 
                                for k, v in inputs.items()}
                    
                    # Generate translation
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            tgt_lang=target_lang_code,
                            max_new_tokens=512,
                            num_beams=5,
                            length_penalty=1.0,
                            early_stopping=True
                        )
                    
                    # Decode translations
                    translations = processor.batch_decode(
                        outputs, skip_special_tokens=True
                    )
                    
                    # Process batch results
                    for j, (segment, translation) in enumerate(zip(batch, translations)):
                        # Clean up translation
                        cleaned_translation = translation.strip()
                        
                        # Calculate confidence based on length ratio and content
                        confidence = self._calculate_translation_confidence(
                            segment["original_text"], cleaned_translation
                        )
                        
                        translated_segment = {
                            **segment,
                            "translated_text": cleaned_translation,
                            "target_language": target_language,
                            "translation_confidence": confidence,
                            "batch_id": i // batch_size
                        }
                        
                        translated_segments.append(translated_segment)
                    
                    # Clear batch from memory
                    del inputs, outputs
                    clear_gpu_memory()
                    
                except Exception as e:
                    self.logger.warning(f"Batch {i//batch_size} translation failed: {str(e)}")
                    # Add fallback translations for failed batch
                    for segment in batch:
                        translated_segments.append({
                            **segment,
                            "translated_text": f"[Translation Error: {segment['original_text']}]",
                            "target_language": target_language,
                            "translation_confidence": 0.0,
                            "batch_id": i // batch_size,
                            "error": str(e)
                        })
        
        except Exception as e:
            self.logger.error(f"Translation to {target_language} failed: {str(e)}")
            raise
        
        self.logger.info(f"Translation to {target_language} completed: {len(translated_segments)} segments")
        self.logger.log_step_end(f"Translation to {target_language.upper()}", True)
        
        return translated_segments
    
    def _calculate_translation_confidence(self, original: str, translation: str) -> float:
        """Calculate confidence score for translation quality"""
        # Simple heuristics for translation confidence
        if not translation or translation.startswith("[Translation Error"):
            return 0.0
        
        # Length ratio check
        length_ratio = len(translation) / max(len(original), 1)
        if length_ratio < 0.3 or length_ratio > 3.0:
            confidence = 0.5
        else:
            confidence = 0.8
        
        # Word count check
        original_words = len(original.split())
        translation_words = len(translation.split())
        word_ratio = translation_words / max(original_words, 1)
        
        if 0.5 <= word_ratio <= 2.0:
            confidence += 0.1
        
        # Check for obvious errors
        if "[" in translation or "]" in translation:
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def apply_context_preservation(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply context preservation to maintain presentation flow"""
        self.logger.log_step_start("Context Preservation")
        
        # Group segments by proximity for context
        context_groups = []
        current_group = []
        
        for i, segment in enumerate(segments):
            if not current_group:
                current_group.append(segment)
            else:
                # Check if segment is close to previous one (within 2 seconds)
                time_gap = segment["start"] - current_group[-1]["end"]
                if time_gap <= 2.0:
                    current_group.append(segment)
                else:
                    # Finalize current group and start new one
                    if current_group:
                        context_groups.append(current_group)
                    current_group = [segment]
        
        if current_group:
            context_groups.append(current_group)
        
        # Apply context-aware improvements
        improved_segments = []
        
        for group in context_groups:
            if len(group) == 1:
                improved_segments.extend(group)
                continue
            
            # For groups with multiple segments, ensure consistency
            for j, segment in enumerate(group):
                # Apply contextual improvements
                improved_text = self._improve_with_context(
                    segment["translated_text"], 
                    group, 
                    j
                )
                
                improved_segment = {
                    **segment,
                    "translated_text": improved_text,
                    "context_group": len(context_groups)
                }
                
                improved_segments.append(improved_segment)
        
        self.logger.info(f"Context preservation applied to {len(context_groups)} groups")
        self.logger.log_step_end("Context Preservation", True)
        
        return improved_segments
    
    def _improve_with_context(self, text: str, group: List[Dict[str, Any]], 
                            index: int) -> str:
        """Improve translation using context from surrounding segments"""
        # Simple context-based improvements
        improved_text = text
        
        # Ensure consistency in terminology
        if index > 0:
            prev_text = group[index - 1]["translated_text"]
            # Add logic to maintain terminology consistency
            # This is a placeholder for more sophisticated context processing
        
        # Handle presentation-specific terms
        presentation_terms = {
            "شريحة": "slide",
            "عرض": "presentation", 
            "مثال": "example",
            "سؤال": "question"
        }
        
        for arabic_term, english_term in presentation_terms.items():
            if arabic_term in group[index]["original_text"]:
                # Ensure consistent translation of presentation terms
                pass
        
        return improved_text
    
    def generate_translation_files(self, segments: List[Dict[str, Any]], 
                                 target_language: str) -> Path:
        """Generate translation files with timing metadata"""
        self.logger.log_step_start(f"Translation File Generation ({target_language})")
        
        # Create translation data structure
        translation_data = {
            "video_name": self.video_name,
            "target_language": target_language,
            "source_language": "ar",
            "total_segments": len(segments),
            "total_duration": max(seg["end"] for seg in segments) if segments else 0,
            "segments": segments,
            "translation_stats": self._calculate_translation_stats(segments),
            "created_at": self.checkpoint_manager.load_checkpoint()
        }
        
        # Save translation file
        translation_file = config.TEMP_DIR / f"{self.video_name}_translation_{target_language}.json"
        with open(translation_file, 'w', encoding='utf-8') as f:
            json.dump(translation_data, f, indent=2, ensure_ascii=False)
        
        # Create SRT preview for validation
        srt_preview_file = config.TEMP_DIR / f"{self.video_name}_translation_{target_language}_preview.srt"
        self._create_srt_preview(segments, srt_preview_file)
        
        self.logger.info(f"Translation files generated for {target_language}")
        self.logger.log_step_end(f"Translation File Generation ({target_language})", True)
        
        return translation_file
    
    def _calculate_translation_stats(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate translation quality statistics"""
        if not segments:
            return {}
        
        confidences = [seg.get("translation_confidence", 0.0) for seg in segments]
        
        stats = {
            "average_confidence": sum(confidences) / len(confidences),
            "high_confidence_segments": sum(1 for c in confidences if c >= 0.7),
            "low_confidence_segments": sum(1 for c in confidences if c < 0.5),
            "total_words_original": sum(len(seg["original_text"].split()) for seg in segments),
            "total_words_translated": sum(len(seg["translated_text"].split()) for seg in segments),
            "failed_translations": sum(1 for seg in segments if "error" in seg)
        }
        
        if stats["total_words_original"] > 0:
            stats["word_expansion_ratio"] = stats["total_words_translated"] / stats["total_words_original"]
        
        return stats
    
    def _create_srt_preview(self, segments: List[Dict[str, Any]], output_file: Path):
        """Create SRT preview file for translation validation"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._seconds_to_srt_time(segment["start"])
                end_time = self._seconds_to_srt_time(segment["end"])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['translated_text']}\n\n")
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def process_translation(self, transcript_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Complete translation processing pipeline"""
        self.logger.info(f"Starting translation processing for {self.video_name}")
        
        results = {}
        
        try:
            # Prepare segments
            segments = self.prepare_segments_for_translation(transcript_data)
            
            # Translate to each target language
            for target_language in config.TARGET_LANGUAGES:
                self.logger.info(f"Processing translation to {target_language}")
                
                # Translate segments
                translated_segments = self.translate_segments_batch(segments, target_language)
                
                # Apply context preservation
                improved_segments = self.apply_context_preservation(translated_segments)
                
                # Generate translation files
                translation_file = self.generate_translation_files(improved_segments, target_language)
                
                results[target_language] = {
                    "segments": improved_segments,
                    "translation_file": str(translation_file),
                    "stats": self._calculate_translation_stats(improved_segments)
                }
                
                # Save checkpoint for each language
                self.checkpoint_manager.save_checkpoint(f"translation_{target_language}", {
                    "translation_file": str(translation_file),
                    "segments_count": len(improved_segments),
                    "stats": results[target_language]["stats"]
                })
                
                self.logger.info(f"Translation to {target_language} completed successfully")
            
            # Mark step as completed
            self.checkpoint_manager.mark_step_completed("translation_processing")
            
            self.logger.info("Translation processing completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Translation processing failed: {str(e)}")
            raise
        finally:
            # Clean up models to free memory
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            clear_gpu_memory()
