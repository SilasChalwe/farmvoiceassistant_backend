import logging
import torch
from transformers import VitsModel, AutoTokenizer
import os
import tempfile
from functools import lru_cache
import threading
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# CPU optimization settings for TTS
torch.set_num_threads(min(4, os.cpu_count()))

# Select device
device = 0 if torch.cuda.is_available() else -1
logger.info(f"TTS Device: {'GPU:0' if device == 0 else 'CPU'}")

# Bemba TTS Model - Facebook MMS TTS for Bemba
TTS_MODEL_ID = "facebook/mms-tts-bem"
logger.info(f"Using Bemba TTS model: {TTS_MODEL_ID}")

# Thread lock for thread-safe TTS
tts_lock = threading.Lock()

@lru_cache(maxsize=1)
def get_tts_model_and_tokenizer():
    """Initialize TTS model and tokenizer with caching for Bemba."""
    try:
        logger.info(f"Loading TTS model and tokenizer for {TTS_MODEL_ID}")
        
        # Load model and tokenizer directly (not via pipeline)
        model = VitsModel.from_pretrained(TTS_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
        
        # Move to appropriate device
        if device != -1:  # GPU
            model = model.cuda()
        else:  # CPU optimizations
            model = model.cpu()
            model.eval()
        
        logger.info("Bemba TTS model and tokenizer ready")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load Bemba TTS model: {e}")
        return None, None

def clean_gemini_response_for_tts(text):
    """
    Clean Gemini response for TTS - remove markdown/formatting but keep all content
    Handle comma truncation issues properly
    """
    if not text or not text.strip():
        return ""
    
    # LOG COMPLETE GEMINI RESPONSE TO CONSOLE
    logger.info("=" * 80)
    logger.info("🤖 COMPLETE GEMINI RESPONSE:")
    logger.info("=" * 80)
    logger.info(text)
    logger.info("=" * 80)
    logger.info(f"📊 Response Stats: {len(text)} characters, {len(text.split())} words")
    logger.info("=" * 80)
    
    logger.info(f"🧹 Starting TTS cleaning for: {len(text)} characters")
    
    # Remove markdown formatting but keep the content
    cleaned_text = text
    
    # Remove markdown headers (##, ###, etc.) but keep the text
    cleaned_text = re.sub(r'^#{1,6}\s*', '', cleaned_text, flags=re.MULTILINE)
    
    # Remove markdown bold/italic formatting but keep text
    cleaned_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_text)  # **bold** -> bold
    cleaned_text = re.sub(r'\*([^*]+)\*', r'\1', cleaned_text)      # *italic* -> italic
    
    # Remove bullet points and list formatting but keep content
    cleaned_text = re.sub(r'^\s*[\*\-\+]\s*', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^\s*\d+\.\s*', '', cleaned_text, flags=re.MULTILINE)
    
    # Clean up parenthetical English translations/explanations - be more selective
    # Only remove obvious English in parentheses, not Bemba explanations
    cleaned_text = re.sub(r'\s*\([A-Za-z\s/,.-]+\)\s*', ' ', cleaned_text)
    
    # Remove extra whitespace and normalize
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '. ', cleaned_text)  # Convert double newlines to periods
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)        # Convert single newlines to spaces
    
    # Clean up punctuation spacing
    cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)
    cleaned_text = re.sub(r'([.,!?])\s*([.,!?])', r'\1 \2', cleaned_text)
    
    # Remove any remaining excessive punctuation
    cleaned_text = re.sub(r'[.]{3,}', '.', cleaned_text)
    cleaned_text = re.sub(r'[!]{2,}', '!', cleaned_text)
    cleaned_text = re.sub(r'[?]{2,}', '?', cleaned_text)
    
    # Final cleanup
    cleaned_text = cleaned_text.strip()
    
    # Ensure proper sentence ending
    if cleaned_text and not cleaned_text[-1] in '.!?':
        cleaned_text += '.'
    
    # LOG CLEANED TEXT
    logger.info("🧹 CLEANED TEXT FOR TTS:")
    logger.info("-" * 60)
    logger.info(cleaned_text)
    logger.info("-" * 60)
    logger.info(f"📊 Cleaned Stats: {len(cleaned_text)} characters, {len(cleaned_text.split())} words")
    
    return cleaned_text

def check_audio_quality(waveform, sample_rate):
    """
    Check if the generated audio has quality issues that might sound like noise.
    """
    if len(waveform) == 0:
        return False, "Empty waveform"
    
    # Check for clipping (values at maximum range)
    clipped_samples = np.sum(np.abs(waveform) >= 0.99)
    clipping_percentage = clipped_samples / len(waveform) * 100
    
    if clipping_percentage > 1:  # More than 1% clipped
        return False, f"Audio clipping detected: {clipping_percentage:.1f}%"
    
    # Check for excessive silence (might indicate generation issues)
    silence_threshold = 0.01
    silent_samples = np.sum(np.abs(waveform) < silence_threshold)
    silence_percentage = silent_samples / len(waveform) * 100
    
    if silence_percentage > 80:  # More than 80% silence
        return False, f"Too much silence: {silence_percentage:.1f}%"
    
    # Check for DC offset (constant bias that can cause pops)
    dc_offset = np.mean(waveform)
    if abs(dc_offset) > 0.1:
        return False, f"DC offset detected: {dc_offset:.3f}"
    
    # Check signal-to-noise ratio
    signal_power = np.mean(waveform ** 2)
    if signal_power < 1e-6:  # Very weak signal
        return False, "Signal too weak"
    
    return True, "Audio quality OK"

def split_text_for_tts(text, max_length=600):
    """
    Split long text into smaller chunks for better TTS quality
    Handle comma truncation by treating commas as natural pause points
    """
    if len(text) <= max_length:
        logger.info(f"📝 Text fits in single chunk: {len(text)} chars")
        return [text]
    
    logger.info(f"✂️ Splitting long text ({len(text)} chars) into chunks...")
    
    # First, try splitting by sentences (periods, exclamations, questions)
    sentences = re.split(r'([.!?]+)', text)
    
    # If sentences are still too long, split by comma phrases
    final_sentences = []
    for sentence in sentences:
        if len(sentence) <= max_length:
            final_sentences.append(sentence)
        else:
            # Split long sentences by commas
            comma_parts = re.split(r'([,;])', sentence)
            current_part = ""
            
            for part in comma_parts:
                if len(current_part + part) <= max_length:
                    current_part += part
                else:
                    if current_part:
                        final_sentences.append(current_part)
                    current_part = part
            
            if current_part:
                final_sentences.append(current_part)
    
    # Now build chunks from these parts
    chunks = []
    current_chunk = ""
    
    for part in final_sentences:
        part = part.strip()
        if not part:
            continue
            
        # Check if adding this part would exceed max_length
        test_chunk = current_chunk + " " + part if current_chunk else part
        
        if len(test_chunk) <= max_length:
            current_chunk = test_chunk
        else:
            # Current chunk is full, start new one
            if current_chunk:
                # Ensure chunk ends properly
                if not current_chunk.rstrip()[-1:] in '.!?':
                    current_chunk = current_chunk.rstrip() + '.'
                chunks.append(current_chunk)
            
            # Start new chunk with current part
            current_chunk = part
    
    # Add the last chunk
    if current_chunk:
        if not current_chunk.rstrip()[-1:] in '.!?':
            current_chunk = current_chunk.rstrip() + '.'
        chunks.append(current_chunk)
    
    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    # Log chunk information
    logger.info(f"✂️ Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"   Chunk {i}: {len(chunk)} chars - '{chunk[:60]}...'")
    
    return chunks

def text_to_speech_high_quality(text: str, output_path: str = None) -> str:
    """
    Generate high-quality TTS audio from complete Gemini response
    """
    logger.info("=" * 80)
    logger.info(f"🎙️ STARTING FULL-LENGTH BEMBA TTS")
    logger.info(f"📝 Input text length: {len(text)} characters")
    logger.info(f"📝 Input word count: {len(text.split())} words")
    logger.info("=" * 80)
    
    if not text or text.strip() == '':
        logger.warning("❌ Empty text provided for TTS")
        return None
    
    with tts_lock:
        try:
            # Get model and tokenizer
            model, tokenizer = get_tts_model_and_tokenizer()
            if model is None or tokenizer is None:
                logger.error("❌ Bemba TTS model not available")
                return None
            
            # Clean text for TTS (remove markdown but keep all content)
            clean_text = clean_gemini_response_for_tts(text)
            
            if not clean_text or len(clean_text.strip()) < 3:
                logger.warning("❌ Text too short after cleaning for TTS")
                return None
            
            # Split text into manageable chunks for better TTS quality
            # Use smaller chunks to handle comma truncation better
            text_chunks = split_text_for_tts(clean_text, max_length=400)
            
            if not text_chunks:
                logger.error("❌ No chunks created from text")
                return None
            
            # Generate audio for each chunk
            all_waveforms = []
            
            logger.info(f"🎵 Processing {len(text_chunks)} chunks:")
            
            for i, chunk in enumerate(text_chunks):
                logger.info(f"🎵 Chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars):")
                logger.info(f"   Text: '{chunk}'")
                
                try:
                    # Tokenize input text
                    inputs = tokenizer(
                        chunk, 
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=350  # Smaller chunks for better handling
                    )
                    
                    # Move inputs to appropriate device
                    if device != -1:  # GPU
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # Generate speech
                    with torch.no_grad():
                        with torch.inference_mode():
                            output = model(**inputs)
                    
                    # Extract waveform from model output
                    waveform = None
                    if hasattr(output, 'waveform'):
                        waveform = output.waveform
                    elif hasattr(output, 'audio'):
                        waveform = output.audio
                    elif hasattr(output, 'prediction'):
                        waveform = output.prediction
                    elif isinstance(output, tuple) and len(output) > 0:
                        waveform = output[0]
                    else:
                        logger.error(f"❌ Could not extract waveform from chunk {i+1}")
                        continue
                    
                    # Move to CPU and convert to numpy
                    if torch.is_tensor(waveform):
                        waveform = waveform.cpu().numpy()
                    
                    # Ensure correct shape
                    if waveform.ndim > 1:
                        waveform = waveform.squeeze()
                    
                    if len(waveform) > 0:
                        all_waveforms.append(waveform)
                        duration = len(waveform) / 16000  # Assume 16kHz sample rate
                        logger.info(f"   ✅ Success: {len(waveform)} samples ({duration:.2f}s)")
                    else:
                        logger.warning(f"   ⚠️ Empty waveform for chunk {i+1}")
                    
                except Exception as chunk_error:
                    logger.error(f"   ❌ Failed to process chunk {i+1}: {chunk_error}")
                    continue
            
            if not all_waveforms:
                logger.error("❌ No audio chunks were successfully generated")
                return None
            
            # Concatenate all waveforms with small pauses between chunks
            sample_rate = getattr(model.config, 'sampling_rate', 16000)
            pause_samples = int(sample_rate * 0.4)  # 400ms pause between chunks
            pause_audio = np.zeros(pause_samples)
            
            logger.info(f"🔗 Combining {len(all_waveforms)} chunks with {pause_samples} sample pauses...")
            
            # Combine all chunks with pauses
            final_waveform = all_waveforms[0]
            for j, waveform in enumerate(all_waveforms[1:], 2):
                final_waveform = np.concatenate([final_waveform, pause_audio, waveform])
                logger.info(f"   Combined chunk {j}")
            
            total_duration = len(final_waveform) / sample_rate
            logger.info(f"🔗 Final audio: {len(final_waveform)} samples ({total_duration:.2f}s)")
            
            # Audio quality checks
            quality_ok, quality_msg = check_audio_quality(final_waveform, sample_rate)
            logger.info(f"🔍 Audio quality: {quality_msg}")
            
            # High-quality audio post-processing
            # Remove DC offset
            final_waveform = final_waveform - np.mean(final_waveform)
            
            # Gentle normalization
            max_val = np.max(np.abs(final_waveform))
            if max_val > 0:
                final_waveform = final_waveform / max_val * 0.6  # Conservative normalization
            
            # High-pass filter to remove rumble
            if len(final_waveform) > 200:
                try:
                    nyquist = sample_rate / 2
                    low_cutoff = 80  # Remove frequencies below 80Hz
                    low = low_cutoff / nyquist
                    if low < 0.5:
                        b, a = signal.butter(2, low, btype='high')
                        final_waveform = signal.filtfilt(b, a, final_waveform)
                        logger.info(f"🎛️ Applied high-pass filter at {low_cutoff}Hz")
                except Exception as filter_error:
                    logger.warning(f"⚠️ Filter application failed: {filter_error}")
            
            # Noise gate
            noise_floor = np.max(np.abs(final_waveform)) * 0.02
            final_waveform = np.where(np.abs(final_waveform) < noise_floor, 0, final_waveform)
            
            # Smooth fade-in and fade-out
            fade_samples = min(800, len(final_waveform) // 10)  # 50ms fade at 16kHz
            if len(final_waveform) > fade_samples * 2:
                # Fade in
                fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))**2
                final_waveform[:fade_samples] *= fade_in
                
                # Fade out
                fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))**2
                final_waveform[-fade_samples:] *= fade_out
            
            # Final limiting
            final_waveform = np.clip(final_waveform, -0.95, 0.95)
            
            # Create output file path
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir='/tmp')
                output_path = temp_file.name
                temp_file.close()
            
            # Convert to 16-bit PCM with dithering
            dither = np.random.normal(0, 1/65536, len(final_waveform)) * 0.3
            waveform_with_dither = final_waveform + dither
            waveform_int16 = np.clip(waveform_with_dither * 32767, -32768, 32767).astype(np.int16)
            
            # Save high-quality WAV file
            wavfile.write(output_path, sample_rate, waveform_int16)
            
            # Verify the file
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                duration = len(final_waveform) / sample_rate
                
                logger.info("=" * 80)
                logger.info(f"🎉 FULL-LENGTH BEMBA TTS COMPLETE!")
                logger.info(f"📁 Audio saved to: {output_path}")
                logger.info(f"⏱️ Audio duration: {duration:.2f} seconds")
                logger.info(f"📊 Sample rate: {sample_rate} Hz")
                logger.info(f"💾 File size: {file_size:,} bytes")
                logger.info(f"🔧 Processed {len(text_chunks)} chunks with filters and effects")
                logger.info(f"📝 Original text: {len(text)} chars → Final audio: {duration:.2f}s")
                logger.info("=" * 80)
                
                return output_path
            else:
                logger.error("❌ Failed to save audio file")
                return None
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ FULL-LENGTH BEMBA TTS FAILED: {str(e)}")
            logger.exception("Exception details:")
            logger.error("=" * 80)
            return None

# Use the high-quality function as the main TTS function
def text_to_speech_bemba(text: str, output_path: str = None) -> str:
    """
    Convert complete Bemba text to speech using Facebook MMS TTS model with full content processing.
    """
    return text_to_speech_high_quality(text, output_path)

# Keep the original function as fallback
def text_to_speech(text: str, output_path: str = None) -> str:
    """
    Fallback TTS function (original implementation).
    """
    return text_to_speech_high_quality(text, output_path)