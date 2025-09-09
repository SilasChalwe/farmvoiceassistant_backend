import logging
import torch
from transformers import pipeline, AutoTokenizer
import os
from functools import lru_cache
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# CPU optimization settings
torch.set_num_threads(min(4, os.cpu_count()))  # Limit threads to prevent overhead
torch.set_num_interop_threads(2)  # Reduce inter-op parallelism

# Select device with CPU optimizations
device = 0 if torch.cuda.is_available() else -1
if device == -1:
    # CPU-specific optimizations
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    torch.backends.mkldnn.enabled = True  # Enable Intel MKL-DNN for CPU
    
logger.info(f"Selected device: {'GPU:0' if device == 0 else 'CPU'}")
logger.info(f"CPU threads: {torch.get_num_threads()}")

# Model info - consider using a smaller/faster model for CPU
MODEL_ID = "NextInnoMind/next_bemba_ai_medium"
logger.info(f"Loading tokenizer and ASR pipeline for {MODEL_ID}")

# Load tokenizer with caching
@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)

tokenizer = get_tokenizer()

# Set forced decoder tokens for Bemba transcription
task_token_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
BEM_TOKEN_ID = 51865
FORCED_DECODER_IDS = [[task_token_id, BEM_TOKEN_ID]]

# Initialize ASR pipeline with CPU optimizations
pipeline_kwargs = {
    "task": "automatic-speech-recognition",
    "model": MODEL_ID,
    "device": device,
}

# Add CPU-specific optimizations
if device == -1:
    pipeline_kwargs.update({
        "dtype": torch.float32,  # Use dtype instead of torch_dtype (deprecated)
        "model_kwargs": {
            "use_cache": True,
            "low_cpu_mem_usage": True,
        }
    })
else:
    # For GPU, you can use different settings
    pipeline_kwargs.update({
        "dtype": torch.float16,  # Use dtype instead of torch_dtype (deprecated)
    })

transcriber = pipeline(**pipeline_kwargs)

# Enable CPU optimizations for the model (safer approach)
if device == -1:
    try:
        # Set model to eval mode for inference
        transcriber.model.eval()
        
        # Enable CPU optimizations if available
        if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
            # Only apply optimizations that are compatible
            transcriber.model = transcriber.model.to(memory_format=torch.channels_last)
        
        logger.info("Applied CPU optimizations to model")
    except Exception as opt_error:
        logger.warning(f"Could not apply some CPU optimizations: {opt_error}")

logger.info("ASR pipeline ready with CPU optimizations")

# Thread lock for thread-safe transcription
transcription_lock = threading.Lock()

def dummy_transcribe(filename: str) -> str:
    """Transcribe an audio file with CPU optimizations."""
    logger.info(f"Starting transcription for: {filename}")

    if "fail" in filename.lower():
        logger.warning("Fail condition triggered by filename")
        return "Transcription Failed"

    with transcription_lock:  # Ensure thread safety
        try:
            logger.info("Running ASR pipeline with Bemba token...")
            
            # CPU-optimized generation parameters
            generation_kwargs = {
                "forced_decoder_ids": FORCED_DECODER_IDS,
                "max_new_tokens": 256,  # Limit output length for speed
                "num_beams": 1,  # Use greedy decoding for speed
                "do_sample": False,  # Disable sampling for consistency
                "use_cache": True,  # Enable KV cache
            }
            
            # Use torch.no_grad() to reduce memory usage
            with torch.no_grad():
                result = transcriber(filename, generate_kwargs=generation_kwargs)
            
            transcription = (result.get("text") or "").strip()
            
            # Log full transcription to console
            logger.info(f"Transcription successful (length={len(transcription)} chars)")
            logger.info(f"Transcribed text:\n{transcription}\n")
            
            return transcription
            
        except Exception as e:
            logger.exception("Transcription failed with an error")
            return f"Transcription Failed: {e}"

# Optional: Warm up the model with a dummy run
def warmup_model():
    """Warm up the model to improve first-run performance."""
    try:
        import tempfile
        import numpy as np
        from scipy.io.wavfile import write
        
        # Create a short dummy audio file
        sample_rate = 16000
        duration = 1  # 1 second
        frequency = 440  # A note
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            write(tmp_file.name, sample_rate, audio_data)
            
            logger.info("Warming up model...")
            dummy_transcribe(tmp_file.name)
            logger.info("Model warmup complete")
            
            # Clean up
            os.unlink(tmp_file.name)
            
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")

# Uncomment to enable warmup on module load
# warmup_model()
