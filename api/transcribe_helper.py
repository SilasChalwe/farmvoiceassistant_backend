import logging
import re
import torch
from transformers import pipeline, AutoTokenizer
import threading
from functools import lru_cache

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Thread lock ----------------
transcription_lock = threading.Lock()

# ---------------- Device ----------------
device = 0 if torch.cuda.is_available() else -1
if device == -1:
    # CPU optimizations
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_num_interop_threads(2)
    torch.backends.mkldnn.enabled = True
    logger.info("CPU optimizations enabled")

logger.info(f"Selected device: {'GPU:0' if device == 0 else 'CPU'}")

# ---------------- Model ----------------
MODEL_ID = "NextInnoMind/next_bemba_ai_medium"
logger.info(f"Loading tokenizer and ASR pipeline: {MODEL_ID}")

@lru_cache(maxsize=1)
def get_tokenizer():
    """Load the tokenizer and apply caching."""
    return AutoTokenizer.from_pretrained(MODEL_ID)

tokenizer = get_tokenizer()
task_token_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
BEM_TOKEN_ID = 51865
FORCED_DECODER_IDS = [[task_token_id, BEM_TOKEN_ID]]

pipeline_kwargs = {
    "task": "automatic-speech-recognition",
    "model": MODEL_ID,
    "device": device,
}

if device == -1:
    pipeline_kwargs.update({
        "dtype": torch.float32,
        "model_kwargs": {"use_cache": True, "low_cpu_mem_usage": True},
    })
else:
    pipeline_kwargs.update({"dtype": torch.float16})

transcriber = pipeline(**pipeline_kwargs)

if device == -1:
    try:
        transcriber.model.eval()
        if hasattr(torch.backends, "mkldnn") and torch.backends.mkldnn.is_available():
            transcriber.model = transcriber.model.to(memory_format=torch.channels_last)
        logger.info("CPU optimizations applied to model")
    except Exception as e:
        logger.warning(f"Failed to apply some CPU optimizations: {e}")

# ---------------- Transcription function ----------------
def transcribe_audio_file(filename: str) -> str:
    """Transcribe an audio file with CPU optimizations and cleaned output."""
    logger.info(f"Starting transcription for: {filename}")
    transcription_text = "No clear speech detected" # Default value

    try:
        with transcription_lock:
            # Added parameters to prevent repetitive output
            generation_kwargs = {
                "forced_decoder_ids": FORCED_DECODER_IDS,
                "max_new_tokens": 256,
                "num_beams": 4,  # Increased num_beams
                "do_sample": False,
                "use_cache": True,
                "no_repeat_ngram_size": 3, # Prevents repeating n-grams
                "repetition_penalty": 1.2, # Penalizes repeated tokens
            }

            with torch.no_grad():
                result = transcriber(filename, generate_kwargs=generation_kwargs)

            raw_text = (result.get("text") or "").strip()

            if raw_text:
                transcription_text = raw_text
            
            logger.info(f"Transcription result: {transcription_text}")

    except Exception as e:
        logger.exception("Transcription failed with an error")
        transcription_text = f"Transcription Failed: {e}"

    return transcription_text

