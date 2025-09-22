import os
import sys
import tempfile
import time
import logging
import threading
from functools import wraps
from collections import defaultdict
import hashlib
import uuid
import secrets
from typing import Dict, List, Optional
import json

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import pipeline, Pipeline
from werkzeug.utils import secure_filename

# --- Configure logging ---
# Centralized logging to a file and stdout for Gunicorn
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    """Application configuration from environment variables."""
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_FILE_SIZE_MB', 100)) * 1024 * 1024
    MODEL_NAME = os.environ.get('MODEL_NAME', 'NextInnoMind/next_bemba_ai')
    RATE_LIMIT_REQUESTS = int(os.environ.get('RATE_LIMIT_REQUESTS', 10))  # per minute
    RATE_LIMIT_WINDOW = int(os.environ.get('RATE_LIMIT_WINDOW', 60))  # seconds
    
    # Storage for API key data
    API_KEY_FILE = '.apikey'
    API_KEYS: Dict[str, Dict[str, str]] = {}
    
    @staticmethod
    def load_api_keys():
        """
        Loads API keys from a JSON file.
        Resets the file if it's corrupted to prevent startup failure.
        """
        if os.path.exists(Config.API_KEY_FILE):
            try:
                with open(Config.API_KEY_FILE, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        logger.warning(f"API key file {Config.API_KEY_FILE} is empty. Initializing with empty dictionary.")
                        Config.API_KEYS = {}
                        return
                    Config.API_KEYS = json.loads(content)
                logger.info(f"Loaded {len(Config.API_KEYS)} API keys from {Config.API_KEY_FILE}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to load API keys due to JSON corruption: {e}. Resetting file.")
                try:
                    with open(Config.API_KEY_FILE, 'w') as f:
                        json.dump({}, f)
                    Config.API_KEYS = {}
                    logger.info(f"Successfully reset corrupted file {Config.API_KEY_FILE}.")
                except Exception as file_e:
                    logger.critical(f"Failed to reset corrupted API key file: {file_e}")
            except Exception as e:
                logger.error(f"Failed to load API keys from file: {e}")

    @staticmethod
    def save_api_keys():
        """
        Saves API keys to a JSON file.
        Uses a temporary file to ensure atomicity and prevent corruption.
        """
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(mode='w', dir='.', delete=False, suffix='.tmp')
            json.dump(Config.API_KEYS, temp_file, indent=4)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()

            os.replace(temp_file.name, Config.API_KEY_FILE)
            logger.info(f"Saved {len(Config.API_KEYS)} API keys to {Config.API_KEY_FILE} safely.")
        except Exception as e:
            logger.error(f"Failed to save API keys to file: {e}", exc_info=True)
            if temp_file and os.path.exists(temp_file.name):
                os.remove(temp_file.name)
        finally:
            if temp_file and not temp_file.closed:
                temp_file.close()

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Load keys on startup
Config.load_api_keys()

# Global model pipeline
pipe: Optional[Pipeline] = None

# Thread-safe rate limiting storage
rate_limit_storage: defaultdict[str, List[float]] = defaultdict(list)
rate_limit_lock = threading.Lock()
api_key_lock = threading.Lock()

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'}

# --- Initialization and Helpers ---
def initialize_model() -> bool:
    """Initialize the ASR pipeline once per Gunicorn worker process."""
    global pipe
    try:
        if pipe is not None:
            logger.info("Model already initialized in this worker.")
            return True
            
        logger.info(f"Loading Whisper Bemba model: {Config.MODEL_NAME}")
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=Config.MODEL_NAME,
            chunk_length_s=30,
            device=device,
            return_timestamps=True
        )
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

# This is a Gunicorn-compatible way to load the model for each worker.
# The Gunicorn 'when_ready' hook can be used for this, but calling it directly is also common.
initialize_model()

def get_client_ip() -> str:
    """Get client IP address, respecting proxy headers."""
    return request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip()

def validate_file_size(file):
    """Validate file size against the configured max length."""
    if hasattr(file, 'content_length') and file.content_length:
        if file.content_length > app.config['MAX_CONTENT_LENGTH']:
            return False
    return True

def allowed_file(filename: str) -> bool:
    """Check if the file extension is supported."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Cryptography and Authentication ---
def authenticate_api_key(f):
    """Decorator to require and validate an API key."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            logger.warning(f"Authentication failed: No API key provided from {get_client_ip()}")
            return jsonify({"error": "Unauthorized: API key is missing"}), 401
        
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        
        with api_key_lock:
            user_info = Config.API_KEYS.get(hashed_key)
        
        if not user_info:
            truncated_api_key = api_key[:4] + '...' + api_key[-4:]
            logger.warning(f"Authentication failed: Invalid API key {truncated_api_key} from {get_client_ip()}")
            return jsonify({"error": "Unauthorized: Invalid API key"}), 401
        
        request.user_info = user_info
        
        return f(*args, **kwargs)
    return decorated_function

# --- Rate Limiting ---
def rate_limit(max_requests: Optional[int] = None, window: Optional[int] = None):
    """Rate limiting decorator using a thread-safe dictionary."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_id = getattr(request, 'user_info', {}).get('hashed_key', get_client_ip())
            current_time = time.time()
            
            max_reqs = max_requests or request.user_info.get('rate_limit', Config.RATE_LIMIT_REQUESTS)
            window_seconds = window or Config.RATE_LIMIT_WINDOW
            
            with rate_limit_lock:
                rate_limit_storage[client_id] = [
                    req_time for req_time in rate_limit_storage[client_id]
                    if current_time - req_time < window_seconds
                ]
                
                if len(rate_limit_storage[client_id]) >= max_reqs:
                    logger.warning(f"Rate limit exceeded for ID {client_id}")
                    return jsonify({
                        "error": "Rate limit exceeded. Please try again later.",
                        "retry_after": window_seconds
                    }), 429
                
                rate_limit_storage[client_id].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# --- API Key Generation Route ---
@app.route('/api/generate-key', methods=['POST'])
def generate_api_key():
    """Generates a unique API key and stores its hash."""
    with api_key_lock:
        client_ip = get_client_ip()
        
        for hashed_key, user_info in Config.API_KEYS.items():
            if user_info.get('ip') == client_ip:
                logger.warning(f"Key already exists for IP {client_ip}.")
                return jsonify({"error": "An API key for this IP address already exists."}), 409

        api_key = str(uuid.uuid4()) + secrets.token_hex(16)
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        
        Config.API_KEYS[hashed_key] = {
            'user_id': str(uuid.uuid4()),
            'hashed_key': hashed_key,
            'rate_limit': Config.RATE_LIMIT_REQUESTS,
            'ip': client_ip
        }
        
        Config.save_api_keys()
        
        logger.info(f"Generated new API key for IP {client_ip}")
        
        return jsonify({
            "success": True,
            "message": "API key generated successfully. Please store it securely.",
            "api_key": api_key,
            "note": "This is the only time the API key will be shown."
        }), 201

# --- Transcription Routes ---
@app.route('/')
def home():
    """Render the main index.html page."""
    model_status = "ready" if pipe is not None else "not_loaded"
    return render_template(
        'index.html',
        model_status=model_status,
        gpu_available=torch.cuda.is_available(),
        model_name=Config.MODEL_NAME,
        max_file_size=app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        supported_formats=list(ALLOWED_EXTENSIONS),
        rate_limit_requests=Config.RATE_LIMIT_REQUESTS,
        rate_limit_window=Config.RATE_LIMIT_WINDOW
    )

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify service status."""
    model_status = "ready" if pipe is not None else "not_loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "gpu_available": torch.cuda.is_available(),
        "model": Config.MODEL_NAME,
        "server_type": "Gunicorn Worker"
    })

@app.route('/transcribe', methods=['POST'])
@authenticate_api_key
@rate_limit()
def transcribe_audio():
    """API endpoint for file-based transcription."""
    if pipe is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not validate_file_size(file):
        return jsonify({"error": f"File too large. Max size: {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)}MB"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Supported: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{secure_filename(file.filename).rsplit('.', 1)[1].lower()}') as temp_file:
            temp_file_path = temp_file.name
            file.save(temp_file_path)
            
            logger.info(f"Starting transcription for {secure_filename(file.filename)} by user {request.user_info.get('user_id', 'N/A')}")
            start_time = time.time()
            result = pipe(temp_file_path)
            processing_time = time.time() - start_time
            
            segments = result.get("segments", [])
            timestamps = []
            for segment in segments:
                timestamps.append({
                    "timestamp": segment.get("timestamp", [0, 0]),
                    "text": segment.get("text", "")
                })

            response = {
                "text": result.get("text", ""),
                "language": "bemba",
                "timestamps": timestamps,
                "success": True,
                "filename": secure_filename(file.filename),
                "processing_time_seconds": round(processing_time, 2)
            }
            return jsonify(response)
            
    except Exception as e:
        logger.error(f"Transcription failed for {file.filename}: {str(e)}")
        return jsonify({"error": f"Transcription failed: {str(e)}", "success": False}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file_path}: {str(e)}")

@app.route('/transcribe_url', methods=['POST'])
@authenticate_api_key
@rate_limit()
def transcribe_from_url():
    """API endpoint for URL-based transcription."""
    if pipe is None:
        return jsonify({"error": "Model not loaded"}), 503
        
    data = request.get_json()
    if not data or 'audio_url' not in data:
        return jsonify({"error": "No audio_url provided"}), 400
    
    audio_url = data['audio_url']
    
    try:
        logger.info(f"Starting URL transcription for {audio_url} by user {request.user_info.get('user_id', 'N/A')}")
        start_time = time.time()
        result = pipe(audio_url)
        processing_time = time.time() - start_time

        segments = result.get("segments", [])
        timestamps = []
        for segment in segments:
            timestamps.append({
                "timestamp": segment.get("timestamp", [0, 0]),
                "text": segment.get("text", "")
            })
        
        response = {
            "text": result.get("text", ""),
            "language": "bemba",
            "timestamps": timestamps,
            "success": True,
            "source_url": audio_url,
            "processing_time_seconds": round(processing_time, 2)
        }
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"URL transcription failed for {audio_url}: {str(e)}")
        return jsonify({"error": f"Transcription failed: {str(e)}", "success": False}), 500

# --- Error Handlers ---
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": f"File too large. Maximum size: {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)}MB"
    }), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error", "success": False}), 500

# Gunicorn doesn't use the standard `signal` module in this way for graceful shutdown.
# It manages workers and sends signals itself.
# The 'on_exit' hook is a better place to handle final tasks.
# Let's add a placeholder for a Gunicorn hook.

def on_exit(server):
    """Gunicorn hook for graceful shutdown."""
    logger.info("Gunicorn master process exiting. Saving API keys...")
    with api_key_lock:
        Config.save_api_keys()

# To use this hook, you'd run Gunicorn with a config file.
# We'll stick to the command line for simplicity in the bash script.
# The `Config.save_api_keys()` call at the end of the `signal_handler` is
# not triggered when Gunicorn manages the shutdown, but it's a good
# fallback for direct script execution. Let's remove the signal handler
# since Gunicorn handles this.
# Removed `signal` module and `signal_handler` function as Gunicorn handles this.