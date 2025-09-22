from flask import Flask, request, jsonify
from transformers import pipeline
import tempfile
import os
from werkzeug.utils import secure_filename
import torch
import logging
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    MODEL_NAME = os.environ.get('MODEL_NAME', 'NextInnoMind/next_bemba_ai')

app.config.from_object(Config)

# Global pipeline variable
pipe = None

def initialize_model():
    """Initialize the ASR pipeline"""
    global pipe
    try:
        logger.info(f"Loading Whisper Bemba model: {Config.MODEL_NAME}")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=Config.MODEL_NAME,
            chunk_length_s=30,
            device=0 if torch.cuda.is_available() else -1,
            return_timestamps=True
        )
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    logger.info('Gracefully shutting down...')
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_first_request
def startup():
    """Initialize model on startup"""
    if not initialize_model():
        logger.error("Failed to initialize model. Exiting...")
        sys.exit(1)

@app.route('/')
def home():
    return jsonify({
        "message": "Whisper Bemba ASR API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "transcribe": "/transcribe (POST)",
            "transcribe_url": "/transcribe_url (POST)",
            "health": "/health (GET)"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    })

@app.route('/health', methods=['GET'])
def health_check():
    model_status = "ready" if pipe is not None else "not_loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "gpu_available": torch.cuda.is_available(),
        "model": Config.MODEL_NAME
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if pipe is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check file format
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file format. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + secure_filename(file.filename).rsplit('.', 1)[1].lower()) as temp_file:
            file.save(temp_file.name)
            
            # Transcribe the audio
            logger.info(f"Transcribing {file.filename}...")
            result = pipe(temp_file.name)
            
            # Clean up temporary file
            os.unlink(temp_file.name)
            
            # Prepare response
            response = {
                "text": result["text"],
                "language": "bemba",
                "timestamps": result.get("chunks", []),
                "success": True,
                "filename": secure_filename(file.filename)
            }
            
            logger.info(f"Successfully transcribed {file.filename}")
            return jsonify(response)
            
    except Exception as e:
        # Clean up if file was created
        if 'temp_file' in locals() and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        
        logger.error(f"Transcription failed for {file.filename}: {str(e)}")
        return jsonify({
            "error": f"Transcription failed: {str(e)}",
            "success": False
        }), 500

@app.route('/transcribe_url', methods=['POST'])
def transcribe_from_url():
    """Alternative endpoint for remote audio files"""
    if pipe is None:
        return jsonify({"error": "Model not loaded"}), 503
        
    data = request.get_json()
    
    if not data or 'audio_url' not in data:
        return jsonify({"error": "No audio_url provided"}), 400
    
    audio_url = data['audio_url']
    
    try:
        logger.info(f"Transcribing from URL: {audio_url}")
        result = pipe(audio_url)
        
        response = {
            "text": result["text"],
            "language": "bemba",
            "timestamps": result.get("chunks", []),
            "success": True,
            "source_url": audio_url
        }
        
        logger.info(f"Successfully transcribed from URL: {audio_url}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"URL transcription failed for {audio_url}: {str(e)}")
        return jsonify({
            "error": f"Transcription failed: {str(e)}",
            "success": False
        }), 500

if __name__ == '__main__':
    if not initialize_model():
        logger.error("Failed to initialize model. Exiting...")
        sys.exit(1)
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )