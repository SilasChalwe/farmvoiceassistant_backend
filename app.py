from flask import Flask, request, jsonify
from transformers import pipeline
import tempfile
import os
from werkzeug.utils import secure_filename
import torch

app = Flask(__name__)

# Initialize the pipeline
print("Loading Whisper Bemba model...")
pipe = pipeline(
    "automatic-speech-recognition",
    model="NextInnoMind/next_bemba_ai",
    chunk_length_s=30,
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    return_timestamps=True
)
print("Model loaded successfully!")

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return jsonify({
        "message": "Whisper Bemba ASR API",
        "status": "active",
        "endpoints": {
            "transcribe": "/transcribe (POST)",
            "health": "/health (GET)"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS)
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "model": "NextInnoMind/next_bemba_ai"
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
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
            print(f"Transcribing {file.filename}...")
            result = pipe(temp_file.name)
            
            # Clean up temporary file
            os.unlink(temp_file.name)
            
            # Prepare response
            response = {
                "text": result["text"],
                "language": "bemba",
                "timestamps": result.get("chunks", []),
                "success": True
            }
            
            return jsonify(response)
            
    except Exception as e:
        # Clean up if file was created
        if 'temp_file' in locals() and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        
        return jsonify({
            "error": f"Transcription failed: {str(e)}",
            "success": False
        }), 500

@app.route('/transcribe_url', methods=['POST'])
def transcribe_from_url():
    """Alternative endpoint for remote audio files"""
    data = request.get_json()
    
    if not data or 'audio_url' not in data:
        return jsonify({"error": "No audio_url provided"}), 400
    
    audio_url = data['audio_url']
    
    try:
        print(f"Transcribing from URL: {audio_url}")
        result = pipe(audio_url)
        
        response = {
            "text": result["text"],
            "language": "bemba",
            "timestamps": result.get("chunks", []),
            "success": True
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": f"Transcription failed: {str(e)}",
            "success": False
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)