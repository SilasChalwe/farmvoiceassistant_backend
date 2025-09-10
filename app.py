from flask import Flask, request, jsonify, send_file
import os
import logging
from pydub import AudioSegment
from pydub.effects import normalize
import numpy as np
import tempfile
import re
from flask_cors import CORS
from api.transcribe_helper import dummy_transcribe
from api.file_helper import delete_file, get_unique_filename
from api.tts_helper import text_to_speech_bemba
from api.gemini_helper import get_agriculture_advice
import time
import base64
from dotenv import load_dotenv
import threading

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Enhanced File Management System
class TTSFileManager:
    def __init__(self, cleanup_interval=300, max_file_age=1800):  # 5 min cleanup, 30 min max age
        self.cleanup_interval = cleanup_interval
        self.max_file_age = max_file_age
        self.active_files = set()
        self.cleanup_thread = None
        self.running = False
    
    def start_cleanup_service(self):
        """Start the background cleanup service"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self.cleanup_thread.start()
            logger.info("TTS file cleanup service started")
    
    def register_file(self, filepath):
        """Register a file for tracking"""
        self.active_files.add(filepath)
        logger.debug(f"Registered TTS file for cleanup: {filepath}")
    
    def _cleanup_worker(self):
        """Background worker to clean up old files"""
        while self.running:
            try:
                current_time = time.time()
                files_to_remove = []
                
                for filepath in list(self.active_files):
                    if os.path.exists(filepath):
                        file_age = current_time - os.path.getmtime(filepath)
                        if file_age > self.max_file_age:
                            try:
                                os.unlink(filepath)
                                files_to_remove.append(filepath)
                                logger.info(f"Cleaned up old TTS file: {filepath}")
                            except Exception as e:
                                logger.warning(f"Failed to cleanup {filepath}: {e}")
                    else:
                        files_to_remove.append(filepath)
                
                # Remove tracked files
                for filepath in files_to_remove:
                    self.active_files.discard(filepath)
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
            
            time.sleep(self.cleanup_interval)
    
    def stop_cleanup_service(self):
        """Stop the cleanup service"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)

# Initialize the file manager
file_manager = TTSFileManager()

def is_audio_file_good(audio_path):
    """
    Check if audio file is sufficient for transcription without RIFF validation
    """
    try:
        # Try to load with pydub first (handles multiple formats)
        try:
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0  # Convert ms to seconds
            
            # Check duration
            if duration < 1.0:
                logger.warning(f"Audio too short: {duration:.2f} seconds")
                return False
            if duration > 30.0:
                logger.warning(f"Audio too long: {duration:.2f} seconds")
                return False
                
            # Check volume level
            if audio.dBFS < -50:  # Very quiet audio
                logger.warning(f"Audio too quiet: {audio.dBFS:.2f} dBFS")
                return False
                
            return True
            
        except Exception as pydub_error:
            logger.warning(f"Pydub loading failed, trying alternative methods: {pydub_error}")
            
            # Fallback: check file size and basic properties
            file_size = os.path.getsize(audio_path)
            if file_size < 1024:  # Less than 1KB
                logger.warning(f"File too small: {file_size} bytes")
                return False
                
            if file_size > 10 * 1024 * 1024:  # More than 10MB
                logger.warning(f"File too large: {file_size} bytes")
                return False
                
            # If we can't analyze it properly, still try to process it
            return True
            
    except Exception as e:
        logger.error(f"Audio quality check failed: {e}")
        # Still try to process the file despite errors
        return True

def enhanced_audio_preprocessing(audio_path):
    """
    Enhanced preprocessing with better noise detection and handling
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        
        # Check signal quality first
        if audio.dBFS < -45:
            # Very quiet audio - needs amplification
            gain_needed = max(-25 - audio.dBFS, 0)
            audio = audio + min(gain_needed, 20)  # Don't over-amplify
            logger.info(f"Applied {gain_needed}dB gain to quiet audio")
        
        # Convert to mono 16kHz
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        # Smart noise reduction
        # Remove very low frequencies (air handling, rumble)
        audio = audio.high_pass_filter(80)
        
        # Remove very high frequencies (electronic noise)
        audio = audio.low_pass_filter(4000)
        
        # Gentle normalization
        normalized_audio = audio.normalize()
        
        # Smart silence removal
        try:
            # More conservative silence removal
            processed_audio = normalized_audio.strip_silence(
                silence_len=500,    # 0.5 seconds of silence
                silence_thresh=normalized_audio.dBFS - 20,  # Relative to audio level
                padding=300         # Keep 0.3s padding
            )
            
            # Ensure we didn't remove too much
            if len(processed_audio) < len(normalized_audio) * 0.5:
                logger.warning("Silence removal too aggressive, using normalized audio")
                processed_audio = normalized_audio
                
        except Exception as silence_error:
            logger.warning(f"Silence removal failed: {silence_error}, using normalized audio")
            processed_audio = normalized_audio
        
        # Export with high quality settings
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        processed_audio.export(
            temp_file.name, 
            format='wav',
            parameters=["-ac", "1", "-ar", "16000", "-sample_fmt", "s16"]
        )
        
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Enhanced audio preprocessing failed: {e}")
        return audio_path

def enhanced_transcription_validation(text):
    """
    Enhanced validation with better repetition detection
    """
    if not text or len(text.strip()) < 3:
        return None, "No speech detected"
    
    original_text = text
    text = text.strip()
    
    # Count unique vs total characters for repetition detection
    unique_chars = len(set(text.lower().replace(' ', '')))
    total_chars = len(text.replace(' ', ''))
    
    if total_chars > 0:
        uniqueness_ratio = unique_chars / total_chars
        logger.info(f"Text uniqueness ratio: {uniqueness_ratio:.3f}")
        
        # If very repetitive, try to find where repetition starts
        if uniqueness_ratio < 0.3:  # Less than 30% unique characters
            logger.info("Detected highly repetitive text, attempting to extract meaningful content")
            
            # Find the first repetitive pattern start
            for i, char in enumerate(text):
                if i > 20:  # Keep at least first 20 chars
                    remaining = text[i:]
                    if len(remaining) > 10:
                        remaining_uniqueness = len(set(remaining.lower())) / len(remaining)
                        if remaining_uniqueness < 0.2:  # Very repetitive section
                            clean_text = text[:i].strip()
                            # Clean up any trailing punctuation or incomplete words
                            clean_text = re.sub(r'[,.\s]+$', '', clean_text)
                            if len(clean_text) > 10:
                                logger.info(f"Extracted meaningful content: '{clean_text}'")
                                return clean_text, "success"
            
            # If we couldn't find a good cutoff, take first 50 chars
            fallback_text = text[:50].strip()
            if len(fallback_text) > 10:
                return fallback_text, "partial_recovery"
    
    # Additional pattern-based cleaning for common Whisper artifacts
    whisper_artifacts = [
        r"'[a-zA-Z]'[a-zA-Z]'[a-zA-Z]'[a-zA-Z].*",  # 'a'a'a'a... pattern
        r"([a-zA-Z])\1{8,}",                         # Character repeated 9+ times
        r"(.{1,3})\1{5,}",                          # Any 1-3 char sequence repeated 6+ times
    ]
    
    for pattern in whisper_artifacts:
        match = re.search(pattern, text)
        if match:
            artifact_start = match.start()
            if artifact_start > 15:  # Keep meaningful content before artifact
                clean_text = text[:artifact_start].strip()
                clean_text = re.sub(r'[,.\s]+$', '', clean_text)
                if len(clean_text) > 10:
                    logger.info(f"Removed Whisper artifact, keeping: '{clean_text}'")
                    return clean_text, "success"
    
    # Text seems okay
    return text, "success"

def truncate_response_for_tts(text, min_words=500, max_words=1000, max_chars=80000):
    """
    Truncate long responses to a minimum of 500 words (or up to 1000), while preserving meaning and not exceeding max_chars.
    """
    if not text:
        return text

    # Split text into words
    words = text.split()
    total_words = len(words)

    # If text is already within the desired word and char limits, return as is
    if total_words <= max_words and len(text) <= max_chars:
        return text

    # Determine how many words to keep
    num_words_to_keep = min(max(total_words, min_words), max_words)
    truncated_words = words[:num_words_to_keep]

    # Reconstruct the truncated text
    truncated_text = ' '.join(truncated_words)

    # If still too long in chars, trim further
    if len(truncated_text) > max_chars:
        truncated_text = truncated_text[:max_chars]
        # Try to cut at the last space to avoid breaking words
        last_space = truncated_text.rfind(' ')
        if last_space > 0:
            truncated_text = truncated_text[:last_space]

    # Ensure the text ends with a period
    truncated_text = truncated_text.strip()
    if truncated_text and not truncated_text.endswith('.'):
        truncated_text += '.'

    # Final check to ensure we have at least min_words if possibl
    if len(truncated_text.split()) < total_words:
        truncated_text += ""

    logger.info(f"TTS text truncated to {len(truncated_text.split())} words and {len(truncated_text)} chars")
    return truncated_text

# LEGACY FUNCTIONS (keeping for compatibility)
def remove_repetitive_text(text):
    """Legacy function - now uses enhanced validation"""
    clean_text, status = enhanced_transcription_validation(text)
    return clean_text if clean_text else text

def validate_and_clean_transcription(text):
    """Legacy function - now uses enhanced validation"""
    if not text or text.strip() == '':
        return "No speech detected. Please try again with clearer audio."
    
    clean_text, status = enhanced_transcription_validation(text)
    
    if status == "success" or status == "partial_recovery":
        # Ensure we end with proper punctuation
        if clean_text and not clean_text[-1] in '.!?':
            clean_text = clean_text + '.'
        return clean_text
    else:
        return clean_text or "Background noise detected. Please speak closer to the microphone."

def preprocess_audio(audio_path):
    """Legacy function - now uses enhanced preprocessing"""
    return enhanced_audio_preprocessing(audio_path)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if TTS is requested (default: True)
    generate_tts = request.form.get('generate_tts', 'true').lower() == 'true'
    
    filename = get_unique_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        file.save(filepath)
        logger.info(f"File saved: {filepath} (size: {os.path.getsize(filepath)} bytes)")

        # Basic file validation
        if os.path.getsize(filepath) == 0:
            return jsonify({'error': 'Empty file uploaded'}), 400

        # Check audio quality
        if not is_audio_file_good(filepath):
            return jsonify({
                'error': 'Audio quality issues detected',
                'suggestions': [
                    'Move microphone closer to speaker',
                    'Minimize background noise', 
                    'Ensure audio is 1-30 seconds long',
                    'Speak clearly and at normal volume'
                ]
            }), 400

        # Enhanced audio preprocessing
        processed_audio_path = enhanced_audio_preprocessing(filepath)
        logger.info("Enhanced audio preprocessing completed")

        try:
            # Get transcription
            raw_text = dummy_transcribe(processed_audio_path)
            logger.info(f"Raw transcription: '{raw_text[:100]}...' ({len(raw_text)} chars)")

            # Enhanced transcription validation
            final_text, validation_status = enhanced_transcription_validation(raw_text)
            
            # Check validation results
            if validation_status not in ["success", "partial_recovery"] or not final_text:
                return jsonify({
                    'success': False,
                    'error': final_text or "Could not extract meaningful speech",
                    'debug_info': {
                        'raw_text_preview': raw_text[:200] + '...' if len(raw_text) > 200 else raw_text,
                        'raw_length': len(raw_text),
                        'validation_status': validation_status
                    }
                }), 400

            # Get agriculture advice from Gemini
            logger.info(f"Calling Gemini with text: '{final_text}'")
            gemini_result = get_agriculture_advice(final_text)
            
            if not gemini_result['success']:
                return jsonify({
                    'success': False,
                    'error': f"Gemini API Error: {gemini_result.get('error', 'Unknown error')}",
                    'text': gemini_result.get('bemba_response', 'Ichipepesho cha API chalekelele.')
                }), 500
            
            # Use the Bemba response from Gemini
            bemba_response_text = gemini_result['bemba_response']
            
            # Truncate response for better TTS experience
            tts_text = truncate_response_for_tts(bemba_response_text)
            
            # Prepare enhanced response data
            response_data = {
                'success': True,
                'text': bemba_response_text,  # Full response for display
                'tts_text': tts_text,         # Truncated for TTS
                'original_text': final_text,   # Original transcription
                'transcription_status': validation_status,
                'response_truncated': len(tts_text) < len(bemba_response_text),
                'transcription': {
                    'original_length': len(raw_text),
                    'cleaned_length': len(final_text),
                    'processing_applied': len(raw_text) != len(final_text),
                    'word_count': len(final_text.split()),
                    'confidence': 'high' if len(final_text) > 20 else 'medium'
                },
                'processing_time': time.time(),
                'timestamp': int(time.time()),
                'model_used': gemini_result.get('model_used'),
            }

            # Generate TTS audio from truncated Bemba response
            if generate_tts and len(tts_text.strip()) > 0:
                logger.info("Generating TTS audio from truncated Bemba response...")
                try:
                    tts_audio_path = text_to_speech_bemba(tts_text)
                    
                    if tts_audio_path and os.path.exists(tts_audio_path):
                        # Register file for automatic cleanup
                        file_manager.register_file(tts_audio_path)
                        
                        tts_file_size = os.path.getsize(tts_audio_path)
                        audio_filename = os.path.basename(tts_audio_path)
                        
                        logger.info(f"TTS audio generated successfully (size: {tts_file_size} bytes)")
                        
                        # Calculate actual duration from file
                        try:
                            audio_segment = AudioSegment.from_wav(tts_audio_path)
                            actual_duration = len(audio_segment) / 1000.0  # Convert ms to seconds
                            duration_str = f"{actual_duration:.1f}s"
                        except Exception as duration_error:
                            logger.warning(f"Could not calculate duration: {duration_error}")
                            duration_str = f"{tts_file_size / 32000:.1f}s"
                        
                        # Add TTS data to response
                        response_data['tts'] = {
                            'generated': True,
                            'audio_url': f"/stream-tts-audio/{audio_filename}",
                            'download_url': f"/download-tts-audio/{audio_filename}",
                            'filename': audio_filename,
                            'file_size': tts_file_size,
                            'duration': duration_str,
                            'sample_rate': 16000,
                            'format': 'wav',
                            'mimetype': 'audio/wav',
                            'text_length': len(tts_text),
                            'truncated': response_data['response_truncated']
                        }
                        
                    else:
                        logger.warning("TTS generation failed")
                        response_data['tts'] = {
                            'generated': False,
                            'error': 'TTS generation failed'
                        }
                        
                except Exception as tts_error:
                    logger.error(f"TTS generation error: {tts_error}")
                    response_data['tts'] = {
                        'generated': False,
                        'error': f'TTS generation failed: {str(tts_error)}'
                    }
            else:
                # TTS not requested or text is empty
                response_data['tts'] = {
                    'generated': False,
                    'reason': 'TTS not requested or empty text'
                }

            # Add debug info if significant processing was applied
            if validation_status == "partial_recovery" or len(raw_text) > len(final_text) * 1.5:
                response_data['debug_info'] = {
                    'raw_preview': raw_text[:100] + '...' if len(raw_text) > 100 else raw_text,
                    'cleaning_applied': True,
                    'validation_status': validation_status
                }

            logger.info(f"Successful enhanced transcription + TTS: '{final_text[:50]}...' ({len(final_text)} chars)")
            return jsonify(response_data)

        finally:
            # Clean up processed temp file
            if (processed_audio_path != filepath and 
                os.path.exists(processed_audio_path) and
                processed_audio_path.startswith('/tmp')):
                try:
                    os.unlink(processed_audio_path)
                    logger.debug(f"Cleaned up temp file: {processed_audio_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

    except Exception as e:
        logger.error(f"Enhanced transcription error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Transcription processing failed. Please try again with clearer audio.',
            'debug_info': {
                'error_type': type(e).__name__,
                'error_message': str(e) if app.debug else None
            }
        }), 500

    finally:
        # Delete original uploaded file
        if os.path.exists(filepath):
            try:
                delete_file(filepath)
                logger.debug(f"Cleaned up uploaded file: {filepath}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup uploaded file: {cleanup_error}")

@app.route('/download-tts-audio/<path:filename>')
def download_tts_audio(filename):
    """Download generated TTS audio file with improved security."""
    try:
        # Enhanced security check (same as stream endpoint)
        if not filename or '..' in filename or filename.startswith('/') or '\\' in filename:
            logger.warning(f"Unsafe file path requested: {filename}")
            return jsonify({'error': 'Invalid file path'}), 400
        
        if not filename.lower().endswith(('.wav', '.mp3', '.m4a')):
            logger.warning(f"Invalid audio file type requested: {filename}")
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Construct full path
        file_path = os.path.join('/tmp', filename)
        
        # Security check
        real_path = os.path.realpath(file_path)
        if not real_path.startswith('/tmp/'):
            logger.warning(f"File path outside allowed directory: {real_path}")
            return jsonify({'error': 'Access denied'}), 403
        
        if not os.path.exists(file_path):
            logger.warning(f"TTS audio file not found: {file_path}")
            return jsonify({'error': 'TTS audio file not found'}), 404
        
        logger.info(f"Serving TTS audio file for download: {file_path}")
        
        # Generate a user-friendly download name
        timestamp = int(time.time())
        download_name = f"bemba_tts_{timestamp}.wav"
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='audio/wav'
        )
        
    except Exception as e:
        logger.error(f"TTS audio download error: {e}")
        return jsonify({'error': 'Failed to download TTS audio file'}), 500

@app.route('/stream-tts-audio/<path:filename>')
def stream_tts_audio(filename):
    """Stream TTS audio file for web playback with improved security and headers."""
    try:
        # Enhanced security check
        if not filename or '..' in filename or filename.startswith('/') or '\\' in filename:
            logger.warning(f"Unsafe file path requested: {filename}")
            return jsonify({'error': 'Invalid file path'}), 400
        
        # Only allow specific file extensions
        if not filename.lower().endswith(('.wav', '.mp3', '.m4a')):
            logger.warning(f"Invalid audio file type requested: {filename}")
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Construct full path - assuming TTS files are in /tmp
        file_path = os.path.join('/tmp', filename)
        
        # Additional security: ensure the resolved path is still in /tmp
        real_path = os.path.realpath(file_path)
        if not real_path.startswith('/tmp/'):
            logger.warning(f"File path outside allowed directory: {real_path}")
            return jsonify({'error': 'Access denied'}), 403
        
        if not os.path.exists(file_path):
            logger.warning(f"TTS audio file not found: {file_path}")
            return jsonify({'error': 'TTS audio file not found'}), 404
        
        # Check file age - delete files older than 1 hour
        file_age = time.time() - os.path.getmtime(file_path)
        if file_age > 3600:  # 1 hour
            logger.info(f"Removing old TTS file: {file_path}")
            try:
                os.unlink(file_path)
                return jsonify({'error': 'TTS audio file expired'}), 404
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup old file: {cleanup_error}")
        
        logger.info(f"Streaming TTS audio file: {file_path}")
        
        return send_file(
            file_path,
            mimetype='audio/wav',
            as_attachment=False,  # Stream, don't download
            conditional=True,     # Support range requests for better mobile compatibility
            etag=True,           # Enable ETag for caching
            last_modified=True,  # Enable Last-Modified for caching
            max_age=1800        # Cache for 30 minutes
        )
        
    except Exception as e:
        logger.error(f"TTS audio streaming error: {e}")
        return jsonify({'error': 'Failed to stream TTS audio file'}), 500

@app.route('/cleanup-tts/<path:filename>', methods=['DELETE'])
def cleanup_tts_file(filename):
    """Clean up temporary TTS audio file."""
    try:
        # Security check
        if not filename or '..' in filename or not filename.startswith('/tmp/'):
            logger.warning(f"Unsafe cleanup path requested: {filename}")
            return jsonify({'error': 'Invalid file path'}), 400
        
        if os.path.exists(filename):
            os.unlink(filename)
            file_manager.active_files.discard(filename)  # Remove from tracking
            logger.info(f"Cleaned up TTS file: {filename}")
            return jsonify({'success': True, 'message': 'File cleaned up'})
        else:
            return jsonify({'success': False, 'message': 'File not found'}), 404
            
    except Exception as e:
        logger.error(f"TTS cleanup error: {e}")
        return jsonify({'error': 'Failed to cleanup TTS file'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'service': 'farm-voice-assistant',
        'version': '2.0-enhanced',
        'features': {
            'enhanced_transcription': True,
            'smart_tts_truncation': True,
            'automatic_file_cleanup': True,
            'improved_audio_processing': True
        }
    })

@app.route('/status', methods=['GET'])
def status_check():
    """Extended status endpoint with system information"""
    return jsonify({
        'status': 'running',
        'uptime': time.time(),
        'active_tts_files': len(file_manager.active_files),
        'cleanup_service_running': file_manager.running,
        'temp_directory': '/tmp',
        'upload_directory': UPLOAD_FOLDER
    })

def initialize_app():
    """Initialize enhanced app components"""
    try:
        # Start the file cleanup service
        file_manager.start_cleanup_service()
        
        # Verify required directories exist
        os.makedirs('/tmp', exist_ok=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        logger.info("Enhanced Farm Voice Assistant initialized successfully")
        logger.info(f"Features: Enhanced transcription, Smart TTS truncation, Auto file cleanup")
        logger.info(f"Upload folder: {UPLOAD_FOLDER}")
        logger.info(f"Temp folder: /tmp")
        
    except Exception as e:
        logger.error(f"App initialization failed: {e}")
        raise

if __name__ == '__main__':
    # Initialize enhanced features
    initialize_app()
    
    logger.info("Starting Enhanced Farm Voice Assistant Server")
    logger.info("Improvements: Better transcription, shorter TTS, automatic cleanup")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down cleanup service...")
        file_manager.stop_cleanup_service()
        logger.info("Enhanced Farm Voice Assistant stopped")