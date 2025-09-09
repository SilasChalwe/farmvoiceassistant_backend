from flask import Flask, request, jsonify
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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

def preprocess_audio(audio_path):
    """
    Preprocess audio with robust error handling - accepts any format
    """
    try:
        # Load audio file (pydub can handle multiple formats)
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Resample to 16kHz if needed (common for ASR models)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        # Check if audio is too quiet and amplify
        if audio.dBFS < -40:
            logger.warning("Audio is too quiet, amplifying by 15dB")
            audio = audio + 15  # Amplify by 15dB

        # Normalize audio
        normalized_audio = normalize(audio)

        # Apply noise reduction filters
        # Remove low-frequency rumble
        filtered_audio = normalized_audio.high_pass_filter(100)
        # Remove high-frequency noise
        filtered_audio = filtered_audio.low_pass_filter(3500)

        # Remove silence (but be less aggressive)
        try:
            non_silent_audio = filtered_audio.strip_silence(
                silence_len=400,
                silence_thresh=-30,
                padding=200
            )
            
            # If stripping silence removed too much, use filtered audio
            if len(non_silent_audio) < 1000:  # Less than 1 second
                logger.warning("Too much silence removed, using filtered audio")
                non_silent_audio = filtered_audio
                
        except Exception as silence_error:
            logger.warning(f"Silence removal failed: {silence_error}, using filtered audio")
            non_silent_audio = filtered_audio

        # Save processed audio as proper WAV
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        non_silent_audio.export(temp_file.name, format='wav', 
                               parameters=["-ac", "1", "-ar", "16000"])

        return temp_file.name

    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        # Return original path as fallback
        return audio_path

def remove_repetitive_text(text):
    """
    Remove repetitive patterns and everything after the repetition starts
    Returns only the meaningful content before repetition begins
    """
    if not text or len(text.strip()) < 10:
        return text
    
    text = text.strip()
    
    # Enhanced repetitive patterns to detect more cases
    patterns = [
        # Exact phrase repetitions (3+ words)
        r"(\b\w+(?:\s+\w+){2,}\b)(?:\s+\1){2,}",
        # Two word phrase repetitions  
        r"(\b\w+\s+\w+\b)(?:\s+\1){3,}",
        # Single word repetitions (4+ times)
        r"(\b\w+\b)(?:\s+\1){4,}",
        # Character repetitions (6+ times)
        r"(.)\1{6,}",
        # Common garbage patterns
        r"[aeiou]{5,}",  # Vowel repetitions
        r"[bcdfghjklmnpqrstvwxyz]{4,}",  # Consonant repetitions
        r"\b[hmm]{3,}\b",  # Humming sounds
        r"\b[uhh]{3,}\b",  # Hesitation sounds
        r"\b[ahh]{3,}\b",  # Breathing sounds
    ]
    
    earliest_repetition = len(text)
    found_repetition = False
    
    # Find the earliest repetition
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            repetitive_start = match.start()
            repetitive_content = match.group()
            
            # Only consider it if there's meaningful content before
            if repetitive_start > 15 and repetitive_start < earliest_repetition:
                earliest_repetition = repetitive_start
                found_repetition = True
                logger.info(f"Found repetition pattern: '{repetitive_content[:30]}...' at position {repetitive_start}")
    
    # If repetition found, return text before it
    if found_repetition:
        meaningful_text = text[:earliest_repetition].strip()
        # Clean up any trailing punctuation or incomplete words
        meaningful_text = re.sub(r'\s+$', '', meaningful_text)
        meaningful_text = re.sub(r'[,.\s]+$', '', meaningful_text)
        
        if len(meaningful_text) > 10:
            logger.info(f"Returning clean text before repetition: '{meaningful_text[:50]}...'")
            return meaningful_text
    
    # Additional check: sliding window approach for subtle repetitions
    words = text.split()
    if len(words) > 15:
        # Check for repeating sequences of different lengths
        for seq_len in range(3, 8):  # Check sequences of 3-7 words
            for i in range(len(words) - seq_len * 2):
                sequence = words[i:i + seq_len]
                next_sequence = words[i + seq_len:i + seq_len * 2]
                
                # If we find identical sequences
                if sequence == next_sequence:
                    # Check if this pattern continues
                    repetition_count = 1
                    check_pos = i + seq_len * 2
                    
                    while (check_pos + seq_len <= len(words) and 
                           words[check_pos:check_pos + seq_len] == sequence):
                        repetition_count += 1
                        check_pos += seq_len
                    
                    # If pattern repeats 2+ times and we have content before
                    if repetition_count >= 1 and i > 5:
                        meaningful_text = ' '.join(words[:i]).strip()
                        if len(meaningful_text) > 10:
                            logger.info(f"Found sliding window repetition: {repetition_count + 1} times, sequence: {' '.join(sequence)}")
                            return meaningful_text
    
    # Check unique word ratio for overall repetitiveness
    if len(words) > 20:
        unique_words = len(set(words))
        unique_ratio = unique_words / len(words)
        
        if unique_ratio < 0.25:  # Very repetitive
            # Find a good cutoff point (around 1/3 of the text)
            cutoff = min(len(words) // 3, 50)  # Max 50 words
            meaningful_text = ' '.join(words[:cutoff]).strip()
            
            if len(meaningful_text) > 15:
                logger.info(f"Low unique ratio ({unique_ratio:.2f}), returning first {cutoff} words")
                return meaningful_text
    
    return text

def validate_and_clean_transcription(text):
    """
    Validate transcription, remove repetitions, and return clean text
    """
    if not text or text.strip() == '':
        return "No speech detected. Please try again with clearer audio."

    # Normalize whitespace
    text = ' '.join(text.split()).strip()
    
    # Remove common transcription artifacts
    text = re.sub(r'\b(um|uh|ah|er|hmm)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()

    # First, remove repetitive content
    clean_text = remove_repetitive_text(text)
    
    # If removal changed the text, use the cleaned version
    if clean_text != text and len(clean_text) > 5:
        logger.info(f"Text cleaned. Original: {len(text)} chars -> Cleaned: {len(clean_text)} chars")
        text = clean_text

    # Final validation
    words = text.split()
    
    # Check for minimum meaningful content
    if len(words) < 2:
        return "Not enough words detected. Please speak complete sentences."
    
    if len(text) < 5:
        return "Text too short. Please speak more clearly."

    # Remove any remaining garbage patterns
    garbage_patterns = [
        r"[aeiou]{4,}",       # Multiple vowels
        r"[bcdfghjklmnpqrstvwxyz]{3,}",  # Multiple consonants  
        r"\b[h]{3,}\b",       # Multiple h's
        r"\b[s]{3,}\b",       # Multiple s's
        r"(.)\1{5,}",         # Any character repeated 6+ times
    ]

    text_modified = False
    for pattern in garbage_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # Remove the garbage pattern
            cleaned = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
            cleaned = ' '.join(cleaned.split()).strip()
            
            if len(cleaned) > 5 and len(cleaned) > len(text) * 0.3:  # Keep if substantial content remains
                text = cleaned
                text_modified = True
                logger.info(f"Removed garbage pattern, remaining: '{text[:50]}...'")

    # Final check - if we have very little left after cleaning
    if len(text) < 10:
        return "Background noise detected. Please speak closer to the microphone."

    # Ensure we end with proper punctuation for better user experience
    if text and not text[-1] in '.!?':
        text = text + '.'

    return text

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

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

        # Preprocess audio
        processed_audio_path = preprocess_audio(filepath)
        logger.info("Audio preprocessing completed")

        try:
            # Get transcription
            raw_text = dummy_transcribe(processed_audio_path)
            logger.info(f"Raw transcription: '{raw_text[:100]}...' ({len(raw_text)} chars)")

            # Clean and validate transcription
            final_text = validate_and_clean_transcription(raw_text)
            
            # Check if cleaning resulted in an error message
            error_indicators = [
                "No speech detected",
                "Not enough words", 
                "Background noise detected",
                "Text too short"
            ]
            
            is_error = any(final_text.startswith(indicator) for indicator in error_indicators)
            
            if is_error:
                return jsonify({
                    'success': False,
                    'error': final_text,
                    'debug_info': {
                        'raw_text_preview': raw_text[:200] + '...' if len(raw_text) > 200 else raw_text,
                        'raw_length': len(raw_text),
                        'processed_length': len(final_text)
                    }
                }), 400

            # Return successful transcription
            response_data = {
                'success': True,
                'text': final_text,
                'metadata': {
                    'original_length': len(raw_text),
                    'cleaned_length': len(final_text),
                    'processing_applied': len(raw_text) != len(final_text),
                    'word_count': len(final_text.split()),
                    'confidence': 'high' if len(final_text) > 20 else 'medium'
                }
            }
            
            # Add debug info if text was significantly modified
            if len(raw_text) > len(final_text) * 1.5:
                response_data['debug_info'] = {
                    'raw_preview': raw_text[:100] + '...' if len(raw_text) > 100 else raw_text,
                    'cleaning_applied': True
                }

            logger.info(f"Successful transcription: '{final_text[:50]}...' ({len(final_text)} chars)")
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
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Transcription processing failed. Please try again.',
            'debug_info': {
                'error_type': type(e).__name__,
                'error_message': str(e)
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'farm-voice-assistant'})

if __name__ == '__main__':
    logger.info("Starting Farm Voice Assistant Server")
    app.run(host='0.0.0.0', port=5000, debug=True)
