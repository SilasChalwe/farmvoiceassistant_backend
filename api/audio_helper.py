import tempfile
from pydub import AudioSegment

def preprocess_audio(audio_path):
    """
    Convert audio to mono 16kHz WAV for transcription.
    Returns path to the processed temporary file.
    """
    audio = AudioSegment.from_file(audio_path)
    
    # Convert to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Convert to 16 kHz
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)

    # Export to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    audio.export(temp_file.name, format='wav')
    return temp_file.name
