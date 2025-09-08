from flask import Flask, request, jsonify
import os
from api.transcribe_helper import dummy_transcribe
from api.file_helper import delete_file, get_unique_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    original_filename = file.filename
    filename = get_unique_filename(original_filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    file.save(filepath)
    
    try:
        # Get dummy transcription (or real transcription later)
        text = dummy_transcribe(filename)
        return jsonify({'text': text})
    finally:
        # Delete file after transcription is done, regardless of success or error
        delete_file(filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
