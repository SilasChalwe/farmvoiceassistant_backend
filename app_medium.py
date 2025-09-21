import os
import tempfile
from multiprocessing import Process, Queue, Manager, Lock
import shortuuid
import torch
from transformers import pipeline, AutoTokenizer
from flask import Flask, request, jsonify

# --- Application setup ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB

# --- Asynchronous task handling with multiprocessing ---
task_queue = Queue()
manager = Manager()
transcription_status = manager.dict()

# Define the number of worker processes
NUM_WORKERS = 4  # Set based on your system's CPU cores

# Global resources for pre-loading
transcriber_global = None
device = 0 if torch.cuda.is_available() else -1
if device == -1:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_num_interop_threads(2)
    torch.backends.mkldnn.enabled = True

def load_transcriber_model():
    """Load the model and tokenizer once for all processes."""
    MODEL_ID = "NextInnoMind/next_bemba_ai_medium"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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

    transcriber_local = pipeline(**pipeline_kwargs)
    
    if device == -1:
        try:
            transcriber_local.model.eval()
            if hasattr(torch.backends, "mkldnn") and torch.backends.mkldnn.is_available():
                transcriber_local.model = transcriber_local.model.to(memory_format=torch.channels_last)
        except Exception as e:
            app.logger.warning(f"Failed to apply some CPU optimizations: {e}")
            
    return transcriber_local, FORCED_DECODER_IDS


def worker(queue, status_dict, transcriber, forced_decoder_ids, lock):
    """Worker function to process transcription tasks."""
    while True:
        task_id, file_path = queue.get()
        app.logger.info(f"Worker process processing task {task_id} for file {file_path}")
        
        transcription = "Transcription failed due to an unknown error."
        try:
            # Use the pre-loaded transcriber instance
            with lock: # Protect access to the transcriber
                generation_kwargs = {
                    "forced_decoder_ids": forced_decoder_ids,
                    "max_new_tokens": 256,
                    "num_beams": 4,
                    "do_sample": False,
                    "use_cache": True,
                    "no_repeat_ngram_size": 3,
                    "repetition_penalty": 1.2,
                }
                
                with torch.no_grad():
                    result = transcriber(file_path, generate_kwargs=generation_kwargs)

                raw_text = (result.get("text") or "").strip()
                if raw_text:
                    transcription = raw_text
                else:
                    transcription = "No clear speech detected"

        except Exception as e:
            app.logger.error(f"Transcription failed for task {task_id}: {e}")
            transcription = f"Transcription Failed: {e}"
        finally:
            status_dict[task_id] = {
                'status': 'completed',
                'result': transcription
            }
            
            try:
                os.remove(file_path)
                app.logger.info(f"Cleaned up temporary file for task {task_id}")
            except OSError as e:
                app.logger.warning(f"Failed to remove temporary file for task {task_id}: {e}")

# --- API Endpoints ---
@app.route('/transcribe', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.flac')
    file.save(temp_file.name)
    temp_file.close()
    
    task_id = shortuuid.uuid()
    
    transcription_status[task_id] = {'status': 'pending'}
    task_queue.put((task_id, temp_file.name))
    
    app.logger.info(f"Task {task_id} queued for processing.")
    
    return jsonify({
        'message': 'File uploaded and transcription started',
        'task_id': task_id
    }), 202

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    status = transcription_status.get(task_id)
    if not status:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(status), 200

@app.route('/')
def home():
    return "<h1>Bemba AI Transcription Service</h1><p>Send a POST request with an audio file to the /transcribe endpoint to get started.</p>"

if __name__ == '__main__':
    # Pre-load model resources once
    transcriber_global, FORCED_DECODER_IDS_global = load_transcriber_model()
    
    # Use a lock to synchronize access to the transcriber model
    # (necessary if model is not thread-safe within a process)
    model_lock = Lock()

    # Start the background worker processes
    for i in range(NUM_WORKERS):
        p = Process(target=worker, args=(task_queue, transcription_status, transcriber_global, FORCED_DECODER_IDS_global, model_lock))
        p.daemon = True
        p.start()
        app.logger.info(f"Started worker process: {p.name}")

    app.run(debug=True, host='0.0.0.0', port=5000)
