Bemba Automatic Speech Recognition API

This project is a Flask-based web server that provides an API for Automatic Speech Recognition (ASR) using a pre-trained Bemba-language model. It leverages the Hugging Face transformers library to perform high-performance, real-time audio transcription. The server is designed for production use, featuring API key authentication, rate limiting, and GPU support.

Features

Bemba ASR: Transcribes audio files and URLs into text using the NextInnoMind/next_bemba_ai model.

API Key Authentication: Securely authenticates API requests using a simple, file-based API key system.

Rate Limiting: Prevents abuse and ensures fair usage with a configurable rate-limiting mechanism.

GPU Acceleration: Automatically detects and utilizes an NVIDIA GPU (via PyTorch) for faster transcription.

Concurrent Processing: Optimized for concurrency using Gunicorn, allowing multiple users to make requests simultaneously.

File & URL Support: Accepts audio input either through file uploads or by providing a URL to a publicly accessible audio file.

Health Check: An endpoint to monitor the application's status and model readiness.

Prerequisites

Before you begin, ensure you have the following installed on your system:

Python 3.8+

pip (Python package installer)

git

virtualenv or conda (recommended for dependency management)

Installation

Follow these steps to set up the project locally.

Clone the Repository

git clone [https://github.com/SilasChalwe/farmvoiceassistant_backend.git](https://github.com/SilasChalwe/farmvoiceassistant_backend.git)
cd farmvoiceassistant_backend



Set up a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install Dependencies
Install all required Python packages. The ASR model requires torch and transformers.

pip install Flask gunicorn torch transformers Flask-Cors



Note: For GPU support, you must install the CUDA-enabled version of PyTorch. Refer to the official PyTorch website for instructions.

Configuration

You can configure the application using environment variables. It is highly recommended to set these before starting the server.

MAX_FILE_SIZE_MB: Maximum allowed size for uploaded audio files in megabytes. (Default: 100)

MODEL_NAME: The name of the Hugging Face model to use for transcription. (Default: NextInnoMind/next_bemba_ai)

RATE_LIMIT_REQUESTS: The number of allowed requests per user within the rate limit window. (Default: 10)

RATE_LIMIT_WINDOW: The time window in seconds for the rate limit. (Default: 60)

Example:

export MAX_FILE_SIZE_MB=50
export RATE_LIMIT_REQUESTS=20



Usage

This application is designed to be run with Gunicorn. A convenience script is provided to start the server.

Start the Server
Use the start_gunicorn.sh script to launch the application with four workers in the background.

./start_app.sh



The server will be running at http://0.0.0.0:5000 and all logs will be saved to app.log. The Gunicorn process ID will be stored in gunicorn.pid.

Generate an API Key
The first step to using the API is to generate a key. Make a POST request to the /api/generate-key endpoint.

curl -X POST http://localhost:5000/api/generate-key



The response will contain your unique API key. Save this key immediately, as it will not be shown again.

Transcribe an Audio File
Use your newly generated API key in the X-API-Key header to transcribe an audio file.

curl -X POST http://localhost:5000/transcribe \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -F "file=@/path/to/your/audio.wav"



Transcribe on the live API endpoint
You can also test the API on the live server. Note that the live server uses a different base URL.

curl -X POST [https://api.nextinnomind.me/transcribe](https://api.nextinnomind.me/transcribe) \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -F "file=@/path/to/your/audio.wav"



Transcribe an Audio from a URL
You can also transcribe audio directly from a publicly accessible URL.

Local Server

curl -X POST http://localhost:5000/transcribe_url \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -d '{"audio_url": "[https://example.com/audio/sample.mp3](https://example.com/audio/sample.mp3)"}'


Live API Endpoint

curl -X POST [https://api.nextinnomind.me/transcribe_url](https://api.nextinnomind.me/transcribe_url) \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -d '{"audio_url": "[https://example.com/audio/sample.mp3](https://example.com/audio/sample.mp3)"}'


API Endpoints

POST /api/generate-key: Generates and returns a new API key.

POST /transcribe: Transcribes an uploaded audio file. Requires X-API-Key header and a file in the request body.

POST /transcribe_url: Transcribes audio from a provided URL. Requires X-API-Key header and a JSON body with the audio_url.

GET /health: A simple health check to verify the server and model status.

GET /: Renders the web interface for the application.