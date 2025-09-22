#!/bin/bash

# --- Configuration for the Gunicorn application ---
# The entry point for the Flask application.
APP_NAME="app:app"

# Directory where the script is located
SCRIPT_DIR=$(dirname "$0")

# Change to the application's directory to ensure Gunicorn
# can find the app module and log files.
cd "$SCRIPT_DIR"

# Number of worker processes as specified.
WORKERS=8

# The address and port to bind to.
BIND_ADDRESS="0.0.0.0:5000"

# Log file path for all Gunicorn output (stdout and stderr).
APP_LOG="app.log"

# --- Main Script Execution ---
echo "Starting Gunicorn server in the background..."

# Start Gunicorn with specified workers and bind address.
# All output is redirected to app.log, and the process runs in the background.
gunicorn -w "$WORKERS" -b "$BIND_ADDRESS" "$APP_NAME" > "$APP_LOG" 2>&1 &

# Store the process ID of the Gunicorn master process in gunicorn.pid.
echo $! > gunicorn.pid

echo "Gunicorn is running with PID $(cat gunicorn.pid) and logging to $APP_LOG"
