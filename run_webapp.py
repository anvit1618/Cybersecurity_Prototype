# Simple script to execute the Flask web application
import sys
import os

# Add src directory to Python path
# This ensures that 'from webapp.app import app, socketio' works correctly
# when run from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from webapp.app import app, socketio # Import app and socketio from the correct location

if __name__ == "__main__":
    print("--- Starting Web Application (Flask-SocketIO) ---")
    print("Access dashboard at: http://127.0.0.1:5000")
    # Use socketio.run() for development; consider gunicorn for production
    # Use debug=True cautiously as it can expose security risks
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, use_reloader=False)