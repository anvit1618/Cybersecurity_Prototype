from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import os

# --- Flask Application Setup ---
# Determine the absolute path to the templates and static folders
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SECRET_KEY'] = 'your_secret_key_here!' # Change this for production!
socketio = SocketIO(app, async_mode='threading') # Use threading for compatibility with requests

# In-memory storage for recent alerts (replace with database in production)
recent_alerts = []
MAX_ALERTS = 50 # Limit the number of alerts stored in memory

print(f"Template folder: {app.template_folder}")
print(f"Static folder: {app.static_folder}")


@app.route('/')
def index():
    """Serves the main dashboard page."""
    # Pass the currently stored alerts to the template when it first loads
    return render_template('index.html', alerts=list(recent_alerts))

@app.route('/alert', methods=['POST'])
def receive_alert():
    """Endpoint for the alert_processor to send new alerts."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    alert_data = request.get_json()
    print(f"Received alert via POST: {alert_data}") # Log received alert

    # Add alert to our in-memory list
    recent_alerts.insert(0, alert_data) # Add to the beginning
    # Keep the list size manageable
    if len(recent_alerts) > MAX_ALERTS:
        recent_alerts.pop()

    # Emit the alert data to all connected SocketIO clients
    try:
        # Emit under the 'new_alert' event name, sending the dictionary
        socketio.emit('new_alert', alert_data)
        print("Emitted alert via SocketIO")
    except Exception as e:
        print(f"Error emitting alert via SocketIO: {e}")


    return jsonify({"status": "success", "message": "Alert received"}), 200

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """Handles new client connections."""
    print('Client connected')
    # Optionally send existing alerts on connection if needed, though handled by initial load

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections."""
    print('Client disconnected')


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server on http://127.0.0.1:5000")
    # Use socketio.run for development server that supports WebSockets
    # Use host='0.0.0.0' to make accessible on local network (use with caution)
    socketio.run(app, debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    # use_reloader=False prevents running setup twice in debug mode which can be confusing