import re
import time
import joblib
import pandas as pd
import numpy as np
import os
import requests # To send alerts to Flask app
# --- Import TensorFlow ---
import tensorflow as tf
# --- End Import ---
from .snort_log_monitor import follow # Import the follower

# --- Configuration ---
MODELS_DIR = 'models'
# MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.joblib') # Keep for reference, we load NN now
NN_MODEL_PATH = os.path.join(MODELS_DIR, 'nn_model.h5') # <<-- Path to the Neural Network model
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
SCALED_COLS_PATH = os.path.join(MODELS_DIR, 'scaled_columns.joblib') # Which columns the scaler applies to
PROCESSED_TRAIN_PATH = os.path.join('data', 'processed_train.csv') # Need this to get column order
SNORT_LOG_FILE = 'snort.log'
FLASK_ALERT_URL = 'http://127.0.0.1:5000/alert' # URL of the Flask endpoint

# --- Load Model, Scaler, and Columns ---
try:
    print("Loading scaler and model configuration...")
    # --- MODIFIED: Load the Keras/TensorFlow model ---
    print(f"Attempting to load Neural Network model from: {NN_MODEL_PATH}")
    # Check if the NN model file actually exists before trying to load
    if not os.path.exists(NN_MODEL_PATH):
            raise FileNotFoundError(f"Neural Network model file not found at {NN_MODEL_PATH}. Make sure training completed successfully and saved 'nn_model.h5'.")
    # Suppress TensorFlow logging noise if needed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress INFO and WARNING messages
    tf.get_logger().setLevel('ERROR')     # Suppress TensorFlow logger errors

    model = tf.keras.models.load_model(NN_MODEL_PATH)
    print("Neural Network model loaded successfully.")
    # --- END MODIFICATION ---

    # Load scaler and feature info (no change here)
    print(f"Loading scaler from: {SCALER_PATH}")
    if not os.path.exists(SCALER_PATH): raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    print(f"Loading scaled column names from: {SCALED_COLS_PATH}")
    if not os.path.exists(SCALED_COLS_PATH): raise FileNotFoundError(f"Scaled columns file not found at {SCALED_COLS_PATH}")
    scaled_columns = joblib.load(SCALED_COLS_PATH)

    print(f"Loading processed training data columns from: {PROCESSED_TRAIN_PATH}")
    if not os.path.exists(PROCESSED_TRAIN_PATH): raise FileNotFoundError(f"Processed training data file not found at {PROCESSED_TRAIN_PATH}")
    processed_df_train = pd.read_csv(PROCESSED_TRAIN_PATH, nrows=0) # Only load header row to get columns
    model_features = processed_df_train.drop('label', axis=1).columns.tolist() # Get feature names from training data

    # Get input layer shape for info
    try:
        # Check if model has input_shape attribute (standard for Keras models)
        if hasattr(model, 'input_shape'):
            input_layer_shape = model.input_shape
            print(f"Model Expected input shape: {input_layer_shape}")
            # Note: Input shape might be (None, num_features). 'None' means variable batch size.
            # We still rely on model_features list derived from CSV for column matching/ordering.
        else:
            print("Model loaded, but could not access standard input_shape attribute.")

    except Exception as e:
            print(f"Model loaded, but encountered an error determining input shape via attribute: {e}")

    print(f"Scaler and feature names loaded successfully. Expecting {len(model_features)} features.")
    print(f"Scaled columns: {scaled_columns}")


except FileNotFoundError as e:
    print(f"Error loading required file: {e}")
    print("Please ensure preprocessing and training have been run successfully ('run_preprocessing.py' and 'run_training.py') and the necessary files exist in the 'models/' and 'data/' directories.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback for unexpected errors
    exit()

# --- Snort Log Parsing (Simplified Example) ---
# This regex is basic and might need significant adjustment based on your actual Snort alert format.
snort_log_pattern = re.compile(
    r"^(?P<timestamp>\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d{6})\s+" # Timestamp (e.g., 06/07-10:30:01.123456)
    r"\[\*\*\]\s+" # [**]
    r"\[\d+:\d+:\d+\]\s+" # Generator ID, Signature ID, Revision ID (e.g., [1:1000001:1])
    r"(?P<message>.*?)\s+" # The alert message
    r"\[\*\*\]\s+" # [**]
    r"(?:\[Classification:.*?\]\s+)?" # Optional Classification
    r"(?:\[Priority:.*?\]\s+)?" # Optional Priority
    r"\{(?P<protocol>\w+)\}\s+" # Protocol (e.g., {TCP})
    r"(?P<src_ip>[\d\.]+):?(?P<src_port>\d+)?\s+" # Source IP and optional Port
    r"->\s+" # Arrow
    r"(?P<dst_ip>[\d\.]+):?(?P<dst_port>\d+)?\s*$" # Destination IP and optional Port
)

def extract_features_from_snort(log_line):
    """
    Parses a Snort log line and attempts to extract/simulate features
    matching the NSL-KDD model input.
    THIS IS A MAJOR SIMPLIFICATION AND LIKELY INACCURATE FOR REAL TRAFFIC.
    """
    match = snort_log_pattern.match(log_line)
    if not match:
        return None, None # Indicate parsing failure

    parsed_data = match.groupdict()

    # --- Simulate NSL-KDD features ---
    features = {}
    features['protocol_type'] = parsed_data.get('protocol', 'unknown').lower() # tcp, udp, icmp
    dst_port = parsed_data.get('dst_port')
    service = 'other'
    if dst_port == '80': service = 'http'
    elif dst_port == '443': service = 'https'
    elif dst_port == '22': service = 'ssh'
    elif dst_port == '25': service = 'smtp'
    elif dst_port == '53': service = 'domain_u'
    # Add more service mappings...
    features['service'] = service

    flag = 'SF' # Default
    if 'reset' in parsed_data.get('message','').lower(): flag = 'REJ'
    if 'syn flood' in parsed_data.get('message','').lower(): flag = 'S0'
    features['flag'] = flag

    # Simulate numerical features (Defaults - NEEDS PROPER FLOW DATA for accuracy)
    default_numerical_val = 0
    simulated_features = {
        'duration': 0, 'src_bytes': 100, 'dst_bytes': 500, 'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 'num_failed_logins': 0,
        'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0,
        'is_host_login': 0, 'is_guest_login': 0, 'count': 1, 'srv_count': 1, 'serror_rate': 0.0, 'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 1, 'dst_host_srv_count': 1, 'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 1.0,
        'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
    }
    features.update(simulated_features) # Add the simulated values

    # --- Create DataFrame for prediction ---
    input_df = pd.DataFrame([features])

    # One-Hot Encode categorical features CONSISTENTLY with training
    input_df = pd.get_dummies(input_df, columns=['protocol_type', 'service', 'flag'], dummy_na=False)

    # Add any missing columns that were present during training (set to 0)
    # Use model_features list derived during loading for consistency
    missing_cols = set(model_features) - set(input_df.columns)
    for c in missing_cols:
        input_df[c] = 0

    # Ensure the order of columns matches the training data
    try:
        input_df = input_df[model_features]
    except KeyError as e:
        print(f"!!! Error aligning columns: Missing key {e}. Feature extraction likely inconsistent.")
        print(f"Model expects columns: {model_features}")
        print(f"Extracted columns before alignment: {list(input_df.columns)}")
        return None, parsed_data # Cannot proceed if columns don't match


    # Scale the numerical features using the loaded scaler
    # Important: Only scale columns that were originally scaled during training
    if scaler and scaled_columns:
       # Make sure only columns that EXIST in input_df and SHOULD be scaled are selected
       cols_to_scale_now = [col for col in scaled_columns if col in input_df.columns]
       if cols_to_scale_now:
          input_df[cols_to_scale_now] = scaler.transform(input_df[cols_to_scale_now])

    return input_df, parsed_data # Return the prepared features and original parsed data


def predict_threat(features_df):
    """Predicts using the loaded model."""
    if features_df is None:
        return None, None

    try:
        # Keras model's predict method gives probabilities directly for sigmoid output
        proba = model.predict(features_df, verbose=0) # verbose=0 to avoid progress bars per prediction
        probability = proba[0][0] # Extract the probability of the positive class (index 1 if using softmax, 0 for sigmoid)
        prediction = 1 if probability > 0.3 else 0 # Apply threshold

        # print(f"Debug: Raw Probability={proba[0]}, Selected Prob={probability}, Prediction={prediction}") # Debugging line if needed
        return prediction, probability

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def send_alert_to_dashboard(alert_data):
    """Sends the processed alert data to the Flask web dashboard."""
    try:
        response = requests.post(FLASK_ALERT_URL, json=alert_data, timeout=5)
        if response.status_code != 200:
            print(f"Failed to send alert to dashboard. Status: {response.status_code}, Response: {response.text}")
        # else: print("Alert sent to dashboard successfully.") # Can be verbose
    except requests.exceptions.RequestException as e:
        print(f"Error sending alert to dashboard: {e}")


def process_alerts():
    """Monitors the Snort log file and processes new alerts."""
    print(f"Monitoring Snort log file: {SNORT_LOG_FILE}")
    print(f"Using Neural Network model: {NN_MODEL_PATH}")
    print(f"Sending detected threats to: {FLASK_ALERT_URL}")
    print("Press Ctrl+C to stop.")
    print("DEBUG: Waiting for new lines...")

    while True: # Keep checking the file
        try:
            with open(SNORT_LOG_FILE, 'r') as logfile:
                # Use the follow generator to get new lines only
                loglines = follow(logfile)
                for line in loglines:
                    stripped_line = line.strip()
                    print("DEBUG: detected line: '{stripped_line}")
                    if not stripped_line: # Skip empty lines
                        print("DEBUG: Skipping empty line.")
                        continue

                    print(f"DEBUG: Attempting regex match...")
                    # print(f"\nProcessing line: {stripped_line}") # Can be verbose
                    features_df, parsed_info = extract_features_from_snort(stripped_line)

                    if features_df is not None:
                        print(f"DEBUG: Regex match SUCCESS. Parsed: {parsed_info}")
                        print(f"DEBUG: Attempting prediction...")
                        # print("Features extracted/simulated successfully.") # Can be verbose
                        prediction, probability = predict_threat(features_df)
                        print(f"DEBUG: Prediction={prediction}, Probability={probability}")

                        if prediction is not None: # Check if prediction was successful
                            if prediction == 1: # If model predicts an attack
                                print("DEBUG: Prediction is ATTACK (1). Preparing payload...")
                                print(f"*** THREAT DETECTED! *** (Prob: {probability:.4f}) Src: {parsed_info.get('src_ip', 'N/A')} Msg: {parsed_info.get('message', 'N/A')[:50]}...")
                                alert_payload = {
                                    "timestamp": parsed_info.get('timestamp', 'N/A'),
                                    "protocol": parsed_info.get('protocol', 'N/A'),
                                    "src_ip": parsed_info.get('src_ip', 'N/A'),
                                    "src_port": parsed_info.get('src_port', 'N/A'),
                                    "dst_ip": parsed_info.get('dst_ip', 'N/A'),
                                    "dst_port": parsed_info.get('dst_port', 'N/A'),
                                    "message": parsed_info.get('message', 'N/A').strip(),
                                    "prediction": "Attack",
                                    "probability": f"{probability:.4f}" if probability is not None else "N/A"
                                }
                                send_alert_to_dashboard(alert_payload)
                            else:
                                print(f"DEBUG: Prediction is NORMAL (0 or None). No alert sent.")
                                # print(f"Prediction: Normal Traffic (Prob: {probability:.4f})") # Optional: Log normal predictions too
                        else:
                            print(f"DEBUG: Prediction FAILED for line: {stripped_line}")
                            print(f"Prediction failed for line: {stripped_line}")

                    else:
                        print(f"DEBUG: Regex match FAILED for line.")
                        # print(f"Line did not match Snort pattern or failed extraction: {stripped_line}") # Can be verbose

        except FileNotFoundError:
            # Check periodically if file has been created
            if not os.path.exists(SNORT_LOG_FILE):
                 print(f"Waiting for log file '{SNORT_LOG_FILE}' to be created...")
            else:
                # File existed but couldn't be opened (maybe permissions?), retry later
                print(f"Could not open log file '{SNORT_LOG_FILE}', retrying...")
            time.sleep(5) # Wait before trying to open again
        except KeyboardInterrupt:
            print("\nAlert processor stopped by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred in the monitoring loop: {e}")
            import traceback
            traceback.print_exc()
            print("Restarting monitoring in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    process_alerts()