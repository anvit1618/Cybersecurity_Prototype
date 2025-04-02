
# AI-Powered Cybersecurity Threat Detection Prototype

This project demonstrates a functional prototype for detecting cybersecurity threats in near real-time using machine learning models trained on the NSL-KDD dataset and integrating with simulated Snort alerts.

---

## Project Components

1. **Data Preprocessing**  
   Cleans, encodes, and scales the NSL-KDD dataset.

2. **ML Model Training**  
   Trains Random Forest, SVM, and a simple Neural Network. Saves the best performing model (Random Forest by default) along with the scaler.

3. **Alert Processor**  
   Monitors a `snort.log` file, parses alerts, extracts features (simulated), uses the trained model to predict threats, and sends findings to the web dashboard.

4. **Web Dashboard**  
   A Flask application using SocketIO to display incoming threat alerts in real-time.

---

## Prerequisites

- Python 3.8+
- Git
- (Optional but recommended) A virtual environment tool (`venv`)
- (For full integration) Snort installed and configured to output alerts to a file named `snort.log` in the project root.

---

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url> cybersecurity_prototype
cd cybersecurity_prototype
```

### 2. Create and Activate a Virtual Environment

**Option A** (Standard approach, if pip is already available)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

**Option B** (If pip is not available or you prefer manual pip installation)

```bash
# Create a virtual environment WITHOUT pip
python3 -m venv venv --without-pip

# Activate the virtual environment
source venv/bin/activate

# Download the official get-pip script
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip inside the virtual environment
python get-pip.py

# (Optional) Remove get-pip script
rm get-pip.py
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the NSL-KDD Dataset

- You typically need `KDDTrain+.txt` and `KDDTest+.txt`.
- Place these files inside the `data/` directory.
- **Important**: The `preprocess.py` script assumes the files have standard column names. If your files lack headers, you may need to manually add them or modify the script. Refer to the NSL-KDD documentation for column names.

---

## Running the Prototype

### Run Data Preprocessing

```bash
python run_preprocessing.py
```

This creates processed data files and a `scaler.joblib` file in the `models/` directory.

### Run Model Training

```bash
python run_training.py
```

This trains the models and saves the best one (default: RandomForest) as `best_model.joblib` in the `models/` directory.

### (Optional) Prepare the Snort Log

- If you have Snort running, ensure it’s configured to write alerts to `cybersecurity_prototype/snort.log`.
- Alternatively, manually add some sample Snort alert lines to `snort.log` for testing (refer to the regex in `alert_processor.py` for the expected format).

### Start the Web Application

Open a new terminal in the project root (and activate the virtual environment):

```bash
python run_webapp.py
```

Access the dashboard in your browser:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

### Start the Alert Processor

Open another new terminal in the project root (and activate the virtual environment):

```bash
python run_alert_processor.py
```

This script monitors `snort.log` in real-time. As new alerts appear (matching the expected format), it processes them, predicts threat levels, and sends alerts to the web dashboard via HTTP POST requests. Alerts should appear automatically on the web page.

---

## Stopping

Press `Ctrl + C` in the terminals running both the web app and the alert processor.

---

## Limitations & Simplifications

- **Real-time**  
  The alert processor tails a log file, which is near real-time but not instantaneous packet capture.

- **Snort Feature Extraction**  
  The `alert_processor.py` simulates feature extraction from Snort log messages. A production integration would require robust parsing and mapping Snort fields to all the features the model was trained on (e.g., duration, bytes, flags).

- **Dataset**  
  Uses NSL-KDD. Real-world traffic is more varied and complex.

- **Error Handling**  
  Minimal error handling is implemented; production environments need more robust handling.

- **Scalability**  
  This prototype is not optimized for heavy or high-throughput deployments.

- **Security**  
  The prototype lacks advanced security measures for both the web app and data communication.



