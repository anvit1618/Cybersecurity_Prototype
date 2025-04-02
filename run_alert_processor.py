# Simple script to execute the alert processing component
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from integration.alert_processor import process_alerts

if __name__ == "__main__":
    print("--- Starting Alert Processor ---")
    process_alerts()
    print("\n--- Alert Processor Stopped ---")