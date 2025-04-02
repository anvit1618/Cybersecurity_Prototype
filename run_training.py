# Simple script to execute the model training step
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from ml_models.train import load_processed_data, train_models

print("--- Running Model Training ---")
try:
    X_train, y_train, X_test, y_test = load_processed_data()
    train_models(X_train, y_train, X_test, y_test)
    print("\n--- Model Training Completed Successfully ---")
except FileNotFoundError:
     print("\n--- Model Training Failed: Preprocessed data not found. Run preprocessing first. ---")
except Exception as e:
    print(f"\n--- An error occurred during training: {e} ---")