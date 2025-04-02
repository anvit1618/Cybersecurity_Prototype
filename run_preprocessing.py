# Simple script to execute the preprocessing step
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_processing.preprocess import load_data, preprocess_data

print("--- Running Data Preprocessing ---")
try:
    df_train_raw, df_test_raw = load_data()
    if df_train_raw is not None and df_test_raw is not None:
        preprocess_data(df_train_raw, df_test_raw)
        print("\n--- Data Preprocessing Completed Successfully ---")
    else:
        print("\n--- Data Preprocessing Failed (Data Loading Error) ---")
except Exception as e:
    print(f"\n--- An error occurred during preprocessing: {e} ---")