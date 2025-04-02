import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Define column names for NSL-KDD dataset (adjust if your file differs)
# These are based on standard NSL-KDD documentation. Crucially includes the 'label' and 'difficulty'
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

# Define paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
TRAIN_FILE = os.path.join(DATA_DIR, 'KDDTrain+.txt') # Use the full training set
TEST_FILE = os.path.join(DATA_DIR, 'KDDTest+.txt') # Use the standard test set
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
PROCESSED_TRAIN_PATH = os.path.join(DATA_DIR, 'processed_train.csv')
PROCESSED_TEST_PATH = os.path.join(DATA_DIR, 'processed_test.csv')

# Ensure model directory exists
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True) # Ensure data dir exists too

def load_data():
    """Loads NSL-KDD train and test data."""
    try:
        print(f"Loading training data from {TRAIN_FILE}...")
        # Try loading with headers, if it fails, load without and assign names
        try:
            df_train = pd.read_csv(TRAIN_FILE, header=None, names=col_names)
        except pd.errors.ParserError:
             print(f"Could not parse {TRAIN_FILE} assuming no header and standard columns.")
             # Potentially add more robust handling here if needed
             raise

        print(f"Loading testing data from {TEST_FILE}...")
        try:
             df_test = pd.read_csv(TEST_FILE, header=None, names=col_names)
        except pd.errors.ParserError:
            print(f"Could not parse {TEST_FILE} assuming no header and standard columns.")
            raise

        print("Data loaded successfully.")
        return df_train, df_test
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Please ensure '{os.path.basename(TRAIN_FILE)}' and '{os.path.basename(TEST_FILE)}' are in the '{DATA_DIR}' directory.")
        exit() # Exit if data is missing
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit()

def preprocess_data(df_train, df_test):
    """Preprocesses the data: encoding, scaling, and creating binary labels."""
    print("Starting preprocessing...")

    # --- Combine temporarily for consistent encoding/scaling ---
    combined_len = len(df_train)
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # --- Feature Engineering/Selection (Minimal for this prototype) ---
    # Drop 'difficulty' as it's not typically used as a feature
    if 'difficulty' in df_combined.columns:
        df_combined = df_combined.drop('difficulty', axis=1)

    # --- Identify Numerical and Categorical Features ---
    # Explicitly define categorical columns based on NSL-KDD
    categorical_cols = ['protocol_type', 'service', 'flag']
    numerical_cols = [col for col in df_combined.columns if col not in categorical_cols + ['label']]

    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")

    # --- Handle Categorical Features (One-Hot Encoding) ---
    print("Applying one-hot encoding...")
    df_combined = pd.get_dummies(df_combined, columns=categorical_cols, dummy_na=False)

    # --- Create Binary Label ('normal' vs 'attack') ---
    print("Creating binary labels...")
    df_combined['label'] = df_combined['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # --- Split back into Train and Test ---
    df_train_processed = df_combined[:combined_len]
    df_test_processed = df_combined[combined_len:]

    # Separate features (X) and labels (y)
    X_train = df_train_processed.drop('label', axis=1)
    y_train = df_train_processed['label']
    X_test = df_test_processed.drop('label', axis=1)
    y_test = df_test_processed['label']

    # --- Scaling Numerical Features ---
    # Get numerical columns AFTER one-hot encoding (names might change slightly)
    # We scale only the *original* numerical columns before they got mixed with OHE columns
    numerical_cols_to_scale = [col for col in numerical_cols if col in X_train.columns]
    print(f"Scaling numerical columns: {numerical_cols_to_scale}")

    if numerical_cols_to_scale: # Ensure there are numerical columns to scale
        scaler = StandardScaler()
        X_train[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
        X_test[numerical_cols_to_scale] = scaler.transform(X_test[numerical_cols_to_scale])

        # --- Save the Scaler ---
        print(f"Saving scaler to {SCALER_PATH}...")
        joblib.dump(scaler, SCALER_PATH)
        # Also save the list of scaled columns, might be useful later
        joblib.dump(numerical_cols_to_scale, os.path.join(MODELS_DIR,'scaled_columns.joblib'))

    else:
        print("Warning: No numerical columns identified for scaling.")
        scaler = None # No scaler needed

    # --- Align columns ---
    # Ensure test set has same columns as train set after OHE, filling missing with 0
    train_cols = X_train.columns
    test_cols = X_test.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train[c] = 0
    X_test = X_test[train_cols] # Ensure order is the same


    print("Preprocessing finished.")
    # Save processed data
    print(f"Saving processed training data to {PROCESSED_TRAIN_PATH}")
    pd.concat([X_train, y_train], axis=1).to_csv(PROCESSED_TRAIN_PATH, index=False)
    print(f"Saving processed testing data to {PROCESSED_TEST_PATH}")
    pd.concat([X_test, y_test], axis=1).to_csv(PROCESSED_TEST_PATH, index=False)

    return X_train, y_train, X_test, y_test, scaler, train_cols

if __name__ == "__main__":
    df_train_raw, df_test_raw = load_data()
    preprocess_data(df_train_raw, df_test_raw)
    print("Data preprocessing script finished.")