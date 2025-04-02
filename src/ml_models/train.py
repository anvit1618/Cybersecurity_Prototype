import pandas as pd
import joblib
import os
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split # Optional: Further split train data for validation

# Define paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
PROCESSED_TRAIN_PATH = os.path.join(DATA_DIR, 'processed_train.csv')
PROCESSED_TEST_PATH = os.path.join(DATA_DIR, 'processed_test.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.joblib') # Save best sklearn model here
NN_MODEL_PATH = os.path.join(MODELS_DIR, 'nn_model.h5') # Save NN model separately

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)

def load_processed_data():
    """Loads preprocessed data."""
    print("Loading processed data...")
    try:
        df_train = pd.read_csv(PROCESSED_TRAIN_PATH)
        df_test = pd.read_csv(PROCESSED_TEST_PATH)
    except FileNotFoundError:
        print("Error: Processed data files not found.")
        print(f"Please run 'run_preprocessing.py' first.")
        exit()

    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']

    print("Processed data loaded.")
    return X_train, y_train, X_test, y_test

def evaluate_model(name, model, X_test, y_test):
    """Evaluates a model and prints metrics."""
    print(f"\n--- Evaluating {name} ---")
    predictions = model.predict(X_test)
    # For NN with sigmoid output, convert probabilities to binary classes
    if isinstance(model, tf.keras.Model):
        predictions = (predictions > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    return accuracy, f1

def train_models(X_train, y_train, X_test, y_test):
    """Trains RF, SVM, and NN models."""
    models = {}
    results = {}

    # --- Random Forest ---
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Use more trees
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    results['Random Forest'] = evaluate_model('Random Forest', rf_model, X_test, y_test)

    # --- Support Vector Machine (SVM) ---
    # Note: SVM can be very slow on large datasets like NSL-KDD.
    # Consider using a smaller subset or LinearSVC for faster training initially.
    print("\nTraining SVM (Linear Kernel)...")
    # Using LinearSVC as it's faster for large datasets than SVC(kernel='linear')
    from sklearn.svm import LinearSVC
    svm_model = LinearSVC(C=1.0, random_state=42, dual=True, max_iter=1000) # Use dual=True if n_samples > n_features is often false after OHE
    try:
        svm_model.fit(X_train, y_train)
        models['SVM'] = svm_model
        # LinearSVC uses decision_function, need predict method for consistent eval
        svm_predictions = svm_model.predict(X_test)
        accuracy_svm = accuracy_score(y_test, svm_predictions)
        precision_svm = precision_score(y_test, svm_predictions, zero_division=0)
        recall_svm = recall_score(y_test, svm_predictions, zero_division=0)
        f1_svm = f1_score(y_test, svm_predictions, zero_division=0)
        cm_svm = confusion_matrix(y_test, svm_predictions)
        print(f"SVM Accuracy: {accuracy_svm:.4f}")
        print(f"SVM Precision: {precision_svm:.4f}")
        print(f"SVM Recall: {recall_svm:.4f}")
        print(f"SVM F1-Score: {f1_svm:.4f}")
        print("SVM Confusion Matrix:")
        print(cm_svm)

        results['SVM'] = (accuracy_svm, f1_svm)

    except Exception as e:
         print(f"Could not train SVM. Error: {e}. SVM can be memory/time intensive.")
         results['SVM'] = (0.0, 0.0) # Assign default low score


    # --- Neural Network (TensorFlow/Keras) ---
    print("\nTraining Neural Network...")
    input_dim = X_train.shape[1]
    nn_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)), # Explicit Input layer
        tf.keras.layers.Dense(64, activation='relu'), # More nodes
        tf.keras.layers.Dropout(0.2), # Dropout for regularization
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') # Binary classification output
    ])
    nn_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    print(nn_model.summary())

    # Optional: Split training data for validation during NN training
    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = nn_model.fit(X_train_nn, y_train_nn,
                         epochs=30, # Train for more epochs with early stopping
                         batch_size=128,
                         validation_data=(X_val_nn, y_val_nn),
                         callbacks=[early_stopping],
                         verbose=1) # Set verbose=1 or 2 to see progress

    models['Neural Network'] = nn_model
    results['Neural Network'] = evaluate_model('Neural Network', nn_model, X_test, y_test)

    # --- Save the Best Model ---
    # Determine best model based on F1-score (good balance for imbalanced data)
    best_model_name = max(results, key=lambda k: results[k][1] if k != 'Neural Network' else results[k][1]) # Check F1 (index 1)

    print(f"\nBest performing model based on F1 Score: {best_model_name}")

    if best_model_name != 'Neural Network':
        print(f"Saving {best_model_name} model to {MODEL_PATH}...")
        joblib.dump(models[best_model_name], MODEL_PATH)
    else:
        print("Best model is Neural Network. Note: Saving NN model separately.")
        # Optionally, save the NN model as the primary if it's best
        # joblib.dump(models[best_model_name], MODEL_PATH) # This won't work well for Keras models

    # Always save the NN model anyway
    print(f"Saving Neural Network model to {NN_MODEL_PATH}...")
    models['Neural Network'].save(NN_MODEL_PATH)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_processed_data()
    train_models(X_train, y_train, X_test, y_test)
    print("\nModel training script finished.")