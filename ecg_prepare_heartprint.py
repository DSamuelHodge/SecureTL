import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def read_ecg(heartid_path, session, id, record):
    file_path = os.path.join(heartid_path, f"Session-{session}", id)
    all_files = [f for f in os.listdir(file_path) if f.endswith('.txt')]
    num_file = len(all_files)

    if record >= num_file:
        print('Invalid record request')
        return None, None

    file_name = os.path.join(file_path, all_files[record])
    
    with open(file_name, 'r') as fid:
        ecg_raw = np.array([float(line.strip()) for line in fid.readlines()])
    
    lbl = int(id)  # Using the user ID as the class label
    
    return ecg_raw, lbl

def preprocess_signal(signal, target_length=1000):
    resampled = tf.signal.resample(signal, target_length)
    normalized = (resampled - tf.reduce_mean(resampled)) / tf.math.reduce_std(resampled)
    return normalized

def prepare_heartprint_data(root_path, sessions, test_size=0.2):
    all_signals = []
    all_labels = []

    for session in sessions:
        for id in range(1, 169):  # Assuming 168 subjects
            id_str = f"{id:03d}"
            for record in range(10):  # Assuming 10 records per subject per session
                signal, label = read_ecg(root_path, session, id_str, record)
                if signal is not None:
                    processed_signal = preprocess_signal(signal)
                    all_signals.append(processed_signal)
                    all_labels.append(label)

    X = np.array(all_signals)
    y = np.array(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

    return X_train, y_train, X_test, y_test

# Usage example
if __name__ == "__main__":
    root_path = '/path/to/heartprint'
    sessions = ['1', '2', '3R', '3L']
    X_train, y_train, X_test, y_test = prepare_heartprint_data(root_path, sessions)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
