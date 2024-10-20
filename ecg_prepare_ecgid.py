'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)
File: ecg_prepare_ecgid.py
- Prepares the ECG-ID Database for use in Secure Triplet Loss training and experiments.

Adapted for the ECG-ID Database:
- 310 ECG recordings from 90 persons
- 20-second recordings, 500 Hz sampling rate
- Both raw and filtered signals available
- 10 annotated beats per recording
- Age and gender information in header files
'''

import os
import numpy as np
import wfdb
import pickle
import aux_functions as af
from sklearn.model_selection import train_test_split

N_TRAIN = 100000   # Number of triplets for the train set
N_TEST = 10000     # Number of triplets for the test set
ECGID_PATH = /content/physionet.org/files/ecgiddb/1.0.0 #/path/to/ecgid/database
SAVE_TRAIN = 'ecg_train_data.pickle'
SAVE_TEST = 'ecg_test_data.pickle'
fs = 500.0  # Sampling frequency of data

def load_ecg_data(record_path):
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')
    
    # Use the filtered signal (index 1)
    signal = record.p_signal[:, 1]
    
    # Extract R-peak locations
    r_peaks = ann.sample

    # Extract age and gender from the header file
    with open(f"{record_path}.hea", 'r') as f:
        header = f.read()
    age = int(header.split('Age: ')[1].split('\n')[0])
    gender = header.split('Sex: ')[1].split('\n')[0]

    return signal, r_peaks, age, gender

def extract_beats(signal, r_peaks, window_size=250):
    beats = []
    for peak in r_peaks:
        start = max(0, peak - window_size // 2)
        end = min(len(signal), peak + window_size // 2)
        beat = signal[start:end]
        if len(beat) < window_size:
            beat = np.pad(beat, (0, window_size - len(beat)))
        beats.append(beat)
    return np.array(beats)

def extract_all_data(database_path):
    X = []
    y = []
    for root, _, files in os.walk(database_path):
        for file in files:
            if file.endswith('.dat'):
                record_path = os.path.join(root, file[:-4])
                signal, r_peaks, age, gender = load_ecg_data(record_path)
                beats = extract_beats(signal, r_peaks)
                X.extend(beats)
                y.extend([f"{file[:3]}_{age}_{gender}"] * len(beats))
    return np.array(X), np.array(y)

# Extract all data
print("Extracting data from ECG-ID database...")
X, y = extract_all_data(ECGID_PATH)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Prepare data for DNN
print("Preparing data for deep neural network...")
X_train, y_train = af.prepare_for_dnn(X_train, y_train)
X_test, y_test = af.prepare_for_dnn(X_test, y_test)

# Generate triplets
print("Generating triplets...")
train_triplets = af.generate_triplets(X_train, y_train, N=N_TRAIN)
test_triplets = af.generate_triplets(X_test, y_test, N=N_TEST)

# Save prepared data
print("Saving prepared data...")
with open(SAVE_TRAIN, 'wb') as handle:
    pickle.dump(train_triplets, handle)
with open(SAVE_TEST, 'wb') as handle:
    pickle.dump(test_triplets, handle)

print(f"Train triplets shape: {train_triplets.shape}")
print(f"Test triplets shape: {test_triplets.shape}")
print("Data preparation completed and saved.")
