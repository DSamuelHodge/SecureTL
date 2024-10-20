'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)
File: ecg_prepare_ecgid.py
- Prepares the ECG-ID Database for use in Secure Triplet Loss training and experiments.
- Uses aux_functions.py for data preparation and triplet generation.

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

N_TRAIN = 100000   # Number of triplets for the train set
N_TEST = 10000     # Number of triplets for the test set
ECGID_PATH = '/path/to/ecgid/database'
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

def extract_data(database_path, subject_range, fs=500.0):
    data = {'X_anchors': [], 'y_anchors': [], 'X_remaining': [], 'y_remaining': []}
    
    for subject in subject_range:
        subject_files = [f for f in os.listdir(database_path) if f.startswith(f"{subject:03d}_")]
        if not subject_files:
            continue
        
        # Use the first recording as anchor
        anchor_file = subject_files[0]
        anchor_path = os.path.join(database_path, anchor_file[:-4])
        signal, r_peaks, age, gender = load_ecg_data(anchor_path)
        data['X_anchors'].append(signal)
        data['y_anchors'].append(f"{subject:03d}_{age}_{gender}")
        
        # Use remaining recordings as additional samples
        for file in subject_files[1:]:
            record_path = os.path.join(database_path, file[:-4])
            signal, r_peaks, age, gender = load_ecg_data(record_path)
            data['X_remaining'].append(signal)
            data['y_remaining'].append(f"{subject:03d}_{age}_{gender}")
    
    return data

# Dividing subjects for training and for testing
print("Extracting data from ECG-ID database...")
train_data = af.extract_data(ECGID_PATH, range(1, 73), fs=500.0)  # 72 subjects for training
test_data = af.extract_data(ECGID_PATH, range(73, 91), fs=500.0)  # 18 subjects for testing

# Preparing data for a deep neural network
print("Preparing data for deep neural network...")
X_train_a, y_train_a = af.prepare_for_dnn(train_data['X_anchors'], train_data['y_anchors'])
X_train_r, y_train_r = af.prepare_for_dnn(train_data['X_remaining'], train_data['y_remaining'])
X_test_a, y_test_a = af.prepare_for_dnn(test_data['X_anchors'], test_data['y_anchors'])
X_test_r, y_test_r = af.prepare_for_dnn(test_data['X_remaining'], test_data['y_remaining'])

# Generating triplets
print("Generating triplets...")
train_triplets = af.generate_triplets(X_train_a, y_train_a, X_train_r, y_train_r, N=N_TRAIN)
test_triplets = af.generate_triplets(X_test_a, y_test_a, X_test_r, y_test_r, N=N_TEST)

# Saving prepared data
print("Saving prepared data...")
with open(SAVE_TRAIN, 'wb') as handle:
    pickle.dump(train_triplets, handle)
with open(SAVE_TEST, 'wb') as handle:
    pickle.dump(test_triplets, handle)

print(f"Train triplets shape: {train_triplets[0].shape}")
print(f"Test triplets shape: {test_triplets[0].shape}")
print("Data preparation completed and saved.")
