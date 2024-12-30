'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: ecg_train_securetl_linkability_model.py
- Used to train a model with ECG data, with the proposed Secure Triplet Loss with linkability
  based on either Kullback-Leibler Divergence (KLD) or simple statistics (SL).

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import os
import torch
import numpy as np
import pickle as pk
from models import SecureModel, SecureECGNetwork
from losses import SecureTripletLossKLD, SecureTripletLossSL
from dataset import SecureECGDataset
from trainer import train_secure_triplet_model
from torch.utils.data import DataLoader


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

SAVE_MODEL = 'model_name'
TRAIN_DATA = 'ecg_train_data.pickle' # Insert path

LEARN_RATE = 1e-4  # learning rate
REG = 0.001           # L2 regularization hyperparameter

N_EPOCHS = 100
BATCH_SIZE = 16
VALID_SPLIT = .2 

GAMMA = 0.9        # Set the gamma parameter here

# Choose one of the loss functions below:
loss = SecureTripletLossKLD(margin=1.0, gamma=GAMMA)
#loss = SecureTripletLossSL(margin=1.0, gamma=GAMMA)

print('Training model: ' + SAVE_MODEL)

# Preparing and dividing the dataset
trainset = SecureECGDataset(TRAIN_DATA)

dataset_size = len(trainset)  # number of samples in training + validation sets
indices = list(range(dataset_size))
split = int(np.floor(VALID_SPLIT * dataset_size))  # number of samples in validation set
np.random.seed(42)
np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, sampler=train_sampler)
valid_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, sampler=valid_sampler)

# Creating the network and the model
network = SecureECGNetwork().to(DEVICE)
model = SecureModel(network)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=REG)

# Training the model
train_hist, valid_hist = train_secure_triplet_model(model, loss, optimizer, train_loader, N_EPOCHS, BATCH_SIZE, DEVICE, patience=25, valid_loader=valid_loader, filename=SAVE_MODEL)


