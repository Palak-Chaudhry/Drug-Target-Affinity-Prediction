import sys, os
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from gnn import GNNNet
from gat import GAT
from util import *
import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
import torch
from util import *
from drug_to_graph import *
from protein_to_graph import *
from tdc.multi_pred import DTI
import matplotlib.pyplot as plt



def get_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci
def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse 

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 128
LR = 0.0001
NUM_EPOCHS = 20

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Main program: iterate over different datasets
result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print(device)
model = GAT()
model.to(device)
model_st = GAT.__name__
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = DTI(name = 'DAVIS')
data.convert_to_log(form = 'binding')
x = data.get_data()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train_data, valid_data = create_dataset(x)

# Debug
sample_batch = next(iter(train_data))
print("\nSample data structure:")
print(f"Molecular graph:")
print(f"- Node features shape: {sample_batch[0].x.shape}")
print(f"- Edge index shape: {sample_batch[0].edge_index.shape}")
print(f"- Edge index type: {type(sample_batch[0].edge_index)}")
print(f"\nProtein graph:")
print(f"- Node features shape: {sample_batch[1].x.shape}")
print(f"- Edge index shape: {sample_batch[1].edge_index.shape}")
print(f"- Edge index type: {type(sample_batch[1].edge_index)}")

train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                            collate_fn=collate)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                            collate_fn=collate)

best_mse = 1000
best_test_mse = 1000
best_ci = 0
best_epoch = -1
model_file_name = 'models/model_' + model_st + '_' + '.model'


# Initialize lists to store values
train_losses = []
val_mses = []
val_cis = [] 
for epoch in range(NUM_EPOCHS):
    ll = train(model, device, train_loader, optimizer, epoch + 1)
    print('train mse loss:', ll)
    print('predicting for valid data')
    G, P = predicting(model, device, valid_loader)
    val = get_mse(G, P)
    ci = get_ci(G,P)
    
    print('valid mse result:', val, best_mse)
    print('valid ci result:', ci, best_ci)
    if val < best_mse:
        best_mse = val
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_file_name)
        print('mse improved at epoch ', best_epoch, '; best_test_mse', best_mse, model_st)
    else:
        print('No improvement since epoch ', best_epoch, '; best_test_mse', best_mse, model_st)
    if ci > best_ci:
        best_ci = ci
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_file_name)
        print('ci improved at epoch ', best_epoch, '; best_test_ci', best_ci, model_st)
    else:
        print('No improvement since epoch ', best_epoch, '; best_test_ci', best_ci, model_st)
    # Append values to lists
    train_losses.append(ll)
    val_mses.append(val)
    val_cis.append(ci)


# Create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

# Plot train loss
ax1.plot(range(1, NUM_EPOCHS+1), train_losses, 'b-')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

# Plot validation MSE
ax2.plot(range(1, NUM_EPOCHS+1), val_mses, 'r-')
ax2.set_title('Validation MSE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE')

# Plot validation CI
ax3.plot(range(1, NUM_EPOCHS+1), val_cis, 'g-')
ax3.set_title('Validation CI')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('CI')

# Adjust layout
plt.tight_layout()

# Save the plots as a single PNG file
fig.savefig('plots_gat_attention.png')

# Close the figure to free up memory
plt.close(fig)
