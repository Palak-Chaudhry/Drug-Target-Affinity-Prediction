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

def create_dataset(x):
    drug_id, protein_id, Y = np.asarray(x['Drug_ID']), np.asarray(x['Target_ID']), np.asarray(x['Y'])
    tot = len(x)
    print(tot)
    drug_id_train, protein_id_train, Y_train = drug_id[:int(tot*0.7)], protein_id[:int(tot*0.7)], Y[:int(tot*0.7)]
    drug_id_valid, protein_id_valid, Y_valid = drug_id[int(tot*0.7):], protein_id[int(tot*0.7):], Y[int(tot*0.7):]
    train_dataset = DTADataset(xd=drug_id_train, target_key=protein_id_train,y=Y_train)
    print("train_dataset loaded")
    valid_dataset = DTADataset(xd=drug_id_valid,target_key=protein_id_valid, y=Y_valid)
    print("valid_dataset loaded")
    return train_dataset, valid_dataset
    
# Usage
if __name__ == "__main__":
    data = DTI(name = 'DAVIS')
    data.convert_to_log(form = 'binding')
    x = data.get_data()
    train_dataset, valid_dataset = create_dataset(x)
    print(len(train_dataset),len(valid_dataset))
