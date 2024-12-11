import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from tdc.multi_pred import DTI
import torch

class MoleculeGraph:
    def __init__(self, smile):
        self.smile = smile
        self.mol = Chem.MolFromSmiles(smile)
        self.c_size = self.mol.GetNumAtoms()
        self.features = self._compute_features()
        self.edge_index = self._compute_edge_index()

    def _compute_features(self):
        features = []
        for atom in self.mol.GetAtoms():
            feature = self.atom_features(atom)
            features.append(feature / sum(feature))
        return features

    def _compute_edge_index(self):
        edges = []
        for bond in self.mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        
        g = nx.Graph(edges).to_directed()
        mol_adj = np.zeros((self.c_size, self.c_size))
        
        for e1, e2 in g.edges:
            mol_adj[e1, e2] = 1
        
        mol_adj += np.eye(mol_adj.shape[0])
        index_row, index_col = np.where(mol_adj >= 0.5)
        
        edge_index = [[i, j] for i, j in zip(index_row, index_col)]
        
        return edge_index

    @staticmethod
    def atom_features(atom):
        # 44 +11 +11 +11 +1
        return np.array(MoleculeGraph.one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                           'Cr','Pt','Hg','Pb','X']) +
                    MoleculeGraph.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    MoleculeGraph.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    MoleculeGraph.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7,
                                                                                    8,9,10]) +
                    [atom.GetIsAromatic()])

    @staticmethod
    def one_of_k_encoding(x, allowable_set):
        if x not in allowable_set:
            raise Exception('input {0} not in allowable set {1}:'.format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    @staticmethod
    def one_of_k_encoding_unk(x, allowable_set):
        '''Maps inputs not in the allowable set to the last element.'''
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))


# Usage
if __name__ == "__main__":

    data = DTI(name = 'DAVIS')
    data.convert_to_log(form = 'binding')
    x = data.get_data()
    for i in range(len(x)):
        mol_graph = MoleculeGraph(x["Drug"][i])
        id = x["Drug_ID"][i]
        # print(mol_graph)
        # print("Number of atoms:", mol_graph.c_size)
        # print("Features:", mol_graph.features)
        # print("Edge index:", mol_graph.edge_index)
        torch.save(mol_graph, f'./data/drug_graphs/{id}.pt')
        print(f"{mol_graph} graph is saved")


# # Example usage:
# smile = "CCO"
# mol_graph = MoleculeGraph(smile)
# print("Number of atoms:", mol_graph.c_size)
# print("Features:", mol_graph.features)
# print("Edge index:", mol_graph.edge_index)




# # mol atom feature for mol graph
# def atom_features(atom):
#     # 44 +11 +11 +11 +1
#     return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
#                                           ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
#                                            'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
#                                            'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
#                                            'Pt', 'Hg', 'Pb', 'X']) +
#                     one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
#                     one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
#                     one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
#                     [atom.GetIsAromatic()])

# # one ont encoding
# def one_of_k_encoding(x, allowable_set):
#     if x not in allowable_set:
#         # print(x)
#         raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
#     return list(map(lambda s: x == s, allowable_set))

# def one_of_k_encoding_unk(x, allowable_set):
#     '''Maps inputs not in the allowable set to the last element.'''
#     if x not in allowable_set:
#         x = allowable_set[-1]
#     return list(map(lambda s: x == s, allowable_set))

# # mol smile to mol graph edge index
# def smile_to_graph(smile):
#     mol = Chem.MolToSmiles(Chem.MolFromSmiles(smile), isomericSmiles=True)

#     c_size = mol.GetNumAtoms()

#     features = []
#     for atom in mol.GetAtoms():
#         feature = atom_features(atom)
#         features.append(feature / sum(feature))

#     edges = []
#     for bond in mol.GetBonds():
#         edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
#     g = nx.Graph(edges).to_directed()
#     edge_index = []
#     mol_adj = np.zeros((c_size, c_size))
#     for e1, e2 in g.edges:
#         mol_adj[e1, e2] = 1
#         # edge_index.append([e1, e2])
#     mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
#     index_row, index_col = np.where(mol_adj >= 0.5)
#     for i, j in zip(index_row, index_col):
#         edge_index.append([i, j])
#     # print('smile_to_graph')
#     # print(np.array(features).shape)
#     return c_size, features, edge_index


# # smile_graph = {}
# # for smile in drugs:
# #     g = smile_to_graph(smile)
# #     smile_graph[smile] = g