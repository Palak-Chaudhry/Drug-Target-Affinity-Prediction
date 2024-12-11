import torch
import os
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from biopandas.pdb import PandasPdb
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

class ProteinGraphConverter:
    def __init__(self):
        self.pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
        self.pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
        self.pro_res_aromatic_table = ['F', 'W', 'Y']
        self.pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
        self.pro_res_acidic_charged_table = ['D', 'E']
        self.pro_res_basic_charged_table = ['H', 'K', 'R']

        self.res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                                 'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                                 'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
        self.res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                              'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                              'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
        self.res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                              'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                              'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
        self.res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                              'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                              'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
        self.res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                             'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                             'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
        self.res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                                          'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                                          'T': 13, 'V': 79, 'W': 84, 'Y': 49}
        self.res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                                          'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                                          'T': 13, 'V': 76, 'W': 97, 'Y': 63}

        self.normalize_tables()

    def normalize_tables(self):
        self.res_weight_table = self.dic_normalize(self.res_weight_table)
        self.res_pka_table = self.dic_normalize(self.res_pka_table)
        self.res_pkb_table = self.dic_normalize(self.res_pkb_table)
        self.res_pkx_table = self.dic_normalize(self.res_pkx_table)
        self.res_pl_table = self.dic_normalize(self.res_pl_table)
        self.res_hydrophobic_ph2_table = self.dic_normalize(self.res_hydrophobic_ph2_table)
        self.res_hydrophobic_ph7_table = self.dic_normalize(self.res_hydrophobic_ph7_table)

    @staticmethod
    def dic_normalize(dic):
        max_value = dic[max(dic, key=dic.get)]
        min_value = dic[min(dic, key=dic.get)]
        interval = float(max_value) - float(min_value)
        for key in dic.keys():
            dic[key] = (dic[key] - min_value) / interval
        dic['X'] = (max_value + min_value) / 2.0
        return dic

    @staticmethod
    def one_of_k_encoding(x, allowable_set):
        return list(map(lambda s: int(x == s), allowable_set))

    def residue_features(self, residue):
        pro_hot = self.one_of_k_encoding(residue, self.pro_res_table)
        res_property1 = [1 if residue in self.pro_res_aliphatic_table else 0,
                         1 if residue in self.pro_res_aromatic_table else 0,
                         1 if residue in self.pro_res_polar_neutral_table else 0,
                         1 if residue in self.pro_res_acidic_charged_table else 0,
                         1 if residue in self.pro_res_basic_charged_table else 0]
        res_property2 = [self.res_weight_table[residue], self.res_pka_table[residue],
                         self.res_pkb_table[residue], self.res_pkx_table[residue],
                         self.res_pl_table[residue], self.res_hydrophobic_ph2_table[residue],
                         self.res_hydrophobic_ph7_table[residue]]
        return np.array(pro_hot + res_property1 + res_property2)

    def pdb_to_graph(self, pdb_path, distance_threshold=7.0):
        atom_df = PandasPdb().read_pdb(pdb_path)
        atom_df = atom_df.df['ATOM']
        residue_df = atom_df.groupby('residue_number', as_index=False)[['x_coord', 'y_coord', 'z_coord', 'b_factor']].mean().sort_values('residue_number')
        coords = residue_df[['x_coord', 'y_coord', 'z_coord']].values
        distance_matrix = squareform(pdist(coords))
        adj = distance_matrix < distance_threshold
        edge_index, _ = dense_to_sparse(torch.tensor(adj.astype(float)))
        
        atom_features = []
        for _, row in atom_df.iterrows():
            residue = row['residue_name']
            if residue not in self.pro_res_table:
                residue = 'X'  # Use 'X' for unknown residues
            feature = self.residue_features(residue)
            atom_features.append(feature)

        atom_features = torch.tensor(np.array(atom_features), dtype=torch.float)
        edge_attr = torch.tensor(distance_matrix[adj], dtype=torch.float).view(-1, 1)

        graph = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr)
        return graph
    
    def save_graph(self, graph, output_path):
        torch.save(graph, output_path)
        print(f"Graph saved to {output_path}")

# Usage

pdb_directory = "./pdbs"
pdb_files = [os.path.join(pdb_directory, f) for f in os.listdir(pdb_directory) if f.endswith(".pdb")]
converter = ProteinGraphConverter()
def convert_pdb_list_to_graphs(pdb_file_list, config):
    graphs = []
    for pdb_file in pdb_file_list:
        graph = converter.pdb_to_graph(pdb_file, config)
        graphs.append(graph)

# output_dir = "./pdbs/pt/"
# for pdb_file in pdb_files:
#     graph = converter.pdb_to_graph(pdb_file)
#     graph_name = os.path.basename(pdb_file).replace(".pdb", ".pt")
#     graph_path = os.path.join(output_dir, graph_name)
#     converter.save_graph(graph, graph_path)