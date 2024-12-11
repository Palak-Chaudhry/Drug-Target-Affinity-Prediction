import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj


# GCN based model
class GNNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, output_dim=512, dropout=0.2):
        super(GNNNet, self).__init__()

        print('GNNNet Loaded')
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        #self.mol_conv4 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        #self.pro_conv4 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        # Batch normalization layers
        self.mol_bn1 = nn.BatchNorm1d(num_features_mol)
        self.mol_bn2 = nn.BatchNorm1d(num_features_mol * 2)
        self.mol_bn3 = nn.BatchNorm1d(num_features_mol * 4)
       # self.mol_bn4 = nn.BatchNorm1d(num_features_mol * 4)
        self.mol_bn_fc1 = nn.BatchNorm1d(1024)
        self.mol_bn_fc2 = nn.BatchNorm1d(output_dim)

        self.pro_bn1 = nn.BatchNorm1d(num_features_pro)
        self.pro_bn2 = nn.BatchNorm1d(num_features_pro * 2)
        self.pro_bn3 = nn.BatchNorm1d(num_features_pro * 4)
        #self.pro_bn4 = nn.BatchNorm1d(num_features_pro * 4)
        self.pro_bn_fc1 = nn.BatchNorm1d(1024)
        self.pro_bn_fc2 = nn.BatchNorm1d(output_dim)

        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.bn_fc2 = nn.BatchNorm1d(512)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.mol_bn1(x)
        x = self.relu(x)

        x = self.mol_conv2(x, mol_edge_index)
        x = self.mol_bn2(x)
        x = self.relu(x)

        x = self.mol_conv3(x, mol_edge_index)
        x = self.mol_bn3(x)
        x = self.relu(x)

        # x = self.mol_conv4(x, mol_edge_index)
        # x = self.mol_bn4(x)
        # x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        x = self.mol_fc_g1(x)
        x = self.mol_bn_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.mol_bn_fc2(x)
        x = self.dropout(x)

        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.pro_bn1(xt)
        xt = self.relu(xt)

        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.pro_bn2(xt)
        xt = self.relu(xt)

        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.pro_bn3(xt)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.pro_bn4(xt)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        xt = self.pro_fc_g1(xt)
        xt = self.pro_bn_fc1(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.pro_bn_fc2(xt)
        xt = self.dropout(xt)

        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.bn_fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.bn_fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
