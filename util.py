import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
import numpy as np


# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='data/', xd=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, target_key=None, target_graph=None):
        self.root = root
        super(DTADataset, self).__init__(root=root, transform= transform, pre_transform=pre_transform)
        self.process(xd, target_key, y)
        
    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y):
        assert (len(xd) == len(target_key) and len(xd) == len(y)), 'The three lists must be the same length!'
        data_list_mol = []
        data_list_pro = []
        data_dict_mol = {}
        data_dict_pro = {}
        data_len = len(xd)
        for i in range(data_len):
            smiles = xd[i]
            tar_key = target_key[i]
            labels = y[i]
            if smiles not in data_dict_mol.keys():
            # convert SMILES to molecular representation using rdkit
            # print(np.array(features).shape, np.array(edge_index).shape)
            # print(target_features.shape, target_edge_index.shape)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
                drug = torch.load(f'data/drug_graphs/{smiles}.pt')
                c_size = drug.c_size
                features = drug.features
                edge_index = drug.edge_index
                GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
                GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
                data_list_mol.append(GCNData_mol)
                data_dict_mol[smiles] = GCNData_mol
            else:
                data_list_mol.append(data_dict_mol[smiles])
            if tar_key not in data_dict_pro.keys():
                prot = torch.load(f'pdbs/pt/{tar_key}.pt')
                c_size = prot.x.shape[0]
                features = prot.x
                edge_index = prot.edge_index
                GCNData_pro = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
                GCNData_pro.__setitem__('target_size', torch.LongTensor([c_size]))
            # print(GCNData.target.size(), GCNData.target_edge_index.size(), GCNData.target_x.size())
                data_list_pro.append(GCNData_pro)
                data_dict_pro[tar_key] = GCNData_pro
            else:
                data_list_pro.append(data_dict_pro[tar_key])
        self.data_mol = data_list_mol
        self.data_pro = data_list_pro

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 10
    TRAIN_BATCH_SIZE = 512
    loss_fn = torch.nn.MSELoss()
    ll = 0
    tot = int(len(train_loader.dataset)/TRAIN_BATCH_SIZE)
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_pro)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
        ll = ll +  loss.item()
    return ll/tot

# predict
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def normalize_edge_index(edge_index):
    """Normalize edge index to shape (2, num_edges)"""
    if edge_index.shape[0] == 2:
        return edge_index
    elif edge_index.shape[1] == 2:
        return edge_index.t().contiguous()
    else:
        raise ValueError(f"Unexpected edge index shape: {edge_index.shape}")

def pad_node_features(tensor, target_size, feature_size):
    """Pad node feature tensor to target size."""
    padding_size = target_size - tensor.size(0)
    if padding_size > 0:
        padding = torch.zeros((padding_size, feature_size), device=tensor.device)
        return torch.cat([tensor, padding], dim=0)
    return tensor

def pad_edge_index(edge_index, num_nodes, target_size):
    """Pad edge index tensor to handle different graph sizes."""
    # First normalize the edge index to shape (2, num_edges)
    edge_index = normalize_edge_index(edge_index)
    
    if edge_index.numel() == 0:
        # If no edges, create self-loops for all nodes
        indices = torch.arange(target_size, device=edge_index.device)
        return torch.stack([indices, indices], dim=0)
    
    # Add self-loops for padding nodes
    padding_size = target_size - num_nodes
    if padding_size > 0:
        # Create self-loops for padding nodes
        padding_indices = torch.arange(num_nodes, target_size, device=edge_index.device)
        padding_self_loops = torch.stack([padding_indices, padding_indices], dim=0)
        
        # Combine original edges with padding self-loops
        edge_index = torch.cat([edge_index, padding_self_loops], dim=1)
    
    # Ensure no indices exceed target_size
    edge_index = torch.clamp(edge_index, 0, target_size - 1)
    
    return edge_index

def prepare_graph_batch(data_list, is_molecular=True):
    """Prepare a batch of graphs with proper padding."""
    # Find maximum sizes
    max_nodes = max(data.x.size(0) for data in data_list)
    feature_size = data_list[0].x.size(1)
    
    processed_graphs = []
    
    for data in data_list:
        num_nodes = data.x.size(0)
        
        # Create a new Data object
        new_data = DATA.Data()
        
        # Pad node features
        new_data.x = pad_node_features(data.x, max_nodes, feature_size)
        
        try:
            # Handle and pad edge indices
            if hasattr(data, 'edge_index'):
                new_data.edge_index = pad_edge_index(data.edge_index, num_nodes, max_nodes)
            else:
                # If no edge_index, create self-loops
                indices = torch.arange(max_nodes, device=data.x.device)
                new_data.edge_index = torch.stack([indices, indices], dim=0)
        
            # Copy other attributes
            new_data.y = data.y
            
            # Copy size attribute based on graph type
            if is_molecular:
                if hasattr(data, 'c_size'):
                    new_data.c_size = data.c_size
            else:
                if hasattr(data, 'target_size'):
                    new_data.target_size = data.target_size
                    
            processed_graphs.append(new_data)
            
        except Exception as e:
            print(f"Error processing graph: {e}")
            print(f"Graph info - Nodes: {num_nodes}, Features: {feature_size}")
            print(f"Edge index shape: {data.edge_index.shape if hasattr(data, 'edge_index') else 'No edge_index'}")
            raise e
    
    return processed_graphs

def collate(data_list):
    """Custom collate function for batching graphs with different sizes."""
    try:
        # Separate molecular and protein graphs
        mol_graphs = [data[0] for data in data_list]
        pro_graphs = [data[1] for data in data_list]
        
        # Print debug information for first graph in batch
        # if len(data_list) > 0:
        #     print("\nProcessing first graph in batch:")
        #     print(f"Molecular graph - Nodes: {mol_graphs[0].x.shape[0]}, " 
        #           f"Edge index: {mol_graphs[0].edge_index.shape}")
        #     print(f"Protein graph - Nodes: {pro_graphs[0].x.shape[0]}, "
        #           f"Edge index: {pro_graphs[0].edge_index.shape}")
        
        # Process molecular and protein graphs separately
        processed_mol_graphs = prepare_graph_batch(mol_graphs, is_molecular=True)
        processed_pro_graphs = prepare_graph_batch(pro_graphs, is_molecular=False)
        
        # Create batches
        batch_mol = Batch.from_data_list(processed_mol_graphs)
        batch_pro = Batch.from_data_list(processed_pro_graphs)
        
        return batch_mol, batch_pro
        
    except Exception as e:
        print(f"Error in collate function: {e}")
        print(f"Number of graphs in batch: {len(data_list)}")
        if len(data_list) > 0:
            print(f"First graph mol shape: {data_list[0][0].x.shape}")
            print(f"First graph pro shape: {data_list[0][1].x.shape}")
            print(f"First graph mol edge_index: {data_list[0][0].edge_index.shape}")
            print(f"First graph pro edge_index: {data_list[0][1].edge_index.shape}")
        raise e

    # batchA = Batch.from_data_list([data[0] for data in data_list])
    # batchB = Batch.from_data_list([data[1] for data in data_list])
    # return batchA, batchB

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
