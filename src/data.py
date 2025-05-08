import torch
import os
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader

def load_dataset(name, path='../data', train_size=None, test_size=None, batch_size=32):
    name = name.upper()
    
    if name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
        dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name)
        torch.manual_seed(1712)
        dataset = dataset.shuffle()
        if train_size and test_size:
            train_dataset = dataset[:train_size]
            test_dataset = dataset[train_size:train_size + test_size]
        else:
            train_dataset = dataset[:int(0.8 * len(dataset))]
            test_dataset = dataset[int(0.8 * len(dataset)):]
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        task_type = 'graph'
    elif name in ['CORA', 'CITESEER', 'PUBMED']:
        dataset = Planetoid(root=os.path.join(path, 'Planetoid'), name=name)
        data = dataset[0]  # only one graph
        train_loader = test_loader = data  # use the same for node-level
        task_type = 'node'
    else:
        raise ValueError(f"Dataset '{name}' not supported.")
    
    return dataset, train_loader, test_loader, task_type
