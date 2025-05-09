import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, MLP

class GIN_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            mlp = MLP([in_channels if i==0 else hidden_channels,
                       hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        return self.classifier(x)


class GCN_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch  = in_channels if i==0 else hidden_channels
            out_ch = out_channels  if i==num_layers-1 else hidden_channels
            self.convs.append(GCNConv(in_ch, out_ch))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        # last layer, no activation
        return self.convs[-1](x, edge_index)


class GATN_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, heads=8, dropout=0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i==0 else hidden_channels * heads
            if i < num_layers-1:
                self.convs.append(
                    GATConv(in_ch, hidden_channels, heads=heads, dropout=dropout)
                )
            else:
                # final layer: single head, no concat
                self.convs.append(
                    GATConv(in_ch, out_channels, heads=1, concat=False, dropout=dropout)
                )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = self.dropout(x)
        return self.convs[-1](x, edge_index)