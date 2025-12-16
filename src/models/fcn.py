import torch
import torch.nn as nn
import torch.functional as F

from src.models.layers import InvariantsLayer

class FCN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, n_layers=3, use_invariants_layer=False):
        super().__init__()
        self.use_inv_layer=use_invariants_layer
        if self.use_inv_layer:
            self.inv_layer=InvariantsLayer()
            self.internal_input_dim = 2
        else:
            self.internal_input_dim=input_dim
        
        layers=[]
        layers.append(nn.Linear(in_features=self.internal_input_dim, out_features=hidden_dim))
        layers.append(nn.Softplus())

        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus())

        layers.append(nn.Linear(hidden_dim, 1))

        self.net=nn.Sequential(*layers)

    def forward(self, x):
        if self.use_inv_layer:
            x = self.inv_layer(x)
        return(self.net(x))
