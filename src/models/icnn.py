import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import PositiveLinear, InvariantsLayer


class ICNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, n_layers=3, use_invariants_layer=False):
        super().__init__()
        self.use_inv_layer = use_invariants_layer

        if self.use_inv_layer:
            self.inv_layer = InvariantsLayer()
            self.internal_input_dim = 2
        else:
            self.internal_input_dim = input_dim

        self.layer_0 = nn.Linear(self.internal_input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                PositiveLinear(in_features=hidden_dim, 
                               out_features=hidden_dim, 
                               input_dim=self.internal_input_dim)
            )
        self.layer_out = PositiveLinear(in_features=hidden_dim, 
                                        out_features=1, 
                                        input_dim=self.internal_input_dim)
        
    def forward(self, x):
        if self.use_inv_layer:
            x = self.inv_layer(x)
        input_invariants = x
        z = self.layer_0(input_invariants)
        z = F.softplus(z)
        for layer in self.layers:
            z = layer(z, input_invariants)
            z = F.softplus(z)
        psi = self.layer_out(z, input_invariants)
        return psi
