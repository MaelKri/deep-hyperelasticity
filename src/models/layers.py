import torch
import torch.nn as nn
import torch.nn.functional as F


class InvariantsLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, F):
        C = torch.matmul(F.transpose(1, 2), F)
        I1 = torch.diagonal(C, dim1=1, dim2=2).sum(dim=1)

        C_squared = torch.matmul(C, C)
        trace_C2 = torch.diagonal(C_squared, dim1=1, dim2=2).sum(dim=1)
        
        I2 = 0.5 * (I1**2 - trace_C2)

        return torch.stack([I1, I2], dim=1)
    
class PositiveLinearOld(nn.Module):
    def __init__(self, in_features, out_features, input_dim):
        super().__init__()
        self.weight_z = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_x = nn.Parameter(torch.Tensor(out_features, input_dim))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_z, a=5**0.5)
        nn.init.kaiming_uniform_(self.weight_x, a=5**0.5)
        
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_z)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, z_prev, x):
        w_z_positive = F.softplus(self.weight_z)
        
        out = F.linear(z_prev, w_z_positive) + F.linear(x, self.weight_x, self.bias)
        
        return out
    
class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, input_dim):
        super().__init__()
        self.weight_z = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_x = nn.Parameter(torch.Tensor(out_features, input_dim))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_x, a=5**0.5)
        nn.init.uniform_(self.weight_z, -3.0, -2.0)
        
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_z)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, z_prev, x):
        w_z_positive = F.softplus(self.weight_z)
        
        out = F.linear(z_prev, w_z_positive) + F.linear(x, self.weight_x, self.bias)
        
        return out