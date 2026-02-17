import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicsInputLayer(nn.Module):
    def __init__(self, norm_1=None, norm_2=None):
        super().__init__()
        self.norm_1 = norm_1
        self.norm_2 = norm_2
        
    def forward(self, F_tensor):
        C = torch.bmm(F_tensor.transpose(1, 2), F_tensor)
        I1 = C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2]

        C2 = torch.bmm(C, C)
        tr_C2 = C2[:, 0, 0] + C2[:, 1, 1] + C2[:, 2, 2]
        I2 = 0.5 * (I1**2 - tr_C2)

        f1 = (I1 - 3.0).view(-1, 1)
        f2 = (torch.sqrt(I2 + 1e-8) - np.sqrt(3.0)).view(-1, 1)

        if self.norm_1 and self.norm_2:
            f1 = self.norm_1.normalize(f1)
            f2 = self.norm_2.normalize(f2)
        return f1, f2

class ICNN(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=3, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.fc_start = nn.Linear(n_inputs, n_hidden)
        self.wz_layers = nn.ModuleList()
        self.wx_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.wz_layers.append(nn.Linear(n_hidden, n_hidden))
            self.wx_layers.append(nn.Linear(n_inputs, n_hidden, bias=False))
        
        self.fc_out = nn.Linear(n_hidden, 1)
        self.act = nn.Softplus()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=-4.0, std=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        w_start_pos = F.softplus(self.fc_start.weight)
        z = self.act(F.linear(x, w_start_pos, self.fc_start.bias))
        
        for wz_layer, wx_layer in zip(self.wz_layers, self.wx_layers):
            w_z_pos = F.softplus(wz_layer.weight)
            w_x_pos = F.softplus(wx_layer.weight)

            z_term = F.linear(z, w_z_pos, wz_layer.bias)
            x_term = F.linear(x, w_x_pos)

            z = self.act(z_term + x_term)
        
        w_out_pos = F.softplus(self.fc_out.weight)
        y = F.linear(z, w_out_pos, self.fc_out.bias)
        return y

class GornetModel(nn.Module):
    def __init__(self, input_layer, n_hidden=64, n_layers=3):
        super().__init__()
        self.input_layer = input_layer
        self.net_exponent = ICNN(n_inputs=1, n_hidden=n_hidden, n_layers=n_layers)
        self.net_w2 = ICNN(n_inputs=1, n_hidden=n_hidden, n_layers=n_layers)

        # self.beta=nn.Parameter(torch.tensor(5.0))

    def forward(self, F_tensor):
        f1, f2 = self.input_layer(F_tensor)

        # z1=self.net_exponent(f1)
        # z1=torch.clamp(z1, max=20.0)
        # beta_pos=torch.nn.functional.softplus(self.beta)
        # W1=beta_pos*(torch.exp(z1)-1.0)

        W1 = self.net_exponent(f1)
        W2_brut = self.net_w2(f2)
        zeros = torch.zeros_like(f2)
        W2_repos = self.net_w2(zeros)
        W2 = W2_brut - W2_repos
        return W1 + W2

def predict_stress(model, F_batch):
    F_batch.requires_grad_(True)
    psi_pred = model(F_batch)
    grads = torch.autograd.grad(psi_pred.sum(), F_batch, create_graph=True)[0]
    
    sigma_dev = torch.bmm(grads, F_batch.transpose(1, 2))
    p_pred = sigma_dev[:, 2, 2].view(-1, 1, 1)
    F_inv_T = torch.linalg.inv(F_batch).transpose(1, 2)
    P_pred = grads - p_pred * F_inv_T
    
    return P_pred, psi_pred