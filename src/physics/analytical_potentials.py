import torch
import torch.nn as nn
import numpy as np
from src.physics.invariants import compute_deformations_measures

class HyperelasticPotential(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, F, t=None):
        raise NotImplementedError("La methode forward doit être implémentée par la sous classe.")
    def compute_stress(self, F, t=None):
        F_in = F.detach().clone().requires_grad_(True)
        psi=self.forward(F_in, t)

        P_raw = torch.autograd.grad(
            outputs=psi.sum(),
            inputs=F_in,
            create_graph=True,
            retain_graph=True,
        )[0]
        F_T = F_in.transpose(-2, -1)
        sigma_raw = P_raw @ F_T
        if F_in.dim() == 3:
            p = sigma_raw[:, 1, 1].view(-1, 1, 1)
            # Inversion de F pour le terme correctif (Batch-safe)
            F_inv_T = torch.inverse(F_in).transpose(-2, -1)
        else:
            p = sigma_raw[1, 1]
            F_inv_T = torch.inverse(F_in).t()
        P_corrected = P_raw - p * F_inv_T
        return(P_corrected)

class NeoHookeanPotential(HyperelasticPotential):
    def __init__(self, mu=0.5, lam=100.0):
        super().__init__()
        # self.mu_default=mu_default
        # self.lambda_default=lambda_default
        self.register_buffer('mu', torch.tensor([float(mu)]))
        self.register_buffer('lam', torch.tensor([float(lam)]))
        # self.mu = mu
        # self.lam=lam

    def get_lame_params(self, t):
        if t is None:
            return self.mu, self.lam
        
        t_safe = t.view(-1, 1)
        mu = 0.5 + 2.0 * t_safe
        kappa = 100.0
        lam = kappa - (2.0/3.0) * mu
        return mu, lam
    
    def forward(self, F, t=None):
        invs=compute_deformations_measures(F)
        I1=invs["I1"]
        I3=invs["I3"]
        mu, lam = self.get_lame_params(t)
        # Psi = (mu/2)*(I1 - 3 - 2*ln(sqrt(I3))) + (lambda/2)*(sqrt(I3) - 1)^2
        epsilon = 1e-8
        term1 = (mu / 2.0) * (I1 - 3.0 - 2.0*torch.log(torch.sqrt(I3 + epsilon)))
        term2 = (lam / 2.0) * (torch.sqrt(I3 + epsilon) - 1.0)**2
        psi = term1 + term2
        return psi

class GornetDesmoratPotential(HyperelasticPotential):
    def __init__(self, h1 = 0.1, h2 = 0.05, h3 = 0.05):
        super().__init__()
        self.register_buffer('h1', torch.tensor([float(h1)]))
        self.register_buffer('h2', torch.tensor([float(h2)]))
        self.register_buffer('h3', torch.tensor([float(h3)]))

    def forward(self, F, t=None):
        invs=compute_deformations_measures(F)
        I1=invs["I1"]
        I2=invs["I2"]

        epsilon = 1e-8
        W1 = self.h1/self.h3 * (torch.exp(self.h3*(I1-3))-1)
        W2 = 6.0*self.h2*(torch.sqrt(I2)-np.sqrt(3.0))
        W=W1+W2
        return(W)



