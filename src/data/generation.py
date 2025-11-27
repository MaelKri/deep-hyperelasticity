import torch
import torch.nn as nn
import numpy as np
from src.physics.analytical_potentials import NeoHookeanPotential, GornetDesmoratPotential

class DeformationSampler:
    def __init__(self, device='cpu'):
        self.device=device

    def _get_compressibility_factor(self, stretch_ratio):
        return 1.0 / torch.sqrt(stretch_ratio)
    
    def sample_uniaxial(self, n_samples, stretch_min=0.0, stretch_max=4.0):
        lambdas = torch.empty(n_samples).uniform_(stretch_min,stretch_max).to(self.device)
        lat_contraction = 1.0 / torch.sqrt(lambdas)
        F = torch.zeros(n_samples, 3, 3, device=self.device)
        F[:, 0, 0] = lambdas
        F[:, 1, 1] = lat_contraction
        F[:, 2, 2] = lat_contraction
        return(F, lambdas)
    
    def sample_biaxial(self, n_samples, stretch_min=0.0, stretch_max=4.0):
        lambdas = torch.empty(n_samples).uniform_(stretch_min, stretch_max).to(self.device)
        z_contraction= 1.0/(lambdas**2)
        F = torch.zeros(n_samples, 3, 3, device=self.device)
        F[:, 0, 0] = lambdas
        F[:, 1, 1] = lambdas
        F[:, 2, 2] = z_contraction
        return(F,lambdas)

    def sample_shear(self, n_samples, gamma_min=-0.5, gamma_max=0.5):
        gammas = torch.empty(n_samples).uniform_(gamma_min, gamma_max).to(self.device)
        F = torch.eye(3, device=self.device).repeat(n_samples, 1, 1)
        F[:, 0, 1] = gammas
        return(F, gammas)

class SyntheticDatasetGenerator:
    def __init__(self, potential_model):
        self.model=potential_model
        self.sampler=DeformationSampler()

    def generate(self, n_samples_per_mode=1000):
        data_list=[]
        modes = ['uniaxial', 'biaxial', 'shear']
        for mode in modes:
            if mode == 'uniaxial':
                F, _ = self.sampler.sample_uniaxial(n_samples_per_mode)
            elif mode == 'biaxial':
                F, _ = self.sampler.sample_biaxial(n_samples_per_mode)
            elif mode == 'shear':
                F, _ = self.sampler.sample_shear(n_samples_per_mode)
            t=None
            P = self.model.compute_stress(F, t)
            data_list.append({
                'F': F.detach(),
                't': None,
                'P': P.detach(),
                'mode': mode
            })
        print(f"Dataset généré avec {len(modes) * n_samples_per_mode} échantillons.")
        return data_list
    def save_dataset(self, data_list, filename="data/raw/synthetic_data.pt"):
        # On concatène tout pour avoir des gros tenseurs uniques
        F_all = torch.cat([d['F'] for d in data_list], dim=0)
        t_all = torch.cat([d['t'] for d in data_list], dim=0)
        P_all = torch.cat([d['P'] for d in data_list], dim=0)
        
        torch.save({'F': F_all, 't': t_all, 'P': P_all}, filename)
        print(f"Sauvegardé sous {filename}")

