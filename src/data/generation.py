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
    
    def sample_pure_shear(self, n_samples, stretch_min=1.0, stretch_max=4.0):
        lambdas = torch.empty(n_samples).uniform_(stretch_min, stretch_max).to(self.device)
        F = torch.eye(3, device=self.device).repeat(n_samples, 1, 1)
        F[:,0,0]=lambdas
        F[:,1,1]=1.0/lambdas
        return(F, lambdas)
    
class SyntheticDatasetGenerator:
    def __init__(self, potential_model):
        self.model=potential_model
        self.sampler=DeformationSampler()

    def generate(self, n_samples_per_mode=1000):
        data_list=[]
        modes = ['uniaxial', 'biaxial', 'shear', 'pure_shear']
        i=0
        for mode in modes:
            if mode == 'uniaxial':
                F, _ = self.sampler.sample_uniaxial(n_samples_per_mode)
                i+=1
            elif mode == 'biaxial':
                F, _ = self.sampler.sample_biaxial(n_samples_per_mode)
                i+=1
            # elif mode == 'shear':
            #     F, _ = self.sampler.sample_shear(n_samples_per_mode)
            #     i+=1
            elif mode == 'pure_shear':
                F, _ = self.sampler.sample_pure_shear(n_samples_per_mode)
                i+=1
            t=None
            P = self.model.compute_stress(F, t)
            data_list.append({
                'F': F.detach(),
                't': None,
                'P': P.detach(),
                'mode': mode
            })
        print(f"Dataset généré avec {i * n_samples_per_mode} échantillons.")
        return data_list
    def save_dataset(self, data_list, filename="data/raw/synthetic_data.pt"):
        # On concatène tout pour avoir des gros tenseurs uniques
        F_all = torch.cat([d['F'] for d in data_list], dim=0)
        #t_all = torch.cat([d['t'] for d in data_list], dim=0)
        P_all = torch.cat([d['P'] for d in data_list], dim=0)
        
        # torch.save({'F': F_all, 't': t_all, 'P': P_all}, filename)
        torch.save({'F': F_all, 'P': P_all}, filename)

        print(f"Sauvegardé sous {filename}")

if __name__ == "__main__":
    # 1. Instancier le modèle physique (Vérité Terrain)
    gt_model = NeoHookeanPotential()
    
    # 2. Générateur
    generator = SyntheticDatasetGenerator(gt_model)
    
    # 3. Création et sauvegarde
    raw_data = generator.generate(n_samples_per_mode=2000)
    # Assurez-vous que le dossier data/raw existe !
    import os
    os.makedirs("data/raw", exist_ok=True)
    generator.save_dataset(raw_data)

