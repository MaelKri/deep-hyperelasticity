import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class Normalizer:
    def __init__(self, data):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def denormalize(self, x):
        return x * (self.std + 1e-8) + self.mean

def generate_deformation_gradient(lambdas, case='uniaxial'):
    batch_size = len(lambdas)
    F = torch.zeros((batch_size, 3, 3))

    if case == 'uniaxial':
        l1 = lambdas
        l2 = 1.0 / torch.sqrt(l1)
        l3 = l2
    elif case == 'biaxial':
        l1 = lambdas
        l2 = lambdas
        l3 = 1.0 / (l1**2)
    elif case == 'shear':
        l1 = lambdas
        l2 = torch.ones_like(lambdas)
        l3 = 1.0 / l1

    F[:, 0, 0] = l1
    F[:, 1, 1] = l2
    F[:, 2, 2] = l3
    return F

def create_dataset(material, lam_max, n_points=1000):
    lambdas = torch.rand(n_points) * (lam_max - 1.0) + 1.0

    n_per_case = n_points // 3
    l_uni = lambdas[:n_per_case]
    l_bi = lambdas[n_per_case:2*n_per_case]
    l_shear = lambdas[2*n_per_case:]

    F_uni = generate_deformation_gradient(l_uni, 'uniaxial')
    F_bi = generate_deformation_gradient(l_bi, 'biaxial')
    F_shear = generate_deformation_gradient(l_shear, 'shear')

    F_data = torch.cat([F_uni, F_bi, F_shear], dim=0)
    P_truth, psi_truth = material.compute_P_ground_truth(F_data)
    
    return F_data, P_truth, psi_truth

def get_dataloaders(F_train, P_train, psi_train, batch_size=32):
    dataset = TensorDataset(F_train, P_train, psi_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset