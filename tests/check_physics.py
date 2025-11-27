import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.generation import DeformationSampler
from src.physics.analytical_potentials import NeoHookeanPotential, GornetDesmoratPotential

def analytical_neohookean_uniaxial_stress(lambdas, mu):
    return mu * (lambdas - 1.0 / (lambdas**2))

def plot_model_behavior():
    device = 'cpu'
    torch.manual_seed(42)
    
    nh_model = NeoHookeanPotential(mu=0.5, lam=100.0)
    
    gd_model = GornetDesmoratPotential(h1=0.1, h2=0.05, h3=0.05)
    
    sampler = DeformationSampler(device=device)
    
    # --- TEST 1 : TRACTION UNIAXIALE (Comparaison avec Analytique) ---
    print("--- Génération Traction Uniaxiale ---")
    n_steps = 50

    lambdas = torch.linspace(1.0, 4.0, n_steps).to(device)
    
    F_uni = torch.zeros(n_steps, 3, 3)
    F_uni[:, 0, 0] = lambdas
    F_uni[:, 1, 1] = 1.0 / torch.sqrt(lambdas)
    F_uni[:, 2, 2] = 1.0 / torch.sqrt(lambdas)
    F_uni.requires_grad = True
    
    P_nh = nh_model.compute_stress(F_uni, t=None) 
    P_gd = gd_model.compute_stress(F_uni, t=None)
    
    P11_nh = P_nh[:, 0, 0].detach().numpy()
    P11_gd = P_gd[:, 0, 0].detach().numpy()
    lam_np = lambdas.detach().numpy()
    
    P11_exact = analytical_neohookean_uniaxial_stress(lam_np, mu=0.5)
    I1 = lam_np**2+2/lam_np
    I2 = 2*lam_np+ 1/lam_np**2
    h1=0.1
    h2 = 0.05
    h3 = 0.05
    # h1=0.0157
    # h2 = 0.0098
    # h3 = 0.000561
    P11_exact2 = 2*(lam_np-1/lam_np**2)*(h1*np.exp(h3*(I1-3))+3*h2/(lam_np*np.sqrt(I2)))
    
    # --- TEST 2 : CISAILLEMENT SIMPLE ---
    print("--- Génération Cisaillement Simple ---")
    gammas = torch.linspace(0.0, 0.5, n_steps).to(device)
    F_shear = torch.eye(3).unsqueeze(0).repeat(n_steps, 1, 1)
    F_shear[:, 0, 1] = gammas
    F_shear.requires_grad = True
    
    P_shear_nh = nh_model.compute_stress(F_shear, t=None)
    P_shear_gd = gd_model.compute_stress(F_shear, t=None)
    
    P12_nh = P_shear_nh[:, 0, 1].detach().numpy()
    P12_gd = P_shear_gd[:, 0, 1].detach().numpy()
    gam_np = gammas.detach().numpy()

    # --- PLOTS ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.plot(lam_np, P11_exact, 'k--', linewidth=2, label='Analytique Exact (NH)')
    ax.plot(lam_np, P11_nh, 'o', markersize=4, label='Votre Code (NH)', alpha=0.7)
    ax.plot(lam_np, P11_gd, 'o', linewidth=2, label='Gornet-Desmorat', alpha=0.7)
    ax.plot(lam_np, P11_exact2, 'b-', linewidth=2, label='Gornet-Desmorat Th', alpha=0.7)

    ax.set_title("Test Traction Uniaxiale ($P_{11}$ vs $\lambda$)")
    ax.set_xlabel("Élongation $\lambda$")
    ax.set_ylabel("Contrainte $P_{11}$ [MPa]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(gam_np, P12_nh, 'b-', label='Neo-Hookean')
    ax.plot(gam_np, P12_gd, 'r-', label='Gornet-Desmorat')
    ax.set_title("Test Cisaillement ($P_{12}$ vs $\gamma$)")
    ax.set_xlabel("Cisaillement $\gamma$")
    ax.set_ylabel("Contrainte $P_{12}$ [MPa]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        plot_model_behavior()
        print("\nTest terminé.")
    except Exception as e:
        print(f"\n Erreur lors du test : {e}")
        import traceback
        traceback.print_exc()