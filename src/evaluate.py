import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Imports depuis votre structure de projet
from src.models.icnn import ICNN
from src.physics.analytical_potentials import NeoHookeanPotential

def get_icnn_stress(model, F):
    """
    Calcule la contrainte P prédite par le modèle ICNN via différentiation automatique.
    P = dPsi/dF
    """
    F_in = F.clone().detach().requires_grad_(True)
    psi = model(F_in)
    
    # Calcul du gradient dPsi/dF
    P_pred = torch.autograd.grad(
        outputs=psi.sum(),
        inputs=F_in,
        create_graph=False,
        retain_graph=False
    )[0]
    
    return P_pred.detach()

def generate_test_data(mode, n_steps=100, device='cpu'):
    """
    Génère des trajectoires de déformation propres (linspace) pour l'évaluation
    """
    # On commence à 1.0 car c'est l'état de repos (pas de contrainte)
    # On va jusqu'à 4.0 comme dans l'entrainement
    lambdas = torch.linspace(1.0, 4.0, n_steps).to(device)
    
    if mode == 'uniaxial':
        # Uniaxial : lambda_1 = l, lambda_2 = lambda_3 = 1/sqrt(l)
        F = torch.zeros(n_steps, 3, 3).to(device)
        lat = 1.0 / torch.sqrt(lambdas)
        F[:, 0, 0] = lambdas
        F[:, 1, 1] = lat
        F[:, 2, 2] = lat
        
    elif mode == 'biaxial':
        # Biaxial : lambda_1 = lambda_2 = l, lambda_3 = 1/l^2
        F = torch.zeros(n_steps, 3, 3).to(device)
        z_contraction = 1.0 / (lambdas**2)
        F[:, 0, 0] = lambdas
        F[:, 1, 1] = lambdas
        F[:, 2, 2] = z_contraction

    elif mode == 'pure_shear':
        # Pure Shear : lambda_1 = l, lambda_2 = 1/l, lambda_3 = 1
        F = torch.zeros(n_steps, 3, 3).to(device)
        inv_lambdas = 1.0 / lambdas
        F[:, 0, 0] = lambdas
        F[:, 1, 1] = inv_lambdas
        F[:, 2, 2] = 1.0
        
    return F, lambdas

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Évaluation sur {device}")

    # 1. Charger le modèle entrainé
    hidden_dim = 64 # Assurez-vous que c'est la même dimension que lors du train
    model = ICNN(input_dim=2, hidden_dim=hidden_dim, n_layers=3, use_invariants_layer=True)
    
    checkpoint_path = "outputs/checkpoints/model_icnn.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Erreur : Le fichier {checkpoint_path} n'existe pas.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval() # Important pour désactiver dropout/batchnorm si présents (pas le cas ici mais bonne pratique)

    # 2. Instancier la vérité terrain (Ground Truth)
    # Assurez-vous que les paramètres mu/lambda sont les mêmes que ceux utilisés pour générer synthetic_data
    gt_model = NeoHookeanPotential(mu=0.5, lam=100.0) 
    gt_model.to(device)

    # 3. Boucle sur les modes de déformation
    modes = ['uniaxial', 'biaxial', 'pure_shear']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, mode in enumerate(modes):
        # A. Génération des données de test
        F_test, lambdas = generate_test_data(mode, n_steps=50, device=device)
        
        # B. Prédiction Ground Truth (Physique)
        # compute_stress gère la correction de pression hydrostatique automatiquement
        P_gt = gt_model.compute_stress(F_test, t=None).detach().cpu()
        
        # C. Prédiction Réseau de Neurones
        P_pred = get_icnn_stress(model, F_test).cpu()
        
        lambdas_np = lambdas.cpu().numpy()

        # D. Sélection de la composante à afficher
        # On affiche généralement P_11 (composante principale de traction)
        p11_gt = P_gt[:, 0, 0]
        p11_pred = P_pred[:, 0, 0]

        # Calcul de l'erreur relative moyenne sur ce mode (optionnel)
        error = torch.mean((p11_gt - p11_pred)**2).item()

        # E. Plot
        ax = axes[i]
        ax.plot(lambdas_np, p11_gt, 'k-', linewidth=2, label='Ground Truth (Néo-Hookéen)')
        ax.plot(lambdas_np, p11_pred, 'r--', linewidth=2, label='Prédiction ICNN')
        
        ax.set_title(f"Mode: {mode.capitalize()} (MSE: {error:.2e})")
        ax.set_xlabel(r"Élongation $\lambda$")
        ax.set_ylabel(r"Contrainte Nominale $P_{11}$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/evaluation_results.png"
    plt.savefig(save_path)
    print(f"Graphiques sauvegardés sous : {save_path}")
    plt.show()

if __name__ == "__main__":
    main()