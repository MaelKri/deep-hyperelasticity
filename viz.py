import matplotlib.pyplot as plt
import numpy as np
import torch
from models import predict_stress
from data import generate_deformation_gradient

def visualize_training_dataset(F_data, P_data, psi_data):
    n_points_total = len(F_data)
    n_per_case = n_points_total // 3
    
    cases = [
        ('Traction Uniaxiale', 0, n_per_case, 'blue'),
        ('Traction Biaxiale', n_per_case, 2 * n_per_case, 'orange'),
        ('Cisaillement Pur', 2 * n_per_case, n_points_total, 'green')
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, start_idx, end_idx, color in cases:
        F_case = F_data[start_idx:end_idx]
        P_case = P_data[start_idx:end_idx]
        psi_case = psi_data[start_idx:end_idx]
        
        lambdas = F_case[:, 0, 0].numpy()
        P11 = P_case[:, 0, 0].numpy()
        
        psi = psi_case.squeeze().numpy()
        
        sort_indices = np.argsort(lambdas)
        lambdas_sorted = lambdas[sort_indices]
        P11_sorted = P11[sort_indices]
        psi_sorted = psi[sort_indices]
        
        axes[0].scatter(lambdas_sorted, P11_sorted, color=color, alpha=0.7, s=15, label=name)
        # axes[0].plot(lambdas_sorted, P11_sorted, label=name, color=color, linewidth=2)
        
        axes[1].scatter(lambdas_sorted, psi_sorted, color=color, alpha=0.7, s=15, label=name)
        # axes[1].plot(lambdas_sorted, psi_sorted, label=name, color=color, linewidth=2)

    axes[0].set_title("Dataset : Contrainte Nominale P11", fontweight='bold')
    axes[0].set_xlabel("Étirement lambda")
    axes[0].set_ylabel("Contrainte P11 (MPa)")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    
    axes[1].set_title("Dataset : Énergie de Déformation Psi", fontweight='bold')
    axes[1].set_xlabel("Étirement lambda")
    axes[1].set_ylabel("Énergie Psi (MJ/m³)")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def compute_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
    
    max_error = np.max(np.abs(residuals))
    return rmse, r2, max_error


def plot_train_results(model, F_train, P_train):
    n_total = len(F_train)
    n_per_case = n_total // 3
    
    cases = [
        ('Traction Uniaxiale', 0, n_per_case),
        ('Traction Biaxiale', n_per_case, 2 * n_per_case),
        ('Cisaillement Pur', 2 * n_per_case, n_total)
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (name, start_idx, end_idx) in enumerate(cases):
        F_case = F_train[start_idx:end_idx]
        P_true_case = P_train[start_idx:end_idx]
        
        # Prédiction (Autograd nécessite enable_grad même en évaluation)
        with torch.enable_grad():
            P_pred_case, _ = predict_stress(model, F_case)
            
        lambdas = F_case[:, 0, 0].detach().numpy()
        y_true = P_true_case[:, 0, 0].detach().numpy()
        y_pred = P_pred_case[:, 0, 0].detach().numpy()
        
        sort_idx = np.argsort(lambdas)
        lambdas_sorted = lambdas[sort_idx]
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        # Calcul des métriques
        rmse, r2, max_err = compute_metrics(y_true_sorted, y_pred_sorted)
        
        ax = axes[i]
        
        ax.scatter(lambdas_sorted, y_true_sorted, color='black', alpha=0.4, s=20, label='Données Train (Vérité)')
        ax.plot(lambdas_sorted, y_pred_sorted, 'g-', label='Prédiction Modèle', linewidth=2.5)
        ax.fill_between(lambdas_sorted, y_true_sorted, y_pred_sorted, color='red', alpha=0.1)
        
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Étirement $\lambda$", fontsize=12)
        ax.set_ylabel(r"Contrainte $P_{11}$ (MPa)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0: ax.legend(loc='lower right')
        
        textstr = '\n'.join((
            r'$\mathrm{R}^2=%.4f$' % (r2, ),
            r'$\mathrm{RMSE}=%.3f$ MPa' % (rmse, ),
            r'$\mathrm{MaxErr}=%.3f$ MPa' % (max_err, )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

    plt.suptitle("Évaluation sur le Set d'Entraînement (Points aléatoires)", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

def plot_test_results(model, material):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    l_test = torch.linspace(1.0, 7.0, 50)
    
    test_cases = [
        ('uniaxial', 'Traction Uniaxiale'), 
        ('biaxial', 'Traction Biaxiale'), 
        ('shear', 'Cisaillement Pur')
    ]
    
    for i, (case_key, case_name) in enumerate(test_cases):
        F_case = generate_deformation_gradient(l_test, case_key)
        
        with torch.enable_grad():
            P_true_tensor, _ = material.compute_P_ground_truth(F_case)
            P_pred_tensor, _ = predict_stress(model, F_case)
            
        x_axis = l_test.numpy()
        y_true = P_true_tensor[:, 0, 0].detach().numpy()
        y_pred = P_pred_tensor[:, 0, 0].detach().numpy()
        
        rmse, r2, max_err = compute_metrics(y_true, y_pred)
        
        ax = axes[i]
        
        ax.plot(x_axis, y_true, 'k--', label='Vérité Analytique (Test)', linewidth=2.5, alpha=0.8)
        ax.plot(x_axis, y_pred, 'b-', label='Modèle ICNN', linewidth=2.5)
        ax.fill_between(x_axis, y_true, y_pred, color='red', alpha=0.1)
        
        ax.set_title(case_name, fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Étirement $\lambda$", fontsize=12)
        ax.set_ylabel(r"Contrainte $P_{11}$ (MPa)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0: ax.legend(loc='lower right')
        
        textstr = '\n'.join((
            r'$\mathrm{R}^2=%.4f$' % (r2, ),
            r'$\mathrm{RMSE}=%.3f$ MPa' % (rmse, ),
            r'$\mathrm{MaxErr}=%.3f$ MPa' % (max_err, )))
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

    plt.suptitle("Validation sur un Set de Test (Linéairement espacé)", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()