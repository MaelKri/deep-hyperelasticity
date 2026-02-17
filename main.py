import os
from datetime import datetime

import argparse
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from physics import GornetDesmoratMaterial
from data import create_dataset, get_dataloaders, Normalizer
from models import PhysicsInputLayer, GornetModel
from engine import train_model
from viz import visualize_training_dataset, plot_train_results, plot_test_results

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement du modèle ICNN pour le matériau de Gornet-Desmorat",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    mat_group = parser.add_argument_group('Paramètres du matériau')
    mat_group.add_argument('--h1', type=float, default=0.0157, help="Paramètre h1 du matériau (MPa)")
    mat_group.add_argument('--h2', type=float, default=0.0098, help="Paramètre h2 du matériau (MPa)")
    mat_group.add_argument('--h3', type=float, default=0.000561, help="Paramètre h3 du matériau (MPa)")
    
    data_group = parser.add_argument_group('Paramètres des données')
    data_group.add_argument('--lambda_max', type=float, default=5.0, help="Étirement maximum (lambda max)")
    data_group.add_argument('--n_points', type=int, default=1500, help="Nombre total de points dans le dataset")
    data_group.add_argument('--batch_size', type=int, default=32, help="Taille des batchs d'entraînement")
    
    train_group = parser.add_argument_group("Paramètres d'entraînement")
    train_group.add_argument('--learning_rate', '--lr', type=float, default=2e-3, help="Taux d'apprentissage (Learning rate)")
    train_group.add_argument('--epochs', type=int, default=200, help="Nombre d'époques d'entraînement")
    
    arch_group = parser.add_argument_group('Architecture du réseau')
    arch_group.add_argument('--n_hidden', type=int, default=32, help="Nombre de neurones cachés")
    arch_group.add_argument('--n_layers', type=int, default=5, help="Nombre de couches")
    arch_group.add_argument('--seed', type=int, default=42, help="Graine aléatoire pour la reproductibilité")

    others=parser.add_argument_group("Autres")
    others.add_argument('--no_save', dest='save_model', action='store_false', help="Désactiver la sauvegarde automatique du modèle")

    return parser.parse_args()

def main():

    args=parse_args()

    torch.manual_seed(args.seed)

    # 1. PRÉPARATION DES DONNÉES
    print("Initialisation du matériau et création du dataset")
    material = GornetDesmoratMaterial(args.h1, args.h2, args.h3)
    F_train, P_train, psi_train = create_dataset(material, lam_max=args.lambda_max, n_points=args.n_points)
    
    with torch.no_grad():
        C_train = torch.bmm(F_train.transpose(1, 2), F_train)
        I1_train = C_train[:, 0, 0] + C_train[:, 1, 1] + C_train[:, 2, 2]
        C2_train = torch.bmm(C_train, C_train)
        tr_C2_train = C2_train[:, 0, 0] + C2_train[:, 1, 1] + C2_train[:, 2, 2]
        I2_train = 0.5 * (I1_train**2 - tr_C2_train)

        f1_raw = (I1_train - 3.0).view(-1, 1)
        f2_raw = (torch.sqrt(I2_train + 1e-8) - np.sqrt(3.0)).view(-1, 1)

    norm_f1 = Normalizer(f1_raw)
    norm_f2 = Normalizer(f2_raw)

    dataloader, dataset = get_dataloaders(F_train, P_train, psi_train, batch_size=args.batch_size)
    print(f"Dataset prêt : {len(dataset)} échantillons.")

    visualize_training_dataset(F_train, P_train, psi_train)

    # 2. INITIALISATION DU MODELE
    input_layer = PhysicsInputLayer(norm_f1, norm_f2)
    model = GornetModel(input_layer, n_hidden=args.n_hidden, n_layers=args.n_layers)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 3. ENTRAiNEMENT
    print("Début de l'entraînement")
    loss_history = train_model(model, dataloader, optimizer, args.epochs, verbose=True)

    # 4. RESULTATS
    plt.figure(figsize=(8, 3))
    plt.plot(loss_history)
    plt.title("Historique Loss ICNN")
    plt.yscale("log")
    plt.grid()
    plt.show()

    print("Évaluation sur le Train set...")
    plot_train_results(model, F_train, P_train)

    print("Évaluation sur le Test set (analytique)...")
    plot_test_results(model, material)

    # 6. SAUVEGARDE
    if args.save_model:
        os.makedirs("checkpoints", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoints/icnn_gornet_{timestamp}.pth"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyperparameters': {
                'h1': args.h1, 'h2': args.h2, 'h3': args.h3,
                'n_hidden': args.n_hidden,
                'n_layers': args.n_layers
            }
        }
        torch.save(checkpoint, filename)
        print(f"\n Modèle sauvegardé sous : {filename}")

if __name__ == "__main__":
    main()