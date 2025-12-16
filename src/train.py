import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse

from src.data.dataset import HyperelasticDataset
from src.models.icnn import ICNN
from src.models.fcn import FCN
from src.training.loss import SobolevLoss

def train(model_type='icnn', epochs=200, batch_size=32, lr=1e-3, hidden_dim=64):

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f'Entrainement sur {device}')

    os.makedirs("outputs/checkpoints", exist_ok=True)

    dataset_path=r"data/raw/synthetic_data.pt"
    transform=None

    if model_type=='fcn' and False:
        transform = lambda x: x.view(-1)

    dataset=HyperelasticDataset(dataset_path, transform=transform)
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('Données chargées')

    if model_type=='icnn':
        model=ICNN(input_dim=2, hidden_dim=hidden_dim, n_layers=3, use_invariants_layer=True)
    elif model_type=='fcn':
        model=FCN(input_dim=2, hidden_dim=hidden_dim, n_layers=3, use_invariants_layer=True)
    else:
        raise ValueError("Modèle inconnu")

    model.to(device)

    optimizer=optim.Adam(model.parameters(), lr=lr)
    criterion=SobolevLoss()

    loss_history=[]
    print(f"Début de l'entrainement ({epochs} epochs)")

    for epoch in range(epochs):

        epoch_loss=0.0

        for batch_F, batch_P in dataloader:

            batch_F=batch_F.to(device)
            batch_P=batch_P.to(device)

            optimizer.zero_grad()
            loss=criterion(model, batch_F, batch_P)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()

        avg_loss=epoch_loss/len(dataloader)
        loss_history.append(avg_loss)
        if (epoch+1)%10==0:
            print(f"Epoch {epoch+1}/{epochs} | Loss : {avg_loss:.4f}")

    save_path=f"outputs/checkpoints/model_{model_type}.pth"
    torch.save(model.state_dict(), save_path)
    print("Modèle sauvegardé")

    plt.figure()
    plt.plot(loss_history, label='Train Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Sobolev Loss")
    plt.title(f'Entraînement {model_type.upper()}')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"outputs/loss_{model_type}.png")
    print(f"Courbe de loss sauvegardée : outputs/loss_{model_type}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='icnn', choices=['icnn', 'fcn'], help="Type de modèle")
    parser.add_argument('--epochs', type=int, default=100, help="Nombre d'époques")
    parser.add_argument('--lr', type=float, default=0.005, help="Learning Rate")
    
    args = parser.parse_args()
    
    train(model_type=args.model, epochs=args.epochs, lr=args.lr)