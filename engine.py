import torch
from models import predict_stress

def train_model(model, dataloader, optimizer, n_epochs, verbose=True):
    loss_history = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for batch_F, batch_P_true, batch_psi_true in dataloader:
            optimizer.zero_grad()
            P_pred, psi_pred = predict_stress(model, batch_F)

            loss_mse = torch.mean((P_pred - batch_P_true)**2)
            relative_error = (P_pred - batch_P_true) / (torch.abs(batch_P_true) + 1e-3)
            loss_relative = torch.mean(relative_error**2)
            
            loss_P = loss_mse + 0.05 * loss_relative 
            loss_psi = torch.mean((psi_pred.squeeze() - batch_psi_true)**2)
            
            loss = loss_P + 0.1 * loss_psi

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | Loss pondérée : {avg_loss:.8f}")
            
    return loss_history