import torch
import torch.nn as nn

class SobolevLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_criterion=nn.MSELoss()
    
    def forward(self, model, F, P_target):
        F_in=F.clone().detach().requires_grad_(True)
        psi=model(F_in)

        P_pred=torch.autograd.grad(
            outputs=psi.sum(),
            inputs=F_in,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        loss=self.mse_criterion(P_pred, P_target)
        return loss