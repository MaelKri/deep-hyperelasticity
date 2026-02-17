import torch
import numpy as np
import scipy.special

class W1_GDM_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, I1, h1, h3):
        ctx.save_for_backward(I1, torch.tensor(h1), torch.tensor(h3))
        I1_np = I1.detach().cpu().numpy()
        val = h1 * np.sqrt(np.pi) / (2.0 * np.sqrt(h3)) * scipy.special.erfi(np.sqrt(h3) * (I1_np - 3.0))
        return torch.tensor(val, dtype=I1.dtype, device=I1.device)
        
    @staticmethod
    def backward(ctx, grad_output):
        I1, h1, h3 = ctx.saved_tensors
        grad_I1 = h1 * torch.exp(h3 * (I1 - 3.0)**2)
        return grad_output * grad_I1, None, None

class GornetDesmoratMaterial:
    def __init__(self, h1, h2, h3):
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3

    def compute_energy(self, F):
        C = torch.bmm(F.transpose(1, 2), F)
        I1 = C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2]

        C2 = torch.bmm(C, C)
        tr_C2 = C2[:, 0, 0] + C2[:, 1, 1] + C2[:, 2, 2]
        I2 = 0.5 * (I1**2 - tr_C2)

        W1 = W1_GDM_Function.apply(I1, self.h1, self.h3)
        W2 = 6.0 * self.h2 * (torch.sqrt(I2 + 1e-8) - np.sqrt(3.0))

        psi = W1 + W2
        return psi, I1, I2

    def compute_P_ground_truth(self, F):
        F = F.clone().detach().requires_grad_(True)
        psi, _, _ = self.compute_energy(F)
        grads = torch.autograd.grad(psi.sum(), F, create_graph=False)[0]

        P_temp = grads
        sigma_temp = torch.bmm(P_temp, F.transpose(1, 2))
        p = sigma_temp[:, 2, 2].view(-1, 1, 1)

        F_inv = torch.linalg.inv(F)
        F_inv_T = F_inv.transpose(1, 2)

        P_truth = P_temp - p * F_inv_T
        return P_truth.detach(), psi.detach()