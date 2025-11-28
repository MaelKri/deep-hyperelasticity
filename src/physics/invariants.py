import torch

def compute_deformations_measures(F):
    # Calcul des invariants à partir des déformations F

    C=torch.matmul(F.transpose(1,2), F) # Tenseur de Cauchy
    I1 = torch.diagonal(C, dim1=1, dim2=2).sum(dim=1)

    C_squared=torch.matmul(C, C)
    traceC2 = torch.diagonal(C_squared, dim1=1, dim2=2).sum(dim=1)

    I2 = 1/2 * (I1*I1-traceC2)

    I3=torch.det(C)
    I3_star = -torch.sqrt(I3)

    invariants={
        "I1":I1,
        "I2":I2,
        "I3":I3,
        "I3_star":I3_star,
    }
    return(invariants)
