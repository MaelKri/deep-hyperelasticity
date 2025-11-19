import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
import numpy as np 
#import tensorflow as tf     
import matplotlib.pyplot as plt



# ref de l'article : Parametrized polyconvex hyperelasticity with physics-augmented neural networks
# but de ce code : reproduire le réseau de neurones numero 1 de l'article  avec PyTorch et forcer la polyconvexité
# Avec PyTorch, on peut créer des couches personnalisées pour les réseaux de neurones. Voici un exemple (reproduction d'une couche linéaire simple) :


class SimpleCustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleCustomLinear, self).__init__()
        # tenseur des poid (W)
        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        # tenseur des biais (b)
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # operation calculée dans les neurones de la couche : Y = X * W + b
        return x @ self.weights + self.bias
    



# cette couche peut ensuite servir dans un réseau de neurones complet:


class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = SimpleCustomLinear(input_size, hidden_size) # la couche personnalisée
        self.activation = nn.ReLU()     
        self.layer2 = SimpleCustomLinear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        return self.layer2(x)
    





# Maintenant, pour forcer la polyconvexité, on peut définir une couche personnalisée qui impose des contraintes sur les poids 
# et utilise des fonctions d'activation convexes.
# sommes de couches convexes alors polyconvexité

# sortie de type W(F) = g([F,cofF,detF])  alors si toutes les conditions plus haut c'est polyconvexe


#in_features : nombre d'entrées
#out_features : nombre de sorties

class PolyconvexLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(PolyconvexLayer, self).__init__()
        # poids contraints à être positifs pour assurer la polyconvexité
        #poids contraints positifs par fonction torch.abs()
        self.weights = nn.Parameter(torch.abs(torch.randn(in_features, out_features)))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # opération linéaire suivie d'une activation convexe (ReLU)
        linear_output = x @ self.weights + self.bias
        return torch.relu(linear_output)
    


# simple réseau à 2 couches utilisant la couche polyconvexe définie ci-dessus
#input size : taille des entrées (par exemple 2 pour lambda et type de chargement)
#hidden size : nombre de neurones dans la couche cachée
#output size : taille des sorties (par exemple 1 pour la prédiction de la contrainte)


#class PolyconvexNetwork(nn.Module):
  #  def __init__(self, input_size, hidden_size, output_size):
      #  super().__init__()
      #  self.layer1 = PolyconvexLayer(input_size, hidden_size)
      #  self.activation = nn.ReLU()  # fonction d'activation convexe
      #  self.layer2 = PolyconvexLayer(hidden_size, output_size)

   # def forward(self, x):
      #  x = self.layer1(x)
      #  x = self.activation(x)
       # return self.layer2(x)      
    
# réseau plus profond avec plusieurs couches cachées polyconvexes
class PolyconvexNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        hidden_sizes : liste des tailles des couches cachées
        ex: [32, 64, 64, 32] pour 4 couches cachées
        """
        super().__init__()
        layers = []

        # première couche
        layers.append(PolyconvexLayer(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # couches cachées suivantes
        for i in range(len(hidden_sizes)-1):
            layers.append(PolyconvexLayer(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        # couche de sortie
        layers.append(PolyconvexLayer(hidden_sizes[-1], output_size))

        # on met toutes les couches dans un ModuleList
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
# Exemple d'instanciation du réseau polyconvexe

# résumé : créer des couches personnalisées avec des contraintes sur les poids et des fonctions d'activation convexes pour garantir la polyconvexité du réseau de neurones.
# On peut ensuite entraîner ce modèle avec des données appropriées pour la modélisation hyperélastique.    




# fonctions analytiques pour les différents chargements


h1 = 0.0157 #MPa
h2 = 0.0098 #MPa
h3 = 0.000561 #MPa 


# modèle GD

# dW1/dI1

def dW1_GD(I1):

    dW1 = h1*np.exp(h3*(I1-3)**2)

    return dW1

# dW2/dI2

def dW2_GD(I2):

    dW2 = 3*h2/(np.sqrt(I2))
    
    return dW2


def I_uni(lam):
    I1 =  2*lam + 1/lam**2
    I2 = 1 + 1/(lam**2) + lam**2
    return I1, I2 


def I_sh(lam):
    I1 =  lam**2 + 1/(lam**2) + 1
    #I2 = 1 + 1/(lam**2) + lam**2
    I2 = I1
    return I1, I2 

def I_equi(lam):
    I1 =  2*(lam**2) + 1/(lam**4)
    I2 = lam**4 + 2*(1/lam**2)
    return I1, I2 

def I_biaxiale(lam1, lam2):
    lam3 = 1/(lam1*lam2)
    I1 =  lam1**2 + lam2**2 + lam3**2
    I2 = (lam1*lam2)**2 + (lam2*lam3)**2 + (lam1*lam3)**2
    return I1, I2 

#uniaxial tension 

def P_uni(lam):
  
    I1, I2 =  I_uni(lam)

    dW1 = dW1_GD(I1)
    dW2 = dW2_GD(I2)

    P = 2*(lam - (1/lam**2))*(dW1 + (1/lam)*dW2)
    return P 



#equibiaxal extension 

def P_equi(lam):
    I1, I2 = I_equi(lam)

    dW1 = dW1_GD(I1)
    dW2 = dW2_GD(I2)

    P = 2*(lam - (1/lam**5))*(dW1 + (lam**2)*dW2)
    return P 


#pure shear 

def P_sh(lam):

    I1, I2 = I_sh(lam)

    dW1 = dW1_GD(I1)
    dW2 = dW2_GD(I2)

    P = 2*(lam - (1/lam**3))*(dW1 + dW2)
    return P 



#biaxial extension

def P_bi1(lam1, lam2):
    I1, I2 = I_biaxiale(lam1, lam2)
    dW1 = dW1_GD(I1)
    dW2 = dW2_GD(I2)
    lam3 = 1 / (lam1 * lam2)
    P1 = 2 * (lam1 * dW1 - lam3**2 / lam1 * dW2)
    return P1

def P_bi2(lam1, lam2):
    I1, I2 = I_biaxiale(lam1, lam2)
    dW1 = dW1_GD(I1)
    dW2 = dW2_GD(I2)
    lam3 = 1 / (lam1 * lam2)
    P2 = 2 * (lam2 * dW1 - lam3**2 / lam2 * dW2)
    return P2



lam_uni = np.linspace(1, 7, 400)
P_uni_vals = P_uni(lam_uni)

lam_equi = np.linspace(1, 4, 400)
P_equi_vals = P_equi(lam_equi)

lam_sh = np.linspace(1, 4, 400)
P_sh_vals = P_sh(lam_sh)


lam_bi_1 = np.linspace(1, 4, 400)
lam_bi_2 = np.linspace(1, 4, 400)

P_bi_vals_1 = P_bi1(lam_bi_1, lam_bi_2)
P_bi_vals_2 = P_bi2(lam_bi_1, lam_bi_2)


# --- On combine tout 
lam_all = np.concatenate([lam_uni, lam_equi, lam_sh, lam_bi_1, lam_bi_2])
P_all = np.concatenate([P_uni_vals, P_equi_vals, P_sh_vals, P_bi_vals_1, P_bi_vals_2])

# === Types de chargements ===
type_uni = np.zeros_like(lam_uni)           # 0
type_equi = np.ones_like(lam_equi)          # 1
type_sh = np.full_like(lam_sh, 2)           # 2
type_bi_1 = np.full_like(lam_bi_1, 3)       # 3
type_bi_2 = np.full_like(lam_bi_2, 4)
types_all = np.concatenate([type_uni, type_equi, type_sh, type_bi_1, type_bi_2])
lam_all = np.concatenate([lam_uni, lam_equi, lam_sh, lam_bi_1, lam_bi_2])
P_all = np.concatenate([P_uni_vals, P_equi_vals, P_sh_vals, P_bi_vals_1, P_bi_vals_2])
X_multi = np.stack([lam_all, types_all], axis=1).astype(np.float32)
y = P_all.reshape(-1, 1).astype(np.float32)





# %%

# Entrées pour le réseau de neurones 
X_uni_t = torch.tensor(np.stack([lam_uni, np.zeros_like(lam_uni)], axis=1), dtype=torch.float32)
X_equi_t = torch.tensor(np.stack([lam_equi, np.ones_like(lam_equi)], axis=1), dtype=torch.float32)
X_sh_t = torch.tensor(np.stack([lam_sh, np.full_like(lam_sh, 2)], axis=1), dtype=torch.float32)
X_bi_1_t = torch.tensor(np.stack([lam_bi_1, np.full_like(lam_bi_1, 3)], axis=1), dtype=torch.float32)
X_bi_2_t = torch.tensor(np.stack([lam_bi_2, np.full_like(lam_bi_2, 4)], axis=1), dtype=torch.float32)






# on utilise le réseau polyconvexe défini ci-dessus

#pour l'instant les entrées sont de taille 2 (lambda et type de chargement)

hidden_sizes = [32, 64, 64, 32]
model = PolyconvexNetwork(input_size=2, hidden_sizes= hidden_sizes, output_size=1)

# Définition de la fonction de perte et de l'optimiseur
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X = torch.tensor(X_multi, dtype=torch.float32)
Y = torch.tensor(y, dtype=torch.float32)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()


model.eval()

with torch.no_grad():  # pas besoin de gradients pour la prédiction
    P_pred_uni = model(X_uni_t)
    P_pred_equi = model(X_equi_t)
    P_pred_sh = model(X_sh_t)
    P_pred_bi_1 = model(X_bi_1_t)
    P_pred_bi_2 = model(X_bi_2_t)



P_pred_uni = P_pred_uni.numpy()
P_pred_equi = P_pred_equi.numpy()
P_pred_sh = P_pred_sh.numpy()
P_pred_bi_1 = P_pred_bi_1.numpy()
P_pred_bi_2 = P_pred_bi_2.numpy()




plt.figure(figsize=(10,6))
plt.plot(lam_uni, P_uni_vals, 'b-', label='Analytique Uniaxial')
plt.plot(lam_uni, P_pred_uni, 'b--', label='NN Uniaxial')
plt.plot(lam_equi, P_equi_vals, 'r-', label='Analytique Equibiaxial')
plt.plot(lam_equi, P_pred_equi, 'r--', label='NN Equibiaxial')
plt.plot(lam_sh, P_sh_vals, 'g-', label='Analytique Shear')
plt.plot(lam_sh, P_pred_sh, 'g--', label='NN Shear')
plt.plot(lam_sh, P_bi_vals_1, 'y-', label='Analytique Biaxial 1')
plt.plot(lam_sh, P_pred_bi_1, 'y--', label='NN Biaxial 1')
plt.plot(lam_sh, P_bi_vals_2, '-', linewidth=2, color='orange', label='Analytique Biaxial 2')
plt.plot(lam_sh, P_pred_bi_2, '--', linewidth=2, color='orange', label='NN Biaxial 2')
plt.xlabel(r'$\lambda$')
plt.ylabel('Première contrainte de Piola-Kirchhoff P')
plt.legend()
plt.show()
