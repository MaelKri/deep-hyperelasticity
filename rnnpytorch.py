import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.utils.parametrize as parametrize
class SimplePosLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimplePosLinear, self).__init__()
        # tenseur des poid (W)
        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        # tenseur des biais (b)
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x):
        # operation calculée dans les neurones de la couche : Y = X * W + b
        
        print(self.weights)
        
        return  x @ self.weights + self.bias
class Positive(nn.Module):
    def forward(self,x):
        return nn.functional.relu(x)

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNetwork, self).__init__()
        # On utilise nn.Linear standard, on gérera la positivité dans la boucle
        self.layer1 = nn.Linear(input_size, hidden_size) 
        self.activation = nn.Softplus() # Softplus est lisse et convexe, mieux que ReLU ici
        
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        
        x = self.layer2(x)
        x = self.activation(x)
        
        x = self.layer3(x)
        return x

class Data(Dataset):
    def __init__(self, X, y):
          self.X = torch.from_numpy(X.astype(np.float32))
          self.y = torch.from_numpy(y.astype(np.float32))
          self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
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
P_vals = P_uni(lam_uni)
I1_vals, I2_vals = I_uni(lam_uni)
I2_star = -np.sqrt(I2_vals)

# lam_equi = np.linspace(1, 4, 400)
# P_equi_vals = P_equi(lam_equi)

# lam_sh = np.linspace(1, 4, 400)
# P_sh_vals = P_sh(lam_sh)


# lam_bi_1 = np.linspace(1, 4, 400)
# lam_bi_2 = np.linspace(1, 4, 400)

# P_bi_vals_1 = P_bi1(lam_bi_1, lam_bi_2)
# P_bi_vals_2 = P_bi2(lam_bi_1, lam_bi_2)
# X_=np.transpose(lam_uni)
# y_=np.transpose(P_uni_vals)

X_ = np.stack([I1_vals, I2_vals, I2_star], axis=1)
y_ = P_vals.reshape(-1, 1)

X_mean = X_.mean(axis=0)
X_std = X_.std(axis=0)
X_scaled = (X_ - X_mean) / X_std

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y_, test_size=.33, random_state=26)

data_train=Data(X_train, y_train)
train_dataloader = DataLoader(dataset=data_train, batch_size=20, shuffle=True)


model = SimpleNetwork(3, 16, 1)

learning_rate = 0.01                      

loss_fn = nn.MSELoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer =torch.optim.Adam(model.parameters(), lr=learning_rate)
                            
num_epochs = 100
loss_values = []


for epoch in range(num_epochs):
    epoch_loss = 0
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(X)
        # loss = loss_fn(pred, y)

        relative_error = (pred - y) / (torch.abs(y) + 0.1)
        loss = torch.mean(relative_error**2)

        # loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name:
                    param.clamp_(min=0.0)
        epoch_loss += loss.item()
    loss_values.append(epoch_loss / len(train_dataloader))

# step = range(len(loss_values))

# fig, ax = plt.subplots(figsize=(8,5))
# plt.plot(step, np.array(loss_values))
# plt.title("Step-wise Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.show()


# data_test=Data(np.stack([X_test],axis=1),y_test.reshape(-1,1))

# with torch.no_grad():
#     y_pred=model(data_test.X)
# plt.plot(X_test,y_pred.numpy(),'r+',label='prédiction',)
# plt.plot(X_test,y_test,'b+',label='test')
# plt.legend()
# plt.show()

plt.figure(figsize=(8,5))
plt.plot(loss_values)
plt.title("Loss per Epoch")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()

data_test = Data(X_test, y_test)
model.eval()
with torch.no_grad():
    y_pred = model(data_test.X)

y_test_np = y_test
y_pred_np = y_pred.numpy()

plt.figure(figsize=(8,5))
plt.plot(y_test_np, y_test_np, 'b-', label='Idéal')
plt.plot(y_test_np, y_pred_np, 'r+', label='Prédiction')
plt.title("Prédiction vs Vérité Terrain")
plt.xlabel("Vraie Contrainte P (Test)")
plt.ylabel("Contrainte Prédite P (Modèle)")
plt.legend()
plt.show()

lam_plot = np.linspace(1, 7, 100)

P_true = P_uni(lam_plot)

I1_plot, I2_plot = I_uni(lam_plot)
I2_star_plot = -np.sqrt(I2_plot)

X_plot_raw = np.stack([I1_plot, I2_plot, I2_star_plot], axis=1)

X_plot_scaled = (X_plot_raw - X_mean) / X_std

X_plot_tensor = torch.from_numpy(X_plot_scaled.astype(np.float32))

model.eval()
with torch.no_grad():
    P_pred_plot = model(X_plot_tensor).numpy()

plt.figure(figsize=(10, 6))
plt.plot(lam_plot, P_true, 'k--', label='Analytique (Ground Truth)', linewidth=2, alpha=0.7)
plt.plot(lam_plot, P_pred_plot, 'r-', label='Prédiction PANN', linewidth=2)

plt.title("Comparaison Contrainte vs Élongation (Uniaxial)")
plt.xlabel(r"Élongation $\lambda$")
plt.ylabel("Contrainte Nominale $P$ (MPa)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
