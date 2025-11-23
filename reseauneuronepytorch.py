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
    def __init__(self,entree_t,couche2,couche3,sortie):
        super(SimpleNetwork,self).__init__()
        self.layer1 = SimplePosLinear(entree_t, couche2) # la couche personnalisée
        parametrize.register_parametrization(self.layer1, "weights", Positive())

        self.activation1 = nn.Sigmoid()
        self.activation2=nn.ReLU()
        


        
        self.layer2=SimplePosLinear(couche2,couche3)
        parametrize.register_parametrization(self.layer2, "weights", Positive())
        self.layer3=SimplePosLinear(couche3,sortie)
        parametrize.register_parametrization(self.layer3, "weights", Positive())
        self.layer4=SimplePosLinear(couche3,sortie)


    def forward(self, x):
        x = self.layer1(x)
        x = self.activation2(x)
        
        x=self.layer2(x)
        x = self.activation2(x)
        x=self.layer3(x)
       
        
        return x

model = SimpleNetwork(1, 2, 3,1)
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
P_uni_vals = P_uni(lam_uni)

lam_equi = np.linspace(1, 4, 400)
P_equi_vals = P_equi(lam_equi)

lam_sh = np.linspace(1, 4, 400)
P_sh_vals = P_sh(lam_sh)


lam_bi_1 = np.linspace(1, 4, 400)
lam_bi_2 = np.linspace(1, 4, 400)

P_bi_vals_1 = P_bi1(lam_bi_1, lam_bi_2)
P_bi_vals_2 = P_bi2(lam_bi_1, lam_bi_2)
X_=np.transpose(lam_uni)
y_=np.transpose(P_uni_vals)
X_train, X_test, y_train, y_test = train_test_split(X_,y_, test_size=.33, random_state=26)
batch_size=20
data_train=Data(np.stack([X_train],axis=1),y_train.reshape(-1,1))
train_dataloader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
learning_rate = 0.01                      

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                            
num_epochs = 100
loss_values = []


for epoch in range(num_epochs):
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        for p in model.parameters():
            p.data.clamp_(0)

step = range(len(loss_values))

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


data_test=Data(np.stack([X_test],axis=1),y_test.reshape(-1,1))

with torch.no_grad():
    y_pred=model(data_test.X)
plt.plot(X_test,y_pred.numpy(),'r+',label='prédiction',)
plt.plot(X_test,y_test,'b+',label='test')
plt.legend()
plt.show()
