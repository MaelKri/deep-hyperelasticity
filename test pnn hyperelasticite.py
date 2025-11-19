import os

# Désactive les logs TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Désactive oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
#paramètres matériaux 

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



# modèle Hart-Smith

a1 = 0.140 #MPa
a2 = 5.2541e-4 #MPa
a3 = 1.29 #MPa 


def dW1_HS(I1):

    dW1 = a1*np.exp(a3*(I1-3)**2)

    return dW1

# dW2/dI2

def dW2_HS(I2):

    dW2 = a1*a2/I2
    
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



#uniaxal
#lam = np.linspace(1, 7, 200)
#P = P_uni(lam)

#plt.plot(lam, P)
#plt.xlabel(r" $\lambda$")
#plt.ylabel(r" $P$")
#plt.title(r" Modèle GD, Chargement uniaxial")
#plt.show()

#equibiaxpal
#lam = np.linspace(1, 7, 200)
#P2 = P_equi(lam)

#plt.plot(lam, P2)
#plt.xlabel(r" $\lambda$")
#plt.ylabel(r" $P$")
#plt.title(r" Modèle GD, Chargement uniaxial")
#plt.show()


# --- Données analytiques (déjà définies plus haut)

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

# Reshape pour TensorFlow
X = lam_all.reshape(-1, 1).astype(np.float32)
y = P_all.reshape(-1, 1).astype(np.float32)



# On condense tout 
# === Types de chargements ===
type_uni = np.zeros_like(lam_uni)           # 0
type_equi = np.ones_like(lam_equi)          # 1
type_sh = np.full_like(lam_sh, 2)           # 2
type_bi_1 = np.full_like(lam_bi_1, 3)       # 3
type_bi_2 = np.full_like(lam_bi_2, 4)       # 4

types_all = np.concatenate([type_uni, type_equi, type_sh, type_bi_1, type_bi_2])

lam_all = np.concatenate([lam_uni, lam_equi, lam_sh, lam_bi_1, lam_bi_2])
P_all = np.concatenate([P_uni_vals, P_equi_vals, P_sh_vals, P_bi_vals_1, P_bi_vals_2])

X_multi = np.stack([lam_all, types_all], axis=1).astype(np.float32)
y = P_all.reshape(-1, 1).astype(np.float32)

# === Réseau ===
model_multi = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(1)
])

model_multi.compile(optimizer='adam', loss='mse')
model_multi.fit(X_multi, y, epochs=1500, batch_size=32, verbose=0)

# === Prédictions ===
X_uni = np.stack([lam_uni, np.zeros_like(lam_uni)], axis=1)
X_equi = np.stack([lam_equi, np.ones_like(lam_equi)], axis=1)
X_sh = np.stack([lam_sh, np.full_like(lam_sh, 2)], axis=1)
X_bi_1 = np.stack([lam_bi_1, np.full_like(lam_bi_1, 3)], axis=1)
X_bi_2 = np.stack([lam_bi_2, np.full_like(lam_bi_2, 4)], axis=1)


P_pred_uni = model_multi.predict(X_uni)
P_pred_equi = model_multi.predict(X_equi)
P_pred_sh = model_multi.predict(X_sh)
P_pred_bi_1 = model_multi.predict(X_bi_1)
P_pred_bi_2 = model_multi.predict(X_bi_2)


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
plt.ylabel(r'$P$ (MPa)')
plt.legend()
plt.title('Comparaison modèle analytique vs réseau de neurones multi-entrée')
plt.show()