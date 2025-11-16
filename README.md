# deep-hyperelasticity

## 1. Le projet :

Ce projet vise à développer des réseaux de neurones informés physiquement (PINN) afin de reproduire et d'améliorer la modélisation du comportement non linéaire des matériaux hyperélastiques . Le travail s'articule autour du modèle analytique GD (Gornet-Desmorat) et vise à inclure la modélisation de l'effet Mullins et l'impact du renforcement par des charges.

---

## 2. Couches de neurones personnalisées

Avec PyTorch, on peut créer des couches personnalisées pour les réseaux de neurones. Voici un exemple (reproduction d'une couche linéaire simple) :

**Définition de la Couche**

```python
import torch.nn as nn

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
```

Cette couche peut ensuite servir dans un réseau de neurones :

```python
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
```