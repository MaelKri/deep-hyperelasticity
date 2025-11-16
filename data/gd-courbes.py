# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 21:34:50 2025

@author: hugoc
"""

import numpy as np

h1=0.0157
h2=0.098
h3=0.000561


def dW1(I1):
    return h1*np.exp(h3*(I1-3)**2)

def dW2(I2):
    return h2/np.sqrt(I2)

def P_traction(l):
    I1=l**2+2/l
    I2=2*l+1/l
    return 2*(l-1/(l**2))*(dW1(I1)+(1/l)*dW2(I2))

def P_cisaillement(l):
    I1=l**2+1+1/(l**2)
    I2=I1
    return 2*(l-1/(l**3))*(dW1(I1)+dW2(I2))

def biaxial(l1,l2):
    l3=1/(l1*l2)
    I1=l1**2+l2**2+l3**2
    I2=(l1*l2)**2+(l2*l3)**2+(l1*l3)**2
    P1=2*(l1-1/(l1**3*l2**2))*(dW1(I1)+l2**2*dW2(I2))
    P2=2*(l2-1/(l2**3*l1**2))*(dW1(I1)+l1**2*dW2(I2))
    return P1,P2



    
