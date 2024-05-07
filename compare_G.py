import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


fitl = 4
total  = 0
for i in range(fitl+1):
    total+=2*i+3




df1 = pd.read_csv('testG.csv',header=0)
df2 = pd.read_csv('testGpredict.csv',header=0)

Gtrue = df1['G']
Gpred = df2['G']

residualsq = (Gtrue-Gpred)**2

x = np.arange(0,len(residualsq))
x2 = np.arange(0,total)

figure,axis = plt.subplots(1,2)
axis0 = axis[0]
axis0.scatter(x,residualsq)
axis0.set_title(r'$\left(G_{l,m}^{true} - G_{l,m}^{predicted}\right)^{2}$')
axis0.set_xlabel(r'$G_{lm}$ fit coefficients')

axis1 = axis[1]
axis1.scatter(x2,Gtrue[0:total],label='True')
axis1.scatter(x2,Gpred,label='Predicted')
axis1.set_title(r'$G_{l,m}$')
axis1.set_xlabel(r'$G_{lm}$ fit coefficients')
axis1.legend()

plt.show()

