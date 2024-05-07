import numpy as np
import sympy
import scipy
from sympy import *
import pandas as pd
import sys
###------- SOME IMPORTANT FUNCTIONS ------------###

def convert_to_spherical(mat):      # convert from x,y,z to r,theta,phi
    new = np.zeros(np.shape(mat))
    for i in range(len(new)):
        new[i][0] = np.sqrt(mat[i][0]**2+mat[i][1]**2+mat[i][2]**2)
        new[i][1] = np.arccos(mat[i][2]/np.sqrt(mat[i][0]**2+mat[i][1]**2+mat[i][2]**2))
        new[i][2] = np.arctan2(mat[i][1],mat[i][0])
    return new



#####----- DEFINING BASIS FUNCTIONS  -------##

l,m,r,theta,phi = symbols("l m r theta phi")

Etambig = (factorial(l-1)*(-2)**(Abs(m)))*(r**l)*assoc_legendre(l,Abs(m),cos(theta))*cos(m*phi)/(factorial(l+Abs(m)))
Etamsmall = (factorial(l-1)*(-2)**(Abs(m)))*(r**l)*assoc_legendre(l,Abs(m),cos(theta))*sin(Abs(m)*phi)/(factorial(l+Abs(m)))

def field_functions(a,b):  #arguments are l,m of field
    if b>=0: 
        radialexpression = diff(Etambig.subs([(l,a+1),(m,b)]),r)
        polarexpression = (1/r)*diff(Etambig.subs([(l,a+1),(m,b)]),theta)
        azimuthexpression = (1/(r*sin(theta)))*diff(Etambig.subs([(l,a+1),(m,b)]),phi)
    else:
        radialexpression = diff(Etamsmall.subs([(l,a+1),(m,b)]),r)
        polarexpression = (1/r)*diff(Etamsmall.subs([(l,a+1),(m,b)]),theta)
        azimuthexpression = (1/(r*sin(theta)))*diff(Etamsmall.subs([(l,a+1),(m,b)]),phi)
    # change coordinates to Cartesian 
    xexpression = sin(theta)*cos(phi)*radialexpression+cos(theta)*cos(phi)*polarexpression-sin(phi)*azimuthexpression
    yexpression = sin(theta)*sin(phi)*radialexpression+cos(theta)*sin(phi)*polarexpression+cos(phi)*azimuthexpression
    zexpression = cos(theta)*radialexpression-sin(theta)*polarexpression
    xfield = lambdify([r,theta,phi],xexpression,"numpy")
    yfield = lambdify([r,theta,phi],yexpression,"numpy")
    zfield = lambdify([r,theta,phi],zexpression,"numpy")
    return xfield,yfield,zfield

###-------------------------------------------------------------###
###---------- MAKE A MATRIX HOLDING THE BASIS FUNCTIONS -----------###

def basis_matrix(maxl,total,positions):
    sphericalpos = convert_to_spherical(positions)
    xbasis = np.zeros((total,len(positions)))
    ybasis = np.zeros((total,len(positions)))
    zbasis = np.zeros((total,len(positions)))
    count = 0
    for i in range(maxl+1):
        mvals = np.arange(-(i+1),i+2)
        for j in range(len(mvals)):
            xfield,yfield,zfield = field_functions(i,mvals[j])
            xbasis[count,:] = xfield(sphericalpos[:,0],sphericalpos[:,1],sphericalpos[:,2])
            ybasis[count,:] = yfield(sphericalpos[:,0],sphericalpos[:,1],sphericalpos[:,2])
            zbasis[count,:] = zfield(sphericalpos[:,0],sphericalpos[:,1],sphericalpos[:,2])
            count += 1
    return xbasis,ybasis,zbasis

###--------------------------------------------------------------###
###---------- MAKE A VECTOR OF GLM COEFFICIENTS ------------------###

def random_field(maxl,total):  
    vec = np.zeros((1,total))
    for i in range(len(vec[0])):
        vec[0][i] = np.random.normal(0,0.25)  # Glm coefficients are picked from normal distribution with arguments (average, standard deviation)
    return vec

def random_b0_field(maxl,total):
    vec = np.zeros((1,total))
    for i in range(len(vec[0])):
        if i==1:               # this index corresponds to G00 (constant Bz field)
            vec[0][i] = np.random.normal(1,0.1)
        else:
            vec[0][i] = np.random.normal(0,0.25)
    return vec

def random_field_suppressed(maxl,total,suppression):
    lvec = []
    for i in range(maxl+1):
        for j in range(2*i+3):
            lvec.append(i)
    vec = np.zeros((1,total))
    for i in range(len(vec[0])):
        vec[0][i] = np.random.normal(0,1/(suppression**lvec[i])) # higher l coefficients are suppressed by 1/(factor to the l power)
    return vec

def b0_field_suppression(maxl,total,suppression):
    lvec = []
    for i in range(maxl+1):
        for j in range(2*i+3):
            lvec.append(i)
    vec = np.zeros((1,total))
    for i in range(len(vec[0])):
        if i==1:
            vec[0][i] = np.random.normal(1,0.1)
        else: 
            if i==0 or i==2: # these correspond to the Bx and By constant fields
                vec[0][i] = np.random.normal(0,1/1000)
            else:
                vec[0][i] = np.random.normal(0,1/(suppression**lvec[i]))
    return vec

###--------------------------------------------------------------------------###
###---------- PRODUCING A FIELD IS NOW A MARIX OPERATION {Bi} = G*{Mi} -----------###

def produce_field(G,xb,yb,zb):    # arguments are Glm vector, xbasis, ybasis, zbasis
    Bx = np.matmul(G,xb)
    By = np.matmul(G,yb)
    Bz = np.matmul(G,zb)
    return Bx,By,Bz

###-----------------------------------------------------------------------###
###--------- ADDING GAUSSIAN NOISE TO FIELD ----------------------------###

def add_noise(B,width):    # arguments are one of the B vectors (Bx,By,Bz) and width of noise
    B[0] = B[0] + np.random.normal(0,width,len(B[0]))
    return B

###--------------------------------------------------------------------------###
###------- WRAPPER FUNCTION THAT DOES IT ALL -----------------------------###

def gen_field(maxl,positions,noisewidth,suppression):
    lvals = []
    mvals = []
    for i in range(maxl+1):
        ms = np.arange(-(i+1),i+2)
        for j in range(len(ms)):
            lvals.append(i)
            mvals.append(ms[j])
    totalf = 0
    for i in range(maxl+1):
        totalf += 2*i+3
    #G = random_field(maxl,totalf)
    #G = random_b0_field(maxl,totalf)
    #G = random_field_suppressed(maxl,totalf,suppression)
    G = b0_field_suppression(maxl,totalf,suppression)
    xbasis,ybasis,zbasis = basis_matrix(maxl,totalf,positions)
    Bx,By,Bz = produce_field(G,xbasis,ybasis,zbasis)
    Bx,By,Bz = add_noise(Bx,noisewidth),add_noise(By,noisewidth),add_noise(Bz,noisewidth) # add Gaussian noise if you want
    return G,Bx,By,Bz,lvals,mvals

###-----------------------------------------------------------------------------###
###---------------- STUFF USER HAS TO DEFINE -----------------------------------###

maxl = 8   # maximum l-th order of field
noise = 0   # width of gaussian noise
suppression = 20

filename = "{}".format(sys.argv[1])

df = pd.read_csv(filename,header=0)

positions = np.zeros((len(df),3))
positions[:,0] = df['x']
positions[:,1] = df['y']
positions[:,2] = df['z']

G,Bx,By,Bz,ls,ms = gen_field(maxl,positions,noise,suppression)

with open("testG.csv",'w+') as f:
    f.write('l,m,G\n')
    for i in range(len(G[0])):
        f.write('{},{},{}\n'.format(ls[i],ms[i],G[0][i]))
f.close()

with open("testfield.csv",'w+') as f:
    f.write('x,y,z,Bx,By,Bz\n')
    for i in range(len(Bx[0])):
        f.write('{},{},{},{},{},{}\n'.format(positions[i][0],positions[i][1],positions[i][2],Bx[0][i],By[0][i],Bz[0][i]))
f.close()













