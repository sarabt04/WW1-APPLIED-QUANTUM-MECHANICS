########################  3-ELECTRON ATOMS  #################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# 1. DISCRETITZATION  ---------------------------------------------------------
R=20
N=2000
delta_r=R/N
r=np.linspace(delta_r,R+delta_r,N)

# 2. BUILD THE MATRICES  ------------------------------------------------------
#Kinetic energy
K_old=np.zeros((N, N))
for i in range(0, N):
    if i > 0:
        K_old[i,i-1]=1
        
    K_old[i, i]=-2
    
    if i < N-1:
        K_old[i,i+1]=1

K=-1/(2*delta_r**2)*K_old

#Effective potential
def Veff(r, Z, l):
    coulumb=-Z/r
    Vavg=2/r-4/3*(Z+1/r)*np.exp(-2*Z*r)-2/3*(1/r+3*Z/4+Z**2*r/4+Z**3*r**2/8)*np.exp(-Z*r)
    ang=l*(l+1)/(2*r**2)
    return coulumb+Vavg+ang

V1=np.zeros((N,N))
for i in range(0,N):
    V1[i, i]=Veff(r[i],Z=3,l=0)

V2=np.zeros((N,N))
for i in range(0,N):
    V2[i, i]=Veff(r[i],Z=3,l=1)
    
      
#Final matrix
As=K+V1
Ap=K+V2

# 3. SOLUTION THE EIGENVALUES PROBLEM  ----------------------------------------
#E, u = np.linalg.eigh(A)
Es, us = eigh(As)
Ep, u2 = eigh(Ap)

print("First energies:", Es[:10])
print("Energies:")
print("1s:", Es[0])
print("2s:", Es[1])
print("2p:", Ep[0])


u0=us[:,0]/np.sqrt(delta_r) #normalized

# 4. PLOT  --------------------------------------------------------------------
plt.plot(r,u0, '--')  
plt.title("Wavefunction")
plt.xlabel("r")
plt.ylabel("u (r)")  
plt.show()