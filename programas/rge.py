import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

Ns = 10
Nf = 54
ms = 440/Ns**0.25
mz = 91.187
mt = 173.34

def RK4(x, t, f, delta, pars): #4th order Runge-Kutta integrator
	k1 = []
	k2 = []
	k3 = []
	k4 = []
	x0 = []
	
	xk = f(x, t, pars)
	for s in range(0, len(x)):
		k1.append(xk[s] * delta)
		x0.append(x[s] + 0.5*k1[s])

	xk = f(x0, t+delta/2, pars)
	for s in range(0, len(x)):
		k2.append(xk[s] * delta)
		x0[s] = x[s] + 0.5*k2[s]
	
	xk = f(x0, t+delta/2, pars)
	for s in range(0, len(x)):
		k3.append(xk[s] * delta)
		x0[s] = x[s] + k3[s]

	xk = f(x0, t+delta, pars)
	for s in range(0, len(x)):
		k4.append(xk[s] * delta)

	for s in range(0, len(x)):
		x[s] = x[s] + (k1[s] + 2*k2[s] + 2*k3[s] + k4[s])/6.0

# Gauge coupling runnings (analytic solution)
def d(b):
    return b/(8*pi**2)

def c(b, g0, m0):
    return 1/g0**2 - d(b)*np.log(m0)

def g1(t):
    return 1/np.sqrt(c(-41.0/6.0, 0.343,mz)+t*d(-41.0/6.0) )
    
def g2(t):
    return 1/np.sqrt(c(19.0/6.0, 0.638,mz)+t*d(19.0/6.0) )

def g3(t):
    return 1/np.sqrt(c(7, 1.217,mz)+t*d(7) )
    
def g4(t):
    b4 = 11.0/3.0*Ns - 2.0/3.0*Nf +1.0/6.0
    return 1/np.sqrt(c(b4, 2.4,ms)+t*d(b4) )

#Running of the Yukawa coupling alone
def betayt(y, t, pars):
    yt = y[0]
    beta_yt = yt*(4.5*yt**2-8*g3(t)**2 -9.0/4.0*g2(t)**2 - 17.0/12.0*g1(t)**2)/(16*pi**2)
    return [beta_yt]
    
#Running of the Yukawa coupling and Gamma
def betagamma(y, t, pars):
    yt, gamma = y
    beta_yt = yt*(4.5*yt**2-8*g3(t)**2 -9.0/4.0*g2(t)**2 - 17.0/12.0*g1(t)**2)/(16*pi**2)
    beta_gamma = 3*yt**2
    return [beta_yt, beta_gamma]

#Running of the three lambda's plus the Yukawa coupling
def betalambda(y, t, pars):
    yt, lh, ls, lhs = y
    beta_yt = yt*(4.5*yt**2-8*g3(t)**2 -9.0/4.0*g2(t)**2 - 17.0/12.0*g1(t)**2)/(16*pi**2)
    beta_lh = (24*lh**2 + Ns*lhs**2 -6*yt**4+12*lh*yt**2)/(16*pi**2)
    beta_ls = (4*(4+Ns)*ls**2 + 2*lhs**2-6*(Ns**2-1)/Ns * ls*g4(t)**2 +3.0/4.0*(Ns**3+Ns**2-4*Ns+2)/Ns**2*g4(t)**4  )/(16*pi**2)
    beta_lhs = lhs*(4*lhs+12*lh+4*(Ns+1)*ls - 3 *(Ns**2-1)/Ns*g4(t)**2 + 6*yt**2)/(16*pi**2)
    return [beta_yt, beta_lh, beta_ls, beta_lhs]

#First we calculate yt(mz) given yt(mt)
yt = [0.9369]
dt = (np.log(mz) - np.log(mt))/20
for t in range(0,20):
    RK4(yt, t, betayt, dt, 0) 

#Now yt(ms) and Gamma(ms)
yt.append(0.0)
dt = (np.log(ms) - np.log(mz))/100
for t in range(0,100):
    RK4(yt, t, betagamma, dt, 0)

#And finally, for the coupling constants
dlh = 0.5*40/(16*pi**2)*(-1.5+2*yt[1])
lh = -1.0/16.0*np.exp(-4.0*yt[1])-dlh
yt = [yt[0], lh, 0, np.sqrt(40/Ns)]

tarray=[]
lharray=[]
lsarray=[]
lhsarray=[]
g4array=[]   
dt = (20*np.log(10) - np.log(ms))/1000
for t in np.linspace(np.log(ms), 20*np.log(10), 1000):
    RK4(yt, t, betalambda, dt, 0) 
    tarray.append(t/np.log(10))
    lharray.append(yt[1])
    lsarray.append(yt[2])
    lhsarray.append(yt[3])
    g4array.append(g4(t))


plt.plot(tarray, g4array, linewidth=2.5, label=r'g_4')
plt.plot(tarray, lharray, linewidth=2.5, label=r'\lambda_h')
plt.plot(tarray, lsarray, linewidth=2.5, label=r'\lambda_s')
plt.plot(tarray, lhsarray, linewidth=2.5, label = r'\lambda_{hs}')
plt.ylim((-0.5, 3))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'\log_{10}(\mu/ 1 \mathrm{GeV})', fontsize=16)
plt.legend(loc=1)
plt.show()