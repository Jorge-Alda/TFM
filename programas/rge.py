"""
Calculates the running of the coupling constants
Program for the Master Thesis 'New Applications of the Coleman-Weinberg model'
Jorge Alda Gallo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

Ns = int(raw_input("Ns = "))
ms = int(raw_input("ms = "))
mz = 91.1876 # Z boson mass 
mt = 173.21 # top quark mass
mh = 125.7 # Higgs boson mass
vh = 246.0 # Higgs vev
gs = 1.2177 # Strong coupling at \mu = mz
yt = 0.9369 # Top Yukawa coupling at \mu = mt
twoloops = 1 # 1 if the two-loops beta functions are used (when available), 0 otherwise
Nf = 0 # Number of fermions in the fundamental repr of SU(NS)
g4_0 = 0.0 # Coupling constant for the SU(NS) gauge group (0 for a global symmetry)

def RK4(x, t, f, delta, pars):
    """ 
        4th order Runge-Kutta integrator
    """
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

    
def betalambda(y, t, pars):
    """
        some beta functions
    """
    gs, yt, lh, ls, lhs, g4 = y
    nf, calch = pars
    beta_gs = -(11.0-2.0/3.0*nf)*gs**3/(4*pi)**2-(102.0-38.0/3.0*nf)*gs**5/(4*pi)**4*twoloops
    beta_yt = yt*(4.5*yt**2-8*gs**2 )/(4*pi)**2 +yt*(-4.5*yt**4+1.5*lh**2-6.0*lh*yt**2+36.0*gs**3*yt**2-324.0/3.0*gs**2)/(4*pi)**4*twoloops
    if calch ==1:
        beta_lh = (24*lh**2 + Ns*lhs**2 -6*yt**4+12*lh*yt**2)/(16*pi**2)+(-312.0*lh**4+80.0*lh*gs**2*yt**2-144.0*lh**2*yt**2-3.0*lh*yt**4-32.0*gs**3*yt**4+30.0*yt**6)/(4*pi)**4*twoloops
    else:
        beta_lh = 0
    beta_ls = (4*(4+Ns)*ls**2 + 2*lhs**2 + 0.75*(Ns**3 + Ns**2 -4*Ns+2)/Ns**2 * g4**4 -6*(Ns**2-1)/Ns *g4**2*ls  )/(16*pi**2)
    beta_lhs = lhs*(4*lhs+12*lh+4*(Ns+1)*ls  + 6*yt**2-3*(Ns**2-1)/Ns*g4**4)/(16*pi**2)
    beta_g4  = -g4**3*(11.0/3.0*Ns - 2.0/3.0*Nf - 1.0/6.0)/(16*pi**2)

    return [beta_gs, beta_yt, beta_lh, beta_ls, beta_lhs, beta_g4]

a = 8 + 2*Ns+16.0*mh**4/ms**4
b = 32*pi**2 - 3*g4_0**2*(Ns**2-1)/Ns
c = 3.0/8.0 * (Ns**3+Ns**2-4*Ns+2)/Ns**2 * g4_0**4 
ls = (-b-np.sqrt(b**2-4*a*c) )/(2*a)
lhs = - np.sqrt(-32*pi**2*ls - 2*(4+Ns)*ls**2 + 3*g4_0**2*ls*(Ns**2-1)/Ns +3.0/8.0 * (Ns**3+Ns**2-4*Ns+2)/Ns**2 * g4_0**4 )
lh = mh**2/(2*vh**2)
vs = ms/np.sqrt(-8*ls)


if vs > mt:
    # Fix gs at \mu=mz with 5 active flavors, and then run until \mu=mh
    coup = [gs,0,0,0,0, 0]
    dt = (np.log(mh) - np.log(mz))/20
    for t in range(0,20):
        RK4(coup, t, betalambda, dt, [5,0]) 
    
    # Fix the coupling \lambda_h at \mu=mz, and then run until \mu=mt
    coup[2] = lh
    dt = (np.log(mt) - np.log(mh))/200
    for t in np.linspace(np.log(mh), np.log(mt), 200):
        RK4(coup, t, betalambda, dt, [5,0])
    
    # Fix the Yukawa yt at \mu= mt, and the run until \mu=vs
    coup[1] = yt
    dt = (np.log(vs) - np.log(mt))/100
    for t in np.linspace(np.log(mt), np.log(vs), 100):
        RK4(coup, t, betalambda, dt, [6,0]) 
    
    # Fix the couplings \lambda_s and \lambda_{hs} at \mu=ms, and run until \mu=1e20 GeV
    coup[3] = ls
    coup[4] = lhs
    coup[5] = g4_0
    tarray=[]
    lharray=[]
    lsarray=[]
    lhsarray=[]
    g4array=[] 
    dt = (20*np.log(10) - np.log(vs))/5000
    [bgs, byt, blh, bls, blhs, bg4] = betalambda(coup, t, [6,1])
    for t in np.linspace(np.log(vs), 20*np.log(10), 5000):
        RK4(coup, t, betalambda, dt, [6,1])
        tarray.append(t/np.log(10))
        lharray.append(coup[2])
        lsarray.append(coup[3])
        lhsarray.append(coup[4])
        g4array.append(coup[5])

elif vs > mh:
    # Fix gs at \mu=mz with 5 active flavors, and then run until \mu=mh
    coup = [gs,0,0,0,0, 0]
    dt = (np.log(mh) - np.log(mz))/20
    for t in range(0,20):
        RK4(coup, t, betalambda, dt, [5,0]) 
    
    # Fix the coupling \lambda_h at \mu=mh, and then run until \mu=vs
    coup[2] = lh
    dt = (np.log(vs) - np.log(mh))/200
    for t in np.linspace(np.log(mh), np.log(vs), 200):
        RK4(coup, t, betalambda, dt, [5,0])

    # Fix the couplings \lambda_s and \lambda_{hs} at \mu=ms, and run until \mu=mt
    coup[3] = ls
    coup[4] = lhs
    coup[5] = g4_0
    tarray=[]
    lharray=[]
    lsarray=[]
    lhsarray=[] 
    g4array=[]
    dt = (np.log(mt) - np.log(vs))/200
    [bgs, byt, blh, bls, blhs, bg4] = betalambda(coup, t, [6,1])
    for t in np.linspace(np.log(vs), np.log(mt), 200):
        RK4(coup, t, betalambda, dt, [5,1])
        tarray.append(t/np.log(10))
        lharray.append(coup[2])
        lsarray.append(coup[3])
        lhsarray.append(coup[4])
        g4array.append(coup[5])
    
    # Fix the Yukawa yt at \mu= mt, and the run until \mu=1e20
    coup[1] = yt
    dt = (20*np.log(10) - np.log(mt))/1000
    for t in np.linspace(np.log(mt), 20*np.log(10), 1000):
        RK4(coup, t, betalambda, dt, [6,1]) 
        tarray.append(t/np.log(10))
        lharray.append(coup[2])
        lsarray.append(coup[3])
        lhsarray.append(coup[4])
        g4array.append(coup[5])
        dist.append((np.abs(coup[2]-coup[3])+np.abs(coup[2]-coup[4])+np.abs(coup[3]-coup[4]))/3.0 )
    
    
else:    
    # Fix gs at \mu=mz with 5 active flavors, and then run until \mu=vs
    coup = [gs,0,0,0,0, 0]
    dt = (np.log(vs) - np.log(mz))/20
    for t in range(0,20):
        RK4(coup, t, betalambda, dt, [5,0])

    # Fix the couplings \lambda_s and \lambda_{hs} at \mu=ms, and then run until \mu=mh
    coup[3] = ls
    coup[4] = lhs
    coup[5] = g4_0
    tarray=[]
    lharray=[]
    lsarray=[]
    lhsarray=[] 
    g4array=[]
    dt = (np.log(mh) - np.log(vs))/200
    [bgs, byt, blh, bls, blhs, bg4] = betalambda(coup, t, [6,1])
    for t in np.linspace(np.log(vs), np.log(mh), 200):
        RK4(coup, t, betalambda, dt, [5,1])
        tarray.append(t/np.log(10))
        lharray.append(coup[2])
        lsarray.append(coup[3])
        lhsarray.append(coup[4])
        g4array.append(coup[5]) 
    
    # Fix the coupling \lambda_h at \mu=mh, and then run until \mu=mt
    coup[2] = lh
    dt = (np.log(mt) - np.log(mh))/200
    for t in np.linspace(np.log(mh), np.log(mt), 200):
        RK4(coup, t, betalambda, dt, [5,1])
        tarray.append(t/np.log(10))
        lharray.append(coup[2])
        lsarray.append(coup[3])
        lhsarray.append(coup[4])
        g4array.append(coup[5])
    
    # Fix the Yukawa yt at \mu= mt, and the run until \mu=1e20 GeV
    coup[1] = yt
    dt = (20*np.log(10) - np.log(mt))/1000
    for t in np.linspace(np.log(mt), 20*np.log(10), 1000):
        RK4(coup, t, betalambda, dt, [6,1])
        tarray.append(t/np.log(10))
        lharray.append(coup[2])
        lsarray.append(coup[3])
        lhsarray.append(coup[4])
        g4array.append(coup[5])  

# Plotting
plt.close()
plt.plot(tarray, lharray, linewidth=2.5, label=r'\lambda_h')
plt.plot(tarray, lsarray, linewidth=2.5, label=r'\lambda_s')
plt.plot(tarray, lhsarray, linewidth=2.5, label = r'\lambda_{hs}')
plt.ylim((-5, 5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=25)
plt.xlabel(r'\log_{10}(\mu/ 1 \mathrm{GeV})', fontsize=25)
plt.legend(loc=2)
plt.show()


