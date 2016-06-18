"""
Calculates decay widths and branching ratios for s, as well as the scalar mixing angle.
Program for the Master Thesis 'New Applications of the Coleman-Weinberg model'
Jorge Alda Gallo
"""

import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt

Ns = int(raw_input("Ns = "))
mz = 91.1876 # Z boson mass 
mt = 173.21 # top quark mass
mw = 80.385 # W boson mass
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

def funcF(x):
    """
        Function for the gluon fusion cross-section
    """
    if x< 1.0:
        return np.arcsin(np.sqrt(x))**2
    else:
        return -0.25*(np.log((1+np.sqrt(1-1.0/x) )/(1-np.sqrt(1-1.0/x))) -1.0j*pi )**2

brtt = []
brWW = []
brZZ = []
brHH = []
totwidth = []
angle = []
fang = open('angles.txt', 'w')
msspace = np.linspace(0.4, 10, 800)
for msTeV in msspace:
    ms = 1000*msTeV
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
        
        # Fix the couplings \lambda_s and \lambda_{hs} at \mu=ms, and calculate their beta functions
        coup[3] = ls
        coup[4] = lhs
        coup[5] = g4_0
        [bgs, byt, blh, bls, blhs, bg4] = betalambda(coup, t, [6,1])
    
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
    
	    # Fix the couplings \lambda_s and \lambda_{hs} at \mu=ms, and calculate their beta functions
        coup[3] = ls
        coup[4] = lhs
        coup[5] = g4_0
        [bgs, byt, blh, bls, blhs, bg4] = betalambda(coup, t, [5,1])
        
    else:    
	    # Fix gs at \mu=mz with 5 active flavors, and then run until \mu=vs
        coup = [gs,0,0,0,0, 0]
        dt = (np.log(vs) - np.log(mz))/20
        for t in range(0,20):
            RK4(coup, t, betalambda, dt, [5,0])

    	# Fix the couplings \lambda_s and \lambda_{hs} at \mu=ms, and calculate their beta functions
        coup[3] = ls
        coup[4] = lhs
        coup[5] = g4_0
        [bgs, byt, blh, bls, blhs, bg4] = betalambda(coup, t, [5,1])
    
    # Calculate mass eigenvalues, mixing angles and f(\tau) and save them to file
    mhs2 = vh/np.sqrt(2)*((2*lhs + blhs)*vs + blh*vh*vh/vs )
    m1 = np.sqrt( 0.5*(mh**2 + ms**2 + np.sqrt( (mh**2 - ms**2)**2 + 4*mhs2**2  )  ) )
    m2 = np.sqrt( 0.5*(mh**2 + ms**2 - np.sqrt( (mh**2 - ms**2)**2 + 4*mhs2**2  )  ) )
    theta = 0.5*np.arctan(2*mhs2/(ms**2-mh**2))
    angle.append(theta)  
    tau = m1**2/(4*mt**2)
    factor = abs(( tau+(tau-1.0)*funcF(tau) )/tau**2  )**2
    fang.write('{0}\t{1}\t{2}\n'.format(m1, theta, factor))  
    
    # Decay width into two Higgs
    av = (2*lhs + blhs)*vs/np.sqrt(2)
    bv = (2*lhs+3*blhs)/4
    cv = blh/(np.sqrt(2)*vs)
    ev = -blh/(4*vs**2)
    fv = np.sqrt(2)*vs*(ls+bls)
    kappa = (av+6*cv*vh**2)*np.cos(theta)**3  +  (-4*np.sqrt(2)*vh*bv - 8*np.sqrt(2)*vh**3*ev)*np.cos(theta)**2*np.sin(theta)  +  (6*fv-2*av-12*cv*vh**2)*np.cos(theta)*np.sin(theta)  +  (2*np.sqrt(2)*vh*bv +4 *np.sqrt(2)*vh**3*ev)*np.sin(theta)**3
    decayHH = kappa**2/(8*pi*ms)*np.sqrt(1-4*mh**2/ms**2)

    # Higgs-like Decay widths 
    decaytt = 3* mt**2*m1**2*np.sin(theta)**2/(8*pi*vh**2*m1)*(1-4*mt**2/m1**2)*np.sqrt(1-4*mt**2/m1**2)
    decayWW = np.sin(theta)**2*mw**4/(4*pi*m1*vh**2)*(1-4*mw**2/m1**2)**0.5*(3+m1**4/(4*mw**4)-m1**2/mw**2)
    decayZZ = np.sin(theta)**2*mz**4/(8*pi*m1*vh**2)*(1-4*mz**2/m1**2)**0.5*(3+m1**4/(4*mz**4)-m1**2/mz**2)
    
    # Total decay width and branching ratios
    decaytot = decaytt + decayWW + decayZZ + decayHH
    totwidth.append(decaytot)
    brtt.append(decaytt/decaytot)
    brWW.append(decayWW/decaytot)
    brZZ.append(decayZZ/decaytot)
    brHH.append(decayHH/decaytot)

# Plotting
plt.close()

plt.figure(1)
plt.plot(msspace, totwidth, 'k', linewidth=3, label=r'$\Gamma(\sigma \to \mathrm{all})$')
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
plt.xlabel(r'$m_\sigma\ (\mathrm{TeV})$', fontsize=25)
plt.ylabel(r'$\Gamma\ (\mathrm{GeV})$', fontsize=25)
plt.yscale('log')

plt.figure(2)
plt.plot(msspace, angle, 'r', linewidth=2.5)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
plt.xlabel(r'$m_\sigma\ (\mathrm{TeV})$', fontsize=25)
plt.ylabel(r'$\theta$', fontsize=25)
plt.yscale('log')

plt.figure(3)
plt.plot(msspace, brHH, linewidth=2.5, label=r'$\mathrm{BR}(s \to hh)$')
plt.plot(msspace, brtt, linewidth=2.5, label=r'$\mathrm{BR}(s \to t\bar{t})$')
plt.plot(msspace, brWW, linewidth=2.5, label=r'$\mathrm{BR}(s \to WW)$')
plt.plot(msspace, brZZ, linewidth=2.5, label=r'$\mathrm{BR}(s \to ZZ)$')
plt.xlabel(r'$m_\sigma\ (\mathrm{TeV})$', fontsize=25)
plt.legend(loc=3)
plt.yscale('log')
plt.ylim((1e-5, 1.5))

plt.show()
fang.close()   
