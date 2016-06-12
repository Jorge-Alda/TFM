import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from operator import itemgetter

Ns = 150
mz = 91.187
mt = 173.34
ms = 2527.0
mh = 125.09
vh = 246.0
gs = 1.2177
yt = 0.9369
twoloops = 1
Nf = 0
g4_0 = 0.0

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

    
#Running of the three lambda's plus the Yukawa coupling
def betalambda(y, t, pars):
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
#ls = -16*pi**2/(4+Ns+8*mh**4/(ms**4))
#lhs = 4*mh**2/ms**2*ls
vs = ms/np.sqrt(-8*ls)
print(ls)
print(lhs)
print(vs)

#First we calculate gs(vs) given gs(mz)

if vs > mt:
    coup = [gs,0,0,0,0, 0]
    dt = (np.log(mh) - np.log(mz))/20
    for t in range(0,20):
        RK4(coup, t, betalambda, dt, [5,0]) 
    
    coup[2] = lh
    dt = (np.log(mt) - np.log(mh))/200
    for t in np.linspace(np.log(mh), np.log(mt), 200):
        RK4(coup, t, betalambda, dt, [5,0])
    
    coup[1] = yt
    dt = (np.log(vs) - np.log(mt))/100
    for t in np.linspace(np.log(mt), np.log(vs), 100):
        RK4(coup, t, betalambda, dt, [6,0]) 
    
    #Now gs and lh @ mh
    coup[3] = ls
    coup[4] = lhs
    coup[5] = g4_0
    tarray=[]
    lharray=[]
    lsarray=[]
    lhsarray=[]
    g4array=[] 
    dist = []
    dt = (20*np.log(10) - np.log(vs))/5000
    [bgs, byt, blh, bls, blhs, bg4] = betalambda(coup, t, [6,1])
    for t in np.linspace(np.log(vs), 20*np.log(10), 5000):
        RK4(coup, t, betalambda, dt, [6,1])
        tarray.append(t/np.log(10))
        lharray.append(coup[2])
        lsarray.append(coup[3])
        lhsarray.append(coup[4])
        g4array.append(coup[5])
        dist.append((np.abs(coup[2]-coup[3])+np.abs(coup[2]-coup[4])+np.abs(coup[3]-coup[4]))/3.0 )

elif vs > mh:
    coup = [gs,0,0,0,0, 0]
    dt = (np.log(mh) - np.log(mz))/20
    for t in range(0,20):
        RK4(coup, t, betalambda, dt, [5,0]) 
    
    coup[2] = lh
    dt = (np.log(vs) - np.log(mh))/200
    for t in np.linspace(np.log(mh), np.log(vs), 200):
        RK4(coup, t, betalambda, dt, [5,0])

    coup[3] = ls
    coup[4] = lhs
    coup[5] = g4_0
    tarray=[]
    lharray=[]
    lsarray=[]
    lhsarray=[] 
    g4array=[]
    dist = []
    dt = (np.log(mt) - np.log(vs))/200
    [bgs, byt, blh, bls, blhs, bg4] = betalambda(coup, t, [6,1])
    for t in np.linspace(np.log(vs), np.log(mt), 200):
        RK4(coup, t, betalambda, dt, [5,1])
        tarray.append(t/np.log(10))
        lharray.append(coup[2])
        lsarray.append(coup[3])
        lhsarray.append(coup[4])
        g4array.append(coup[5])
        dist.append((np.abs(coup[2]-coup[3])+np.abs(coup[2]-coup[4])+np.abs(coup[3]-coup[4]))/3.0 )
    
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
    coup = [gs,0,0,0,0, 0]
    dt = (np.log(vs) - np.log(mz))/20
    for t in range(0,20):
        RK4(coup, t, betalambda, dt, [5,0])

    coup[3] = ls
    coup[4] = lhs
    coup[5] = g4_0
    tarray=[]
    lharray=[]
    lsarray=[]
    lhsarray=[] 
    g4array=[]
    dist = []
    dt = (np.log(mh) - np.log(vs))/200
    [bgs, byt, blh, bls, blhs, bg4] = betalambda(coup, t, [6,1])
    for t in np.linspace(np.log(vs), np.log(mh), 200):
        RK4(coup, t, betalambda, dt, [5,1])
        tarray.append(t/np.log(10))
        lharray.append(coup[2])
        lsarray.append(coup[3])
        lhsarray.append(coup[4])
        g4array.append(coup[5])
        dist.append((np.abs(coup[2]-coup[3])+np.abs(coup[2]-coup[4])+np.abs(coup[3]-coup[4]))/3.0 ) 
    
    coup[2] = lh
    dt = (np.log(mt) - np.log(mh))/200
    for t in np.linspace(np.log(mh), np.log(mt), 200):
        RK4(coup, t, betalambda, dt, [5,1])
        tarray.append(t/np.log(10))
        lharray.append(coup[2])
        lsarray.append(coup[3])
        lhsarray.append(coup[4])
        g4array.append(coup[5])
        dist.append((np.abs(coup[2]-coup[3])+np.abs(coup[2]-coup[4])+np.abs(coup[3]-coup[4]))/3.0 ) 
    
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


print tarray[min(enumerate(dist), key=itemgetter(1))[0]]
maxl = max(max(lharray), -min(lharray), max(lsarray), -min(lsarray), max(lhsarray), -min(lhsarray) )
print(maxl)
plt.close()
plt.plot(tarray, lharray, linewidth=2.5, label=r'\lambda_h')
plt.plot(tarray, lsarray, linewidth=2.5, label=r'\lambda_s')
plt.plot(tarray, lhsarray, linewidth=2.5, label = r'\lambda_{hs}')
plt.plot(tarray, g4array, linewidth=2.5, label = r'g_4')
plt.ylim((-5, 5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=25)
plt.xlabel(r'\log_{10}(\mu/ 1 \mathrm{GeV})', fontsize=25)
plt.legend(loc=2)
#plt.plot(tarray, dist)
plt.show()

mhs2 = vh/np.sqrt(2)*((2*lhs + blhs)*vs + blh*vh*vh/vs )
m1 = np.sqrt( 0.5*(mh**2 + ms**2 + np.sqrt( (mh**2 - ms**2)**2 + 4*mhs2**2  )  ) )
m2 = np.sqrt( 0.5*(mh**2 + ms**2 - np.sqrt( (mh**2 - ms**2)**2 + 4*mhs2**2  )  ) )
print(str(m1)+'GeV\t' + str(m2) +'GeV' )
theta = -0.5*np.arctan(2*mhs2/(ms**2-mh**2))
print(theta)

