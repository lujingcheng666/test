import numpy as np
from numpy import pi

import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import copy

from extend_BC import extend, extend2D

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

Nx = 100;
Nv = 200;

TT = 70;

xmin = 0.; xmax = 10*pi;
x = np.linspace(xmin, xmax, Nx+1);
hx = x[1]-x[0];
Nx = len(x);

ion = 1.;

test_case = 'ts';

if test_case == 'ts': 
    Sig = 1;
    w1 = 0.5;
    w2 = 1-w1;
    u1 = 2.4;
    u2 = -2.4;
    fv = lambda x,v: w1/np.sqrt(2*pi)/Sig*np.exp(-(v-u1)**2/2/Sig**2)+\
        w2/np.sqrt(2*pi)/Sig*np.exp(-(v-u2)**2/2/Sig**2);

if test_case == 'bt':
    # Sig1 = 1;
    # Sig2 = np.sqrt(0.5);
    # w1 = 0.8;
    # w2 = 0.2;
    # u1 = -2;
    # u2 = 3.5;
    
    Sig1 = 1;
    Sig2 = 0.5;
    w1 = 0.9;
    w2 = 0.1;
    u1 = -2;
    u2 = 3.5;
    fv1 = lambda v: w1/np.sqrt(2*pi)/Sig1*np.exp(-(v-u1)**2/2/Sig1**2);
    fv2 = lambda v: w2/np.sqrt(2*pi)/Sig2*np.exp(-(v-u2)**2/2/Sig2**2);
    fv = lambda x,v: fv1(v)+fv2(v);

if test_case == 'bgk':
    w = 2*pi/(max(x)-min(x));
    
    A = 0.2;
    phi_x = A*np.sin(w*x);                          
    rho = np.exp(-phi_x);                       
    ion = rho-A*w**2 *np.sin(w*x);

    phieq = lambda x: A*np.sin(w*x);                            
    fv = lambda x,v: 1/np.sqrt(2*pi)*np.exp(-v**2/2-phieq(x));

vmin = -8; vmax = 8;

v = np.linspace(vmin, vmax, Nv+1);
hv = v[1]-v[0];
Nv = len(v);
X, V = np.meshgrid(x, v, indexing='xy');

w = 2*pi/(max(x)-min(x));
r0 = 1;

feq = r0*fv(X,V);

if test_case == 'ts':
    eps = 1e-3;
    f0 = (1+eps*np.cos(w*x))*feq;
    # eps = 1e-3;
    # f0 = (1-eps*np.sin(w*x)+2*eps*np.cos(2*w*x))*feq;
elif test_case == 'bt':
    eps = 3e-3;
    f0 = feq+eps*np.sin(w*x)*fv2(V);
elif test_case == 'bgk':
    eps = 1e-2;
    f0 = (1+eps*np.cos(w*x))*feq;


feedback_noise = 0e-4;

def interp2(X, Y, Z, Xp, Yp, mode='bilinear'):
    x = np.ascontiguousarray(X[0, :])
    y = np.ascontiguousarray(Y[:, 0])
    Z = np.ascontiguousarray(Z)
    if mode == 'bilinear':
        interpolator = sp.interpolate.RegularGridInterpolator((y, x), Z, bounds_error=False, fill_value=None);
        Zp = interpolator((Yp.reshape(-1), Xp.reshape(-1))).reshape(Xp.shape);
    elif mode == 'cubic':
        spline = sp.interpolate.RectBivariateSpline(y, x, Z, kx=3, ky=3)
        Zp = spline.ev(Yp.reshape(-1), Xp.reshape(-1)).reshape(Xp.shape) 
    return Zp


'''
VP solver
'''
def electric(x, h, r, ion):
    L = max(x)-min(x);
    xp = (x-min(x)).reshape(1, -1);
    dxG = (xp.T <= xp)*(1-xp/L)+(xp.T > xp)*(-xp/L);
    E = -np.sum(dxG*((r-ion).reshape(1, -1))*h, 1);
    dxE = (E[1:]-E[:-1])/h;
    cor = -(E[-1]-E[0])/(max(x)-min(x));
    E[1:] = E[0]+np.cumsum(dxE+cor, axis=0)*h;
    return E


def VPSL(hx, hv, k, x, v, f, H, ion):
    Nx = len(x);
    Nv = len(v);
    X, V = np.meshgrid(x, v, indexing='xy');
    m = 20;
    BCl = 'P'; BCr = 'P'; BCd = 'D'; BCu = 'D'
    # stage 1 convection
    Xp = X-k/2*V;
    Vp = V;
    Xe, Ve, fe = extend2D(X, V, f, hx, hv, m, BCl,
                          f[:, 0], BCr, f[:, -1], BCd, f[0, :], BCu, f[-1, :]);
    f1 = interp2(Xe, Ve, fe, Xp, Vp);
    # stage 2 convection
    r = np.sum(f1*hv, 0);
    E = electric(x, hx, r, ion);
    Ftot = E+H;

    Xp = X;
    Vp = V-k*Ftot;
    Xe, Ve, fe1 = extend2D(X, V, f1, hx, hv, m, BCl,
                           f1[:, 0], BCr, f1[:, -1], BCd, f1[0, :], BCu, f1[-1, :]);
    fe1 = np.ascontiguousarray(fe1);
    f2 = interp2(Xe, Ve, fe1, Xp, Vp);
    # stage 3 convection
    Xp = X-k/2*V;
    Vp = V;
    Xe, Ve, fe2 = extend2D(X, V, f2, hx, hv, m, BCl,
                           f2[:, 0], BCr, f2[:, -1], BCd, f2[0, :], BCu, f2[-1, :]);
    f3 = interp2(Xe, Ve, fe2, Xp, Vp);

    return f3


def VP2DForward(x, v, f0, feq, Hfun, ion, TT, gam=1):
    hx = x[1]-x[0];
    hv = v[1]-v[0];

    Time = [];
    FF = [];
    Energy = [];

    time = 0;
    f = f0;
    r = np.sum(f*hv, 0);
    E = electric(x, hx, r, ion);
    energy = 1/2*sum(E**2)*hx;

    Time.append(time);
    FF.append(f);
    Energy.append(energy);

    cfl = 5;
    while time < TT:
        k = cfl*hx/max(abs(v));
        if time+k > TT:
            k = TT-time;
            
        noise = feedback_noise*np.random.randn(Nv,Nx);
        df = f-feq+noise;
        H = Hfun(x,v,df,feq,ion,gam);
        f = VPSL(hx, hv, k, x, v, f, H, ion);
        time = time+k;

        r = np.sum(f*hv, 0);
        E = electric(x, hx, r, ion);
        energy = 1/2*np.sum(E**2)*hx;

        Time.append(time)
        FF.append(f)
        Energy.append(energy)

        # print('time = ',time)

    FF = np.array(FF)
    Time = np.array(Time)
    Energy = np.array(Energy)
    Perturb = 1/2*np.sum(np.sum((FF-feq)**2*hx*hv,1),1)

    return FF, Time, Energy, Perturb

'''
cancallation based operator
'''
def Sign(x):
    sign = np.sign(x);
    sign[sign==0]=1;
    return sign
def Helmholtz_proj(x,y,H1,H2):
    Nx = len(x);
    xe,  
def Hfun(x,v,df,feq,ion,gam=1):
    hx = x[1]-x[0];
    hv = v[1]-v[0];
    Nx = len(x);
    Nv = len(v);
    
    dvfeq = np.zeros([Nv,Nx]);
    dvfeq[1:-1] = (feq[2:]-feq[:-2])/2/hv;
    dvfeq[0,:] = (feq[1,:]-feq[0,:])/hv;
    dvfeq[-1,:] = (feq[-1,:]-feq[-2,:])/hv;
    dH = gam*np.sum(df*dvfeq*hv,0);
    # dH = gam*np.sum(df**2*hv,0)/(np.sum(df*dvfeq*hv,0)+Sign(np.sum(df*dvfeq*hv,0))*1e-3)
    
    drho = np.sum(df*hv,0);
    dE = electric(x, hx, drho, 0);
    H = -dE+dH;
    return H

Hfun0 = lambda x,v,df,feq,ion,gam : np.zeros(len(x));
Fini,Time,Enini,Perturb_ini = VP2DForward(x, v, f0, feq, Hfun0, ion, TT)

Gam = [1,5,10];

label = ['no H'];
Enlist = [Enini];
Plist = [Perturb_ini];
for gam in Gam:
    FF,Time,Energy,Perturb = VP2DForward(x, v, f0, feq, Hfun, ion, TT,gam)
    Enlist.append(Energy)
    Plist.append(Perturb)
    label.append('H cancellation-based, '+ r'$\gamma$'+f' = {gam}')  

color = ['r','g','b','orange']    

for i in range(len(Enlist)):
    plt.semilogy(Time,Enlist[i],color = color[i])
plt.legend(label)
plt.title('Electric energy')
plt.xlabel('t')
plt.show()

for i in range(len(Plist)):
    plt.semilogy(Time,Plist[i],color=color[i])
plt.legend(label)
plt.title(r'$\frac{1}{2}||f-f_{eq}||^2$')
plt.xlabel('t')
plt.show()