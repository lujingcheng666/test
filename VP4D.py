import numpy as np
from numpy import pi

import scipy as sp

import matplotlib.pyplot as plt

from extend_BC import extend2D

from Extend4D import extend4D, interp4

import gc

Nx = 70;
Ny = 70;
Nv1 = 120;
Nv2 = 120;

TT = 30;

xmin = 0.; xmax = 10*pi;
ymin = 0.; ymax = 10*pi;
v1min = -8; v1max = 8;
v2min = -8; v2max = 8;
x = np.linspace(xmin, xmax, Nx+1);
y = np.linspace(ymin, ymax, Ny+1);
v1 = np.linspace(v1min, v1max, Nv1+1);
v2 = np.linspace(v2min, v2max, Nv2+1);

hx = x[1]-x[0];
hy = y[1]-y[0];
hv1 = v1[1]-v1[0];
hv2 = v2[1]-v2[0];

Nx = len(x);
Ny = len(y);
Nv1 = len(v1);
Nv2 = len(v2);


X,Y,V2,V1 = np.meshgrid(x,y,v2,v1,indexing='xy');
vv1,vv2 = np.meshgrid(v1,v2,indexing='xy');
xx,yy = np.meshgrid(x,y,indexing='xy');

ion = 1.;

ux1 = 2.;
uy1 = 2.;
ux2 = -2.;
uy2 = -2.;
w1 = 0.5;
w2 = 0.5;
fv = lambda x,y,v1,v2: w1/2/pi*np.exp(-(v1-ux1)**2/2-(v2-uy1)**2/2)+\
    w2/2/pi*np.exp(-(v1-ux2)**2/2-(v2-uy2)**2/2);

wx = 2*pi/(xmax-xmin);
wy = 2*pi/(ymax-ymin);    
r0 = 1;

feq = fv(X,Y,V1,V2);

eps = 1e-2;
f0 = (1+eps*np.sin(wx*X)*np.cos(wy*Y))*feq;

def plot_surface(X,Y,Z):
    fig = plt.figure();
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(X,Y,Z,cmap='jet')
    plt.show()
    
def Poisson2Dmat(x,y):
    Nx = len(x);
    Ny = len(y);
    hx = x[1]-x[0]; 
    hy = y[1]-y[0];
    A = sp.sparse.lil_matrix((Nx*Ny,Nx*Ny));
    
    for i in range(Ny):
        for j in range(Nx):
            ip = (i+1)%(Ny-1);
            im = (i-1)%(Ny-1);
            jp = (j+1)%(Nx-1);
            jm = (j-1)%(Nx-1);
            
            A[i*Nx+j,im*Nx+j] = -1/hy**2;
            A[i*Nx+j,i*Nx+jm] = -1/hx**2;
            A[i*Nx+j,i*Nx+j] = 2/hx**2+2/hy**2;
            A[i*Nx+j,i*Nx+jp] = -1/hx**2;
            A[i*Nx+j,ip*Nx+j] = -1/hy**2;
    
    # u_boundary = 0
    A[0,:] = 0;
    A[0,0] = 1;
    A = A.tocsr();
    return A
    
def electric2D(x,y,hx,hy,r,ion,A):
    Nx = len(x);
    Ny = len(y);
    X,Y = np.meshgrid(x,y,indexing='xy');
    
    rhs = r-ion;
    rhs[0] = 0; # phi_boundary = 0
    phi = sp.sparse.linalg.spsolve(A, rhs.reshape(-1)).reshape(Ny,Nx);
    
    # plot_surface(X, Y, phi)
    
    m = 2;
    BCl = 'P'; BCr = 'P'; BCd = 'P'; BCu = 'P';
    Xe,Ye,phie = extend2D(X, Y, phi, hx, hy, m, BCl, phi[:,0], BCr , phi[:,-1],
                          BCd, phi[0,:], BCu, phi[-1,:])
    
    Ex = -(phie[m:m+Ny,m+1:m+Nx+1]-phie[m:m+Ny,m-1:m+Nx-1])/2/hx;
    Ey = -(phie[m+1:m+Ny+1,m:m+Nx]-phie[m-1:m+Ny-1,m:m+Nx])/2/hy;
    
    
    return Ex,Ey

def VP4DSL(X,Y,V1,V2,k,f,Hx,Hy,ion,A):
    Ny,Nx,Nv2,Nv1 = f.shape;
    
    x = X[0,:,0,0];
    y = Y[:,0,0,0];
    v1 = V1[0,0,0,:];
    v2 = V2[0,0,:,0];
    
    hx = x[1]-x[0];
    hy = y[1]-y[0];
    hv1 = v1[1]-v1[0];
    hv2 = v2[1]-v2[0];
    
    m = 10;
    # stage 1 convection
    Xe,Ye,V1e,V2e,fe = extend4D(X, Y, V1, V2, hx, hy, hv1, hv2, f, m);
    Xp = X-k/2*V1;
    Yp = Y-k/2*V2;
    V1p = V1;
    V2p = V2;
    f1 = interp4(Xe, Ye, V1e, V2e, fe, Xp, Yp, V1p, V2p);
    
    del fe
    # stage 2 convection
    _, _, _, _, f1e = extend4D(X, Y, V1, V2, hx, hy, hv1, hv2, f1, m);
    r = np.sum(np.sum(f1*hv1*hv2,2),2);
    Ex, Ey = electric2D(x, y, hx, hy, r, ion, A);
    Fx = Ex+Hx;
    Fy = Ey+Hy;
    
    Xp = X;
    Yp = Y;
    V1p = V1-k*Fx[:,:,None,None];
    V2p = V2-k*Fy[:,:,None,None];
    f2 = interp4(Xe, Ye, V1e, V2e, f1e, Xp, Yp, V1p, V2p);
    
    del f1,f1e
    # stage 3 convection
    _, _, _, _, f2e = extend4D(X, Y, V1, V2, hx, hy, hv1, hv2, f2, m)
    Xp = X-k/2*V1;
    Yp = Y-k/2*V2;
    V1p = V1;
    V2p = V2;
    f3 = interp4(Xe, Ye, V1e, V2e, f2e, Xp, Yp, V1p, V2p);
    
    del f2,f2e
    gc.collect()
    
    return f3

def grad_v_feq(feq, hv1, hv2):
    Ny,Nx,Nv2,Nv1 = feq.shape;
    
    dv1feq = np.zeros_like(feq);
    dv2feq = np.zeros_like(feq);
    
    dv1feq[:,:,:,1:-1] = (feq[:,:,:,2:] - feq[:,:,:,:-2]) / (2*hv1);
    dv1feq[:,:,:,0] = (feq[:,:,:,1] - feq[:,:,:,0]) / hv1;
    dv1feq[:,:,:,-1] = (feq[:,:,:,-1] - feq[:,:,:,-2]) / hv1;
    
    dv2feq[:,:,1:-1,:] = (feq[:,:,2:,:] - feq[:,:,:-2,:]) / (2*hv2);
    dv2feq[:,:,0,:] = (feq[:,:,1,:] - feq[:,:,0,:]) / hv2;
    dv2feq[:,:,-1,:] = (feq[:,:,-1,:] - feq[:,:,-2,:]) / hv2;
    
    return dv1feq, dv2feq

def Helmholtz_proj(x,y,v1,v2,Hx,Hy,A):
    Nx = len(x);
    Ny = len(y);
    X,Y = np.meshgrid(x,y,indexing='xy');
    hx = x[1]-x[0];
    hy = y[1]-y[0]; 
    
    m = 2;
    BCl = 'P'; BCr = 'P'; BCd = 'P'; BCu = 'P';
    
    idx = np.arange(m,m+Nx).reshape(1,-1);
    idy = np.arange(m,m+Ny).reshape(-1,1);
    
    Xe, Ye, Hxe = extend2D(X, Y, Hx, hx, hy, m, BCl, Hx[:,0], BCr, Hx[:,-1], BCd, Hx[0,:], BCu, Hx[-1,:]);
    _, _, Hye = extend2D(X, Y, Hy, hx, hy, m, BCl, Hy[:,0], BCr, Hy[:,-1], BCd, Hy[0,:], BCu, Hy[-1,:]);
    
    divH = (Hxe[idy,idx+1]-Hxe[idy,idx-1])/2/hx+(Hye[idy+1,idx]-Hye[idy-1,idx])/2/hy;
    rhs = divH.reshape(-1);
    rhs[0] = 0;
    # -\Delta phih = divH
    phih = sp.sparse.linalg.spsolve(A, rhs.reshape(-1)).reshape(Ny,Nx);
    _,_,phie = extend2D(X, Y, phih, hx, hy, m, BCl, phih[:,0], BCr , phih[:,-1],
                          BCd, phih[0,:], BCu, phih[-1,:]);
    
    Hx_proj = -(phie[m:m+Ny,m+1:m+Nx+1]-phie[m:m+Ny,m-1:m+Nx-1])/2/hx;
    Hy_proj = -(phie[m+1:m+Ny+1,m:m+Nx]-phie[m-1:m+Ny-1,m:m+Nx])/2/hy;
    return Hx_proj, Hy_proj
    
    
def Hfun2D(x,y,v1,v2,df,dv1feq,dv2feq,ion,A,gam=1.):
    hx = x[1]-x[0];
    hy = y[1]-y[0];
    hv1 = v1[1]-v1[0];
    hv2 = v2[1]-v2[0];    
    
    dHx = gam*np.sum(np.sum(df*dv1feq*hv1*hv2,2),2);
    dHy = gam*np.sum(np.sum(df*dv2feq*hv1*hv2,2),2);
    
    dHx, dHy = Helmholtz_proj(x, y, v1, v2, dHx, dHy, A);
        
    drho = np.sum(np.sum(df*hv1*hv2,2),2);
    dEx, dEy = electric2D(x, y, hx, hy, drho, 0, A);
    
    Hx = -dEx+dHx;
    Hy = -dEy+dHy;
    
    return Hx, Hy
    
A = Poisson2Dmat(x, y); 

r = np.sum(np.sum(f0*hv1*hv2,2),2);
Ex, Ey = electric2D(x, y, hx, hy, r, ion, A);
plot_surface(xx, yy, Ex)
plot_surface(xx, yy, Ey)

dxEx = (Ex[1:-1,2:]-Ex[1:-1,:-2])/2/hx;
dyEy = (Ey[2:,1:-1]-Ey[:-2,1:-1])/2/hy;
err = np.max(abs(dxEx+dyEy-(r[1:-1,1:-1]-ion)));
plot_surface(xx[1:-1,1:-1],yy[1:-1,1:-1],dxEx+dyEy)
plot_surface(xx,yy,r-ion)


time = 0;
f = f0;
dv1feq, dv2feq = grad_v_feq(feq, hv1, hv2);

Energy = [];
Time = [];
Perturb = [];

r = np.sum(np.sum(f*hv1*hv2,2),2);
Ex, Ey = electric2D(x, y, hx, hy, r, ion, A);
energy = 1/2*np.sum(Ex**2+Ey**2)*hx*hy;
perturb = 1/2*np.sum((f0-feq)**2)*hx*hy*hv1*hv2;

Time.append(time)
Energy.append(energy)
Perturb.append(perturb)

Htype = 1;
gam = 2;

cfl = 5;
while time<TT:
    k = cfl/(max(abs(v1))/hx+max(abs(v2))/hy);
    if time+k>TT:
        k = TT-time;
    
    if Htype == 0:
        Hx = 0;
        Hy = 0;
    elif Htype == 1:
        df = f-feq;
        Hx, Hy = Hfun2D(x,y,v1,v2,df,dv1feq,dv2feq,ion,A,gam);
    
    f = VP4DSL(X, Y, V1, V2, k, f, Hx, Hy, ion, A);
    
    time = time+k;
    
    r = np.sum(np.sum(f*hv1*hv2,2),2);
    Ex, Ey = electric2D(x, y, hx, hy, r, ion, A);
    energy = 1/2*np.sum(Ex**2+Ey**2)*hx*hy;
    perturb = 1/2*np.sum((f-feq)**2)*hx*hy*hv1*hv2;
    
    Time.append(time)
    Energy.append(energy)
    Perturb.append(perturb)
    
    plt.pcolor(vv1,vv2,f[int(np.floor(Ny/2)),int(np.floor(Nx/2)),:,:],cmap='jet')
    plt.title(f't = {time}')
    plt.show()
    
    print(f'time = {time}, perturb {perturb}, energy {energy}')
    

# file_name = 'VP4D_eps0.01_no_H.npz'
# file_name = 'VP4D_eps0.01_cancellation.npz'

# np.savez(file_name, f=f, eps=eps, Perturb=Perturb, Energy=Energy, Time=Time)
