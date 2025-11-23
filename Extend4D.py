import numpy as np
from numpy import pi

import scipy as sp

import torch 

def interp4(X, Y, V1, V2, Z, Xp, Yp, V1p, V2p,method='linear'):
    x, y, v1, v2 = np.ascontiguousarray(X[0,:,0,0]), np.ascontiguousarray(Y[:,0,0,0]),\
        np.ascontiguousarray(V1[0,0,0,:]),np.ascontiguousarray(V2[0,0,:,0])
    interpolator = sp.interpolate.RegularGridInterpolator((y, x, v2, v1), Z, bounds_error=False, fill_value=None,method=method);
    Zp = interpolator((Yp.reshape(-1), Xp.reshape(-1), V2p.reshape(-1), V1p.reshape(-1))).reshape(Xp.shape);
    return Zp

def extend4D(X,Y,V1,V2,hx,hy,hv1,hv2,f,m):
    Ny,Nx,Nv2,Nv1 = X.shape;
    
    x = X[0,:,0,0];
    y = Y[:,0,0,0];
    v1 = V1[0,0,0,:];
    v2 = V2[0,0,:,0];
    
    idx = np.arange(m,m+Nx).reshape(1,-1,1,1);
    idy = np.arange(m,m+Ny).reshape(-1,1,1,1);
    idv1 = np.arange(m,m+Nv1).reshape(1,1,1,-1);
    idv2 = np.arange(m,m+Nv2).reshape(1,1,-1,1);
    q = np.arange(1,m+1);
    
    xe = np.zeros(Nx+2*m);
    ye = np.zeros(Ny+2*m);
    v1e = np.zeros(Nv1+2*m);
    v2e = np.zeros(Nv2+2*m);
    
    xe[m:m+Nx] = x;
    ye[m:m+Ny] = y;
    v1e[m:m+Nv1] = v1;
    v2e[m:m+Nv2] = v2;
    
    xe[m-q] = x[0]-q*hx; xe[m+Nx-1+q] = x[-1]+q*hx;
    ye[m-q] = y[0]-q*hy; ye[m+Ny-1+q] = y[-1]+q*hy;
    v1e[m-q] = v1[0]-q*hv1; v1e[m+Nv1-1+q] = v1[-1]+q*hv1;
    v2e[m-q] = v2[0]-q*hv2; v2e[m+Nv2-1+q] = v2[-1]+q*hv2;
    
    Xe,Ye,V2e,V1e = np.meshgrid(xe,ye,v2e,v1e,indexing='xy');
    
    fe = np.zeros([Ny+2*m,Nx+2*m,Nv2+2*m,Nv1+2*m]);
    fe[idy,idx,idv2,idv1] = f;
    
    # extend y direction periodically
    fe[m-q.reshape(-1,1,1,1),idx,idv2,idv1] = f[Ny-1-q,:,:,:];
    fe[m+Ny-1+q.reshape(-1,1,1,1),idx,idv2,idv1] = f[q,:,:,:];
    # extend x direction periodically
    fe[:,m-q.reshape(1,-1,1,1),idv2,idv1] = fe[:,m+Nx-1-q.reshape(1,-1,1,1),idv2,idv1];
    fe[:,m+Nx-1+q.reshape(1,-1,1,1),idv2,idv1] = fe[:,m+q.reshape(1,-1,1,1),idv2,idv1];
    # extend v2 direction
    fe[:,:,m-q.reshape(1,1,-1,1),idv1] = 2*fe[:,:,m,idv1]-fe[:,:,m+q.reshape(1,1,-1,1),idv1];
    fe[:,:,m+Nv2-1+q.reshape(1,1,-1,1),idv1] = 2*fe[:,:,m+Nv2-1,idv1]-fe[:,:,m+Nv2-1-q.reshape(1,1,-1,1),idv1];
    # extend v1 direction
    fe[:,:,:,m-q] = 2*fe[:,:,:,m,None]-fe[:,:,:,m+q];
    fe[:,:,:,m+Nv1-1+q] = 2*fe[:,:,:,m+Nv1-1,None]-fe[:,:,:,m+Nv1-1-q];
    
    return Xe,Ye,V1e,V2e,fe

if __name__ == '__main__':
    x = np.linspace(0,2*pi,64);
    y = np.linspace(0,2*pi,64);
    v1 = np.linspace(-8,8,128);
    v2 = np.linspace(-5,7,128);

    hx = x[1]-x[0];
    hy = y[1]-y[0];
    hv1 = v1[1]-v1[0];
    hv2 = v2[1]-v2[0];

    X,Y,V2,V1 = np.meshgrid(x,y,v2,v1,indexing='xy')

    f = (1+0.5*np.sin(X)*np.cos(Y))*(np.exp(-V1**2/2)+np.exp(-(V2-1)**2/2));

    m = 2;
    Xe,Ye,V1e,V2e,fe = extend4D(X, Y, V1, V2, hx, hy, hv1, hv2, f, m);
    
    feref = (1+0.5*np.sin(Xe)*np.cos(Ye))*(np.exp(-V1e**2/2)+np.exp(-(V2e-1)**2/2));
    
    extend_err = np.max(abs(fe-feref));
    
    print(f'extension err: {extend_err}')
    
    finterp = interp4(Xe, Ye, V1e, V2e, fe, X, Y, V1, V2);
    interp_err = np.max(abs(finterp-f));
    
    print(f'interpolation err: {interp_err}')
