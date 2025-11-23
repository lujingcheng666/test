import numpy as np

'''
Ghost cell extension
'''
def extend(x,u,h,m,BCl,ul,BCr,ur):
    u = np.array(u).reshape(-1);
    N = len(x);
    
    q = np.arange(1,m+1,1);

    xe = np.zeros(N+2*m);
    ue = np.zeros(N+2*m);

    xe[m:m+N] = x;
    xe[m-q] = x[0]-q*h;
    xe[m+N-1+q] = x[N-1]+q*h;

    ue[m:m+N] = u; 
    if BCl == 'P':
       ue[m-q] = u[N-1-q];
    elif BCl == 'D':
        ue[m-q] = 2*ul-u[q];
    elif BCl == 'N':
        ue[m-q] = u[q];
        
    if BCr == 'P':
        ue[m+N-1+q] = u[q];
    elif BCr == 'D':
        ue[m+N-1+q] = 2*ur-u[N-1-q];
    elif BCr == 'N':
        ue[m+N-1+q] = u[N-1-q];
        
    return xe,ue

def extends(xs,us,h,m,BCl,ul,BCr,ur):
    us = np.array(us).reshape(-1);
    Ns = len(xs);
    
    q = np.arange(1,m+1,1);
    
    xes = np.zeros(Ns+2*m);
    ues = np.zeros(Ns+2*m);
    
    xes[m:m+Ns] = xs;
    xes[m-q] = xs[0]-q*h;
    xes[m+Ns-1+q] = xs[Ns-1]+q*h;
    
    ues[m:m+Ns] = us;
    if BCl == 'P':
        ues[m-q] = us[Ns-q];
    elif BCl == 'D':
        ues[m-q] = 2*ul-us[q-1];
    elif BCl == 'N':
        ues[m-q] = us[q-1];
    
    if BCr == 'P':
        ues[m+Ns-1+q] = us[q-1];
    elif BCr == 'D':
        ues[m+Ns-1+q] = 2*ur-us[Ns-q];
    elif BCr == 'N':
        ues[m+Ns-1+q] = us[Ns-q];
    
    return xes, ues
        

def extend2D(X,Y,u,hx,hy,m,BCl,ul,BCr,ur,BCd,ud,BCu,uu):
    Ny,Nx = u.shape;
    
    x = X[0,:];
    y = Y[:,0];
    
    q = np.arange(1,m+1);

    Xe = np.zeros([Ny+2*m,Nx+2*m]);
    Ye = np.zeros([Ny+2*m,Nx+2*m]);
    ue = np.zeros([Ny+2*m,Nx+2*m]);
    
    idx = np.arange(m,m+Nx);
    idy = np.arange(m,m+Ny);
    
    Xe[:,idx] = x;
    Xe[:,m-q] = x[0]-q*hx;
    Xe[:,m+Nx-1+q] = x[Nx-1]+q*hx;
    
    Ye[idy,:] = y.reshape(-1,1);
    Ye[m-q,:] = y[0]-q.reshape(-1,1)*hy;
    Ye[m+Ny-1+q,:] = y[Ny-1]+q.reshape(-1,1)*hy;

    ue[idy.reshape(-1,1),idx] = u; 
    if BCl == 'P':
       ue[idy.reshape(-1,1),m-q] = u[:,Nx-1-q];
    elif BCl == 'D':
        ue[idy.reshape(-1,1),m-q] = 2*ul-u[:,q];
    elif BCl == 'N':
        ue[idy.reshape(-1,1),m-q] = u[:,q];
        
    if BCr == 'P':
        ue[idy.reshape(-1,1),m+Nx-1+q] = u[:,q];
    elif BCr == 'D':
        ue[idy.reshape(-1,1),m+Nx-1+q] = 2*ur-u[:,Nx-1-q];
    elif BCr == 'N':
        ue[idy.reshape(-1,1),m+Nx-1+q] = u[:,Nx-1-q];
    
    if BCd == 'P':
        ue[m-q,:] = ue[m+Ny-1-q,:];
    elif BCd == 'D':
        ue[m-q,:] = 2*ue[m,:]-ue[m+q,:];
    elif BCd == 'N':
        ue[m-q,:] = ue[m+q,:];
        
    if BCu == 'P':
        ue[m+Ny-1+q,:] = ue[m+q,:];
    elif BCu == 'D':
        ue[m+Ny-1+q,:] = 2*ue[m+Ny-1,:]-ue[m+Ny-1-q,:];
    elif BCu == 'N':
        ue[m+Ny-1+q,:] = ue[m+Ny-1-q,:];
        
    return Xe,Ye,ue

if __name__ == '__main__':
    from numpy import pi
    import matplotlib.pyplot as plt
    
    x = np.linspace(0,2*pi,100);
    h = x[1]-x[0];
    m = 5;
    u = np.sin(x);
    xe,ue = extend(x,u,h,m,'P',u[0],'P',u[-1])
    
    plt.plot(xe,ue)
    plt.show()
    
    y = np.linspace(0,2*pi,200);
    X,Y = np.meshgrid(x,y);
    u = np.sin(X+Y);
    hx = x[1]-x[0];
    hy = y[1]-y[0];
    Xe,Ye,ue = extend2D(X,Y,u,hx,hy,m,'P',u[:,0],'P',u[:,-1],\
                        'P',u[0,:],'P',u[-1,:]);
    
    fig1 = plt.figure()
    ax = fig1.add_subplot(111,projection = '3d')
    ax.plot_surface(Xe,Ye,ue)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    plt.show()
    
    fig2 = plt.pcolor(Xe,Ye,ue,cmap = 'jet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(fig2)
    plt.show()