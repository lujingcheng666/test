import numpy as np
from numpy import pi

import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import copy

from extend_BC import extend, extend2D

from math import factorial

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

Nx = 100;
Nv = 200;

TT = 30;

xmin = 0.; xmax = 10*pi;
x = np.linspace(xmin, xmax, Nx+1);
hx = x[1]-x[0];
Nx = len(x);

vmin = -8; vmax = 8;
v = np.linspace(vmin, vmax, Nv+1);
hv = v[1]-v[0];
Nv = len(v);

X, V = np.meshgrid(x, v, indexing='xy');

ion = 1.;

test_case = 'ts';

if test_case == 'ts': 
    Sig = 1;
    w1 = 0.5;
    w2 = 1-w1;
    u1 = 2.4;
    u2 = -2.4;
    fv = lambda v: w1/np.sqrt(2*pi)/Sig*np.exp(-(v-u1)**2/2/Sig**2)+\
        w2/np.sqrt(2*pi)/Sig*np.exp(-(v-u2)**2/2/Sig**2);

if test_case == 'bt':
    Sig1 = 1;
    Sig2 = 0.5;
    w1 = 0.9;
    w2 = 0.1;
    u1 = -2;
    u2 = 3.5;
    fv1 = lambda v: w1/np.sqrt(2*pi)/Sig1*np.exp(-(v-u1)**2/2/Sig1**2);
    fv2 = lambda v: w2/np.sqrt(2*pi)/Sig2*np.exp(-(v-u2)**2/2/Sig2**2);
    fv = lambda v: fv1(v)+fv2(v);

w = 2*pi/(max(x)-min(x));
r0 = 1;

feq = r0*fv(V);
if test_case == 'ts':
    eps = 1e-3;
    f0 = (1+eps*np.cos(w*x))*feq;
if test_case == 'bt':
    eps = 3e-3;
    f0 = feq+eps*np.sin(w*x)*fv2(V);



def interp2(X, Y, Z, Xp, Yp, mode='bilinear'):
    x = np.ascontiguousarray(X[0, :]);
    y = np.ascontiguousarray(Y[:, 0]);
    Z = np.ascontiguousarray(Z);
    if mode == 'bilinear':
        interpolator = sp.interpolate.RegularGridInterpolator((y, x), Z, bounds_error=False, fill_value=None);
        Zp = interpolator((Yp.reshape(-1), Xp.reshape(-1))).reshape(Xp.shape);
    elif mode == 'cubic':
        spline = sp.interpolate.RectBivariateSpline(y, x, Z, kx=3, ky=3);
        Zp = spline.ev(Yp.reshape(-1), Xp.reshape(-1)).reshape(Xp.shape);
    return Zp


'''
low rank neural operator
'''

class LNO(nn.Module):
    def __init__(self, x_sample, v_sample, rank):
        super().__init__()
        self.x_sample = x_sample;
        self.v_sample = v_sample;
        self.rank = rank;

        self.psi = nn.Sequential(
            nn.Linear(2, 64).to(device),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32).to(device),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2*rank+1).to(device)
        )

        self.f_encoder = nn.Sequential()
        # self.f_encoder = nn.Sequential(
        #     nn.Linear(1,64).to(device),
        #     nn.ReLU(),
        #     nn.Linear(64,1)
        #     )

        self.post = nn.Sequential()
        # self.post = nn.Sequential(
        #     nn.Linear(2*rank+1,64,bias=False).to(device),
        #     nn.ReLU(),
        #     nn.Linear(64,64,bias=False).to(device),
        #     nn.ReLU(),
        #     nn.Linear(64,2*rank+1,bias=False).to(device)
        #     )

    def forward(self, x_out, f):
        x_sample = self.x_sample;
        v_sample = self.v_sample;
        rank = self.rank;
        w = 2*pi/(x_sample.max()-x_sample.min());

        X_sample, V_sample = torch.meshgrid(x_sample, v_sample, indexing='xy');
        xv_sample = torch.cat((X_sample.reshape(-1, 1), V_sample.reshape(-1, 1)), dim=1);
        hx = x_sample[1]-x_sample[0];
        hv = v_sample[1]-v_sample[0];
        Nt, Nv, Nx = f.shape;

        Psi = self.psi(xv_sample).reshape(Nv, Nx, 2*rank+1);
        f_encoded = self.f_encoder(f.unsqueeze(-1)).squeeze(-1);
        weights = torch.sum(torch.sum(f.unsqueeze(-1)*Psi.unsqueeze(0)*hx*hv, 1), 1);
        weights = self.post(weights);
        phi = torch.cat([torch.sin(torch.arange(1, rank+1).to(device).reshape(-1, 1)*w*x_out),
                         torch.cos(torch.arange(0, rank+1).to(device).reshape(-1, 1)*w*x_out)]);
        h = weights@phi;
        return h

def get_nn_grad(net):
    param_grads = []
    for param in net.parameters():
        param_grads.append(param.grad.reshape(-1))
    grads = torch.cat(param_grads)
    return grads

'''
functions for generating random initial state
'''
def Hermite(x,N,normalize=0):
    Nx = len(x);
    He = np.zeros([N+1,Nx]);
    He[0,:] = np.ones(Nx);
    He[1,:] = x;
    for n in range(1,N):
        He[n+1,:] = x*He[n,:]-n*He[n-1,:];
    if normalize:
        for n in range(N+1):
            He[n,:] /= np.sqrt(np.sqrt(2*pi)*factorial(n))
    return He

def XVbasis(x,v,K,L):
    Lx = max(x)-min(x);
    w = 2*pi/Lx;
    xbasis = np.concatenate([np.sin(np.arange(1,K+1).reshape(-1, 1)*w*x),
                     np.cos(np.arange(0,K+1).reshape(-1, 1)*w*x)]);
    xbasis[:K,:] *= np.sqrt(2/Lx);
    xbasis[K,:] *= np.sqrt(1/Lx);
    xbasis[K+1:,:] *= np.sqrt(2/Lx);
    He = Hermite(v,L,normalize=1);
    vbasis = He*np.exp(-v**2/4);
    xvbasis = [];
    for i in range(xbasis.shape[0]):
        for j in range(vbasis.shape[0]):
            xvbasis.append(xbasis[i].reshape(1,-1)*vbasis[j].reshape(-1,1))
    
    xvbasis = np.array(xvbasis)
    return xvbasis
    
def sample_perturbation_weight(n_dim, n_samples=1, mode='radius_uniform', epsp=1.0):
    direction = np.random.randn(n_samples, n_dim)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    if mode == 'radius_uniform':
        radii = np.random.uniform(0, 1, size=(n_samples, 1))  # 模长均匀
    elif mode == 'volume_uniform':
        radii = np.random.uniform(0, 1, size=(n_samples, 1)) ** (1 / n_dim)  # 体积均匀
    elif mode == 'fixed_norm':
        radii = np.ones((n_samples, 1))
    else:
        raise ValueError("mode must be 'radius_uniform', 'volume_uniform', or 'fixed_norm'")

    return epsp * direction * radii

def generate_perturbation(xvbasis,epsp):
    mode='volume_uniform';
    weight = sample_perturbation_weight(n_dim=xvbasis.shape[0],mode=mode,epsp=epsp).reshape(-1,1,1);
    fp = epsp*np.sum(weight*xvbasis,0);
    return fp
    

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


def VP2DForward(x, v, f0, feq, hnet, ion, TT):
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

        x_pt = torch.tensor(x).float().to(device);
        df_pt = torch.tensor(f-feq).float().unsqueeze(0).to(device);
        H = hnet(x_pt, df_pt).detach().cpu().numpy();
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

    return FF, Time, Energy


def VP2DLambSL(x, v, k, lamb, f, feq, w_running, H, ion):
    Nx = len(x);
    Nv = len(v);
    hx = x[1]-x[0];
    hv = v[1]-v[0];
    [X, V] = np.meshgrid(x, v, indexing='xy');

    m = 20;
    BCl = 'P'; BCr = 'P'; BCd = 'D'; BCu = 'D';
    # stage 1 convection
    Xe, Ve, lambe = extend2D(X, V, lamb, hx, hv, m, BCl,
                             lamb[:, 0], BCr, lamb[:, -1], BCd, lamb[0, :], BCu, lamb[-1, :]);
    Xp = X+k/2*V;
    Vp = V;
    lambm1 = interp2(Xe, Ve, lambe, Xp, Vp);
    # stage 2 convection
    r = np.sum(f*hv, 0);
    E = electric(x, hx, r, ion);
    Ftot = E+H;
    Ftot = np.ones([Nv, 1])*Ftot;

    Xe, Ve, lambm1e = extend2D(X, V, lambm1, hx, hv, m, BCl,
                               lambm1[:, 0], BCr, lambm1[:, -1], BCd, lambm1[0, :], BCu, lambm1[-1, :]);
    Xp = X;
    Vp = V+k/2*Ftot;
    lambm2 = interp2(Xe, Ve, lambm1e, Xp, Vp);
    # stage 3 source
    lambm3 = lambm2-k*(f-feq)*w_running;
    # stage 4 convection
    Xe, Ve, lambm3e = extend2D(X, V, lambm3, hx, hv, m, BCl,
                               lambm3[:, 0], BCr, lambm3[:, -1], BCd, lambm3[0, :], BCu, lambm3[-1, :]);
    Xp = X;
    Vp = V+k/2*Ftot;
    lambm4 = interp2(Xe, Ve, lambm3e, Xp, Vp);
    # stage 5 convection
    Xe, Ve, lambm4e = extend2D(X, V, lambm4, hx, hv, m, BCl,
                               lambm4[:, 0], BCr, lambm4[:, -1], BCd, lambm4[0, :], BCu, lambm4[-1, :]);
    Xp = X+k/2*V;
    Vp = V;
    lambm5 = interp2(Xe, Ve, lambm4e, Xp, Vp);

    return lambm5


def VP2DLambBackward(x, v, lambT, FF, feq, w_running, Time, hnet, ion, TT):
    Nx = len(x);
    Nv = len(v);
    Nt = len(Time);

    TT = max(Time);
    time = TT;
    lamb = lambT;

    Lamb = np.zeros([Nt, Nv, Nx]);
    Lamb[Nt-1,:] = lambT;

    for nt in range(Nt-1, 0, -1):
        k = Time[nt]-Time[nt-1];
        x_pt = torch.tensor(x).float().to(device);
        f = (FF[nt,:,:]+FF[nt-1,:,:])/2;
        df_pt = torch.tensor(f-feq).float().unsqueeze(0).to(device);
        H = hnet(x_pt, df_pt).detach().cpu().numpy();

        lamb = VP2DLambSL(x, v, k, lamb, f, feq, w_running, H, ion);
        time = time-k;

        Lamb[nt-1,:] = lamb;

    return Lamb


def VPloss(x, v, FF, feq, Lamb, hnet, Time):
    hx = x[1]-x[0];
    hv = v[1]-v[0];

    dvF = np.zeros([len(Time), Nv, Nx]);
    dvF[:,1:-1,:] = (FF[:,2:,:]-FF[:,:-2,:])/2/hv;
    dvF[:,-1,:] = (FF[:,-1,:]-FF[:,-2,:])/hv;
    dvF[:,0,:] = (FF[:,1,:]-FF[:,0,:])/hv;

    dvF_pt = torch.tensor(dvF).to(device);
    Lamb_pt = torch.tensor(Lamb).to(device);
    x_pt = torch.tensor(x).float().to(device);
    dF_pt = torch.tensor(FF-feq).float().to(device);
    H = hnet(x_pt, dF_pt);
    dloss = torch.sum(H*torch.sum(Lamb_pt*dvF_pt, 1), 1)*hx*hv;

    dt = torch.tensor(Time[1:]-Time[:-1]).to(device);
    loss = torch.sum((dloss[:-1]+dloss[1:])/2*dt);

    return loss


x_pt = torch.tensor(x).float().to(device);
v_pt = torch.tensor(v).float().to(device);

hnet = LNO(x_sample=x_pt, v_sample=v_pt, rank=15)
hnet.psi[-1].weight.data.fill_(0)
hnet.psi[-1].bias.data.fill_(0)

hnet0 = LNO(x_sample=x_pt, v_sample=v_pt, rank=1)
for param in hnet0.parameters():
    param.data.fill_(0)

iter = 0;
maxiter = 3000;
preiter = 200;
w_running = 1;
w_terminal = 0;
TTfuture = 70;
xvbasis = XVbasis(x,v,K=5,L=5); # 2K+1 basis func in x, L+1 basis func in v
epsp = 1e-3; # magitude of random perturbation

Fini_total, Ttotal, Enini_total = VP2DForward(x, v, f0, feq, hnet0, ion, TTfuture);

Lbest = 1/2*np.sum((Fini_total[-1,:]-feq)**2*hx*hv);
param_best = copy.deepcopy(hnet.state_dict());
# param_best_list = [];

# ts: lr=5e-3, bt: lr=2e-3
optimizer_pre = torch.optim.Adagrad(hnet.parameters(), lr=5e-3) 
# ts: lr=5e-4, bt: lr=3e-4
optimizer_main = torch.optim.Adam(hnet.parameters(), lr=5e-4) 

while iter < maxiter:
    fp = generate_perturbation(xvbasis, epsp);
    f0p = f0+fp;
    FF, Time, Energy = VP2DForward(x, v, f0p, feq, hnet, ion, TT);
    
    dt = Time[1]-Time[0];
    L = w_running*1/2*np.sum((FF-feq)**2*hx*hv*dt) + \
        w_terminal*1/2*np.sum((FF[-1, :]-feq)**2*hx*hv);
        
    LgradT = FF[-1, :]-feq;
    lambT = -LgradT*w_terminal;
    Lamb = VP2DLambBackward(x, v, lambT, FF, feq, w_running, Time, hnet, ion, TT);
    
    if iter < preiter:
        optimizer = optimizer_pre
    else:
        optimizer = optimizer_main
        
    optimizer.zero_grad()
    loss = VPloss(x, v, FF, feq, Lamb, hnet, Time);
    loss.backward()
    optimizer.step()
    
    grads = get_nn_grad(hnet);
    res = max(abs(grads)).item();
    
    Ftotal, Ttotal, Entotal = VP2DForward(x, v, f0, feq, hnet, ion, TTfuture);
    perturb = 1/2*np.sum(np.sum((Ftotal-feq)**2*hx*hv, 1), 1)
    perturb_ini = 1/2*np.sum(np.sum((Fini_total-feq)**2*hx*hv, 1), 1)
    
    Lfuture = 1/2*np.sum((Ftotal[-1,:]-feq)**2*hx*hv);
    
    if Lfuture<Lbest:
        param_best = copy.deepcopy(hnet.state_dict());
        Lbest = Lfuture;
        best_iter = iter;
    
    # if (iter+1)%1000 == 0:
    #     param_best_list.append(param_best)
        
    print('iter = ', iter, ', res = ', res, ', L = ', L,', Lfuture = ',Lfuture,\
          ', Lbest = ',Lbest)

    plt.semilogy(Ttotal, perturb, Ttotal, perturb_ini, 'r')
    plt.title(f'iter = %s' % (iter))
    plt.axvline(TT, color='k')
    plt.show()

    iter += 1;

hnet_best = copy.deepcopy(hnet);
hnet_best.load_state_dict(param_best)

TTtest = 70;
Ftest_ini, Ttest, Entest_ini = VP2DForward(x, v, f0, feq, hnet0, ion, TTtest);
Ftest, Ttest, Entest = VP2DForward(x, v, f0, feq, hnet_best, ion, TTtest);

dFtest_pt = torch.tensor(Ftest-feq).float().to(device);
Htest = hnet_best(x_pt, dFtest_pt).cpu().detach();

perturb_ini = 1/2*np.sum(np.sum((Ftest_ini-feq)**2*hx*hv, 1), 1);
perturb = 1/2*np.sum(np.sum((Ftest-feq)**2*hx*hv, 1), 1);

plt.semilogy(Ttest, Entest, Ttest, Entest_ini, 'r')
plt.title('electric energy')
plt.axvline(TT, color='k')
plt.show()

plt.semilogy(Ttest, perturb, Ttest, perturb_ini, 'r')
plt.title(r'$\frac{1}{2}||f-f_{eq}||^2$')
plt.axvline(TT, color='k')
plt.show()

plot = plt.pcolor(X, V, Ftest_ini[-1, :], cmap='jet')
plt.colorbar(plot)
plt.show()

plot = plt.pcolor(X, V, Ftest[-1, :], cmap='jet')
plt.colorbar(plot)
plt.show()

plt.plot(x, Htest[0, :], x, Htest[-1, :])
plt.show()
