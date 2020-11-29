# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy import constants as const


class synth(object):
    def __init__(self, rho, T, vz, dz):
        """
        Synthetic observation of 21cm line emission
        author: A. Marchal
        
        Parameters
        ----------
        
        Returns:
        --------
        """    
        super(synth, self).__init__()
        if len(rho.shape) == 1:
            self.dim = 1
            self.rho = np.zeros((len(rho),1,1)) *u.g*u.cm**-3
            self.T = np.zeros((len(T),1,1)) *u.K
            self.vz = np.zeros((len(vz),1,1)) *u.km*u.s**-1

            self.rho[:,0,0] = rho *u.g*u.cm**-3
            self.T[:,0,0] = T *u.K
            self.vz[:,0,0] = vz * 1.e-5 *u.km*u.s**-1
            
        else:
            self.dim = 3
            self.rho = rho *u.g*u.cm**-3
            self.T = T *u.K
            self.vz = vz * 1.e-5 *u.km*u.s**-1

        # Constant
        self.m_h = 1.6737236e-27 *u.kg
        self.C = 1.82243e18 *u.K**-1 *u.cm**-2 / (u.km * u.s**-1)
        self.dz = dz *u.pc
        self.dz_cm = self.dz.to(u.cm)


    def gen(self, vmin=-40, vmax=40, dv=0.8, T_lim=[0,np.inf], thin=False):        
        # Cut temperature field 
        Tk_lim_inf = T_lim[0]
        Tk_lim_sup = T_lim[1]

        idx_phase = np.where((self.T.value > Tk_lim_inf) & (self.T.value < Tk_lim_sup))
        
        rho_cube_phase = np.zeros((self.rho.value.shape[0], self.rho.value.shape[1], self.rho.value.shape[2]))
        T_cube_phase = np.zeros((self.rho.value.shape[0], self.rho.value.shape[1], self.rho.value.shape[2]))
        vz_cube_phase = np.zeros((self.rho.value.shape[0], self.rho.value.shape[1], self.rho.value.shape[2]))
        
        rho_cube_phase[idx_phase] = self.rho.value[idx_phase]
        T_cube_phase[idx_phase] = self.T.value[idx_phase]
        vz_cube_phase[idx_phase] = self.vz.value[idx_phase]

        # Preliminary calculation
        Delta2 = ((const.k_B.value * T_cube_phase / self.m_h.value)) * 1.e-6 #km.s-1
        n = rho_cube_phase/(self.m_h.value*1.e3)
        Delta = np.sqrt(Delta2)
        n_Delta = n / Delta

        # Spectral range
        u = np.arange(vmin,vmax+dv, dv)

        map_u = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
        for i in np.arange(T_cube_phase.shape[1]):
            for j in np.arange(T_cube_phase.shape[2]):
                map_u[:,i,j] = u

        if thin == True:
            Tb_thin_fast = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))            
            tau_thin_fast = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))

            for i in tqdm(range(len(u))):
                phi = 1. / np.sqrt(2.*np.pi) / Delta * np.exp(- (u[i] - (vz_cube_phase))**2 / (2.*Delta2))
                n_phi = n * phi
                tau_v = n_phi / T_cube_phase

                n_phi[np.where(n_phi != n_phi)] = 0.
                tau_v[tau_v != tau_v] = 0.

                Tb_thin_fast[i] = 1. / self.C.value * np.sum(n_phi,0) * self.dz_cm.value
                tau_thin_fast[i] = 1. / self.C.value * np.sum(tau_v,0) * self.dz_cm.value
            
            if self.dim == 1:
                return Tb_thin_fast[:,0,0], tau_thin_fast[:,0,0]
            else:
                return Tb_thin_fast, tau_thin_fast

        else:
            Tb = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
            tau_in_front = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
            
            for i in tqdm(range(T_cube_phase.shape[0])):    
                Tb_z = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
                tau_z = 1. / (self.C.value * np.sqrt(2.*np.pi)) * n_Delta[i] / T_cube_phase[i] * np.exp(- (map_u - (vz_cube_phase[i]))**2 
                                                                                               / (2.*Delta2[i])) * self.dz_cm.value
                idx_nonzero = ~np.isnan(tau_z[0])
                Tb_z[:,idx_nonzero] = T_cube_phase[i,idx_nonzero] * (1. - np.exp(-1.*tau_z[:,idx_nonzero])) * np.exp(-1.*tau_in_front[:,idx_nonzero])
                
                tau_in_front[:,idx_nonzero] += tau_z[:,idx_nonzero]
                
                Tb += Tb_z 
            if self.dim == 1:
                return Tb[:,0,0], tau_in_front[:,0,0]
            else:
                return Tb, tau_in_front


if __name__ == '__main__':    
    from SPARK.absorption import lbfgs_abs   

    # Open data 
    path = '/home/amarchal/SPARK/data/'
    
    hdu_list_rho = fits.open(path + 'rho_cube_sample.fits')
    hdu_list_T = fits.open(path + 'T_cube_sample.fits')
    hdu_list_vz = fits.open(path + 'vz_cube_sample.fits')
    
    #Velocity range and channel spacing
    vmin = -40 #km.s-1
    vmax = 40 #km.s-1
    dv = 0.5 #km.s-1
    
    rho_cube = hdu_list_rho[0].data #g.cm-3
    T_cube = hdu_list_T[0].data #K
    vz_cube = hdu_list_vz[0].data #cm.s-1

    dz=40/1024 #pc     

    core = synth(rho=rho_cube, T=T_cube, vz=vz_cube, dz=dz)
    # cube, tau = core.gen(vmin=-40, vmax=40, dv=0.8, thin=False)
    cube_thin, tau_thin = core.gen(vmin=-40, vmax=40, dv=dv, thin=True)

    #Add noise in the synthetic cube
    noise = 0.05
    # for i in range(cube_thin.shape[1]):
    #     for j in range(cube_thin.shape[2]):
            # cube_thin[:,i,j] += np.random.randn(cube_thin.shape[0]) * noise
            # tau_thin[:,i,j] += np.random.randn(tau_thin.shape[0]) * noise

    #Try SPARK
    Tb = cube_thin[:,16,16] / np.sum(cube_thin[:,16,16]) * 100 + (np.random.randn(len(cube_thin[:,16,16])) * noise)
    tau_s = tau_thin[:,16,16] / np.sum(tau_thin[:,16,16]) * 100 + (np.random.randn(len(tau_thin[:,16,16])) * noise)

    core = lbfgs_abs(Tb=Tb, tau=tau_s, rms_Tb=noise, rms_tau=noise) 
    
    amp_fact_init = 2./3.
    sig_init = 2*dv
    iprint_init = -1
    iprint = -1
    maxiter_init = 800
    maxiter = 800
    n_gauss = 6
    lambda_Tb = 1
    lambda_tau = 1
    lambda_mu = 1
    lambda_sig = 1
    lb_amp = 0.
    ub_amp = np.max([np.max(Tb), np.max(tau_s)])
    lb_mu = 1
    ub_mu = len(tau_s)
    lb_sig = 1
    ub_sig = int(10/dv)
    rchi2_limit = 1

    result = core.new_run(n_gauss=n_gauss,
                          lb_amp=lb_amp,
                          ub_amp=ub_amp,
                          lb_mu=lb_mu,
                          ub_mu=ub_mu,
                          lb_sig=lb_sig,
                          ub_sig=ub_sig,
                          lambda_Tb=lambda_Tb,
                          lambda_tau=lambda_tau,
                          lambda_mu=lambda_mu,
                          lambda_sig=lambda_sig,
                          amp_fact_init=amp_fact_init,
                          sig_init=sig_init,
                          maxiter=maxiter,
                          maxiter_init=maxiter_init,
                          iprint=iprint,
                          iprint_init=iprint_init,
                          rchi2_limit=rchi2_limit)

    #Compute model              
    v = np.arange(vmin,vmax+dv, dv)
    res = result[-1]
    n_gauss_out = int(np.array(res[0]).shape[0] / 2 / 3)
    cube = np.moveaxis(np.array([Tb,tau_s]),0,1)
    params = np.reshape(res[0], (3*n_gauss_out, cube.shape[1]))    
    model_cube = core.model(params, cube, n_gauss_out)
    
    #Plot                                                                        
    pvalues = np.logspace(-1, 0, n_gauss_out)
    pmin = pvalues[0]
    pmax = pvalues[-1]
    
    def norm(pval):
        return (pval - pmin) / float(pmax - pmin)

    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(20,16))
    fig.subplots_adjust(hspace=0.)
    x = np.arange(cube.shape[0])
    ax1.step(v, cube[:,0], color='cornflowerblue', linewidth=2.)
    ax1.plot(v, model_cube[:,0], color='k')
    ax2.step(v, -cube[:,1], color='cornflowerblue', linewidth=2.)
    ax2.plot(v, -model_cube[:,1], color='k')
    for i in np.arange(cube.shape[1]):
        for k in np.arange(n_gauss_out):
            line = core.gaussian(x, params[0+(k*3),i], params[1+(k*3),i], 
                                 params[2+(k*3),i])
            if i == 1:
                if 
                ax2.plot(v, -line, color=plt.cm.viridis(pvalues[k]), linewidth=2.)
            else:
                ax1.plot(v, line, color=plt.cm.viridis(pvalues[k]), linewidth=2.)

    ax1.set_ylabel(r'T$_{B}$ [K]', fontsize=16)
    ax2.set_ylabel(r'$- \tau$', fontsize=16)
    ax2.set_xlabel(r'v [km s$^{-1}$]', fontsize=16)
