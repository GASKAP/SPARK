# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from scipy import optimize 

import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.table as pytabs

class lbfgs_abs(object):
    def __init__(self, Tb, tau, rms_Tb=None, rms_tau=None, hdr=None):
        """
        Joint fit emission and absoption spectra for GASKAP collaboration
        author: A. Marchal
        
        Parameters
        ----------
        
        Returns:
        --------
        """    
        super(lbfgs_abs, self).__init__()
        self.Tb = Tb
        self.tau = tau
        self.rms_Tb = rms_Tb if rms_Tb is not None else 1.
        self.rms_tau = rms_tau if rms_tau is not None else 1.
        self.hdr = hdr if hdr is not None else None
        if self.hdr is not None : self.v = self.mean2vel(self.hdr["CRVAL3"]*1.e-3, self.hdr["CDELT3"]*1.e-3, 
                                                         self.hdr["CRPIX3"], np.arange(len(self.tau)))

    def run(self, n_gauss=18, lb_amp=0, ub_amp=100, lb_mu=1, ub_mu=500, lb_sig=1, ub_sig=100, lambda_Tb=1, 
            lambda_tau=1, lambda_mu=100, lambda_sig=100, iprint_init=1, amp_fact_init=0.666, sig_init=2., 
            maxiter=15000, maxiter_init=15000, iprint=1):

        #Flag test basic properties
        if len(self.Tb) != len(self.tau) : 
            print("Emission and absoption spectra must have the same size.")
            sys.exit()
        if n_gauss%3 != 0: 
            print("Please select N (number of Gaussians) such as N%3=0.")
            sys.exit()
    
        #Dimensions
        dim_v = len(self.Tb)

        #RMS cube format 
        rms = [self.rms_Tb,self.rms_tau]

        #Fit Tb spectrum 
        n_gauss_Tb = int(n_gauss * 2 / 3)
        n_gauss_tau = int(n_gauss / 3)
    
        params_Tb = self.init_spectrum(np.full((3*n_gauss_Tb),1.), n_gauss_Tb, self.Tb, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, 
                                  iprint_init, amp_fact_init, sig_init, maxiter_init)

        
        params_tau = self.init_spectrum(np.full((3*n_gauss_tau),1.), n_gauss_tau, self.tau, lb_amp, ub_amp, lb_mu, 
                                   ub_mu, lb_sig, ub_sig, iprint_init, amp_fact_init, sig_init, 
                                   maxiter_init)
         
        #Allocate and init arrays
        cube = np.moveaxis(np.array([self.Tb,self.tau]),0,1)
        params = np.full((3*n_gauss,2),1.)
        bounds = self.init_bounds(cube, params, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig)
        
        #Copy result as init of the absorption spectrum 
        params[:3*n_gauss_Tb,0] = params_Tb
        params[:3*n_gauss_Tb,1] = params_Tb
        params[3*n_gauss_Tb:,0] = params_tau
        params[3*n_gauss_Tb:,1] = params_tau

        #Update both with regularization
        result = optimize.fmin_l_bfgs_b(self.f_g, params.ravel(), args=(n_gauss, cube, rms, lambda_Tb, lambda_tau, 
                                                                   lambda_mu, lambda_sig), 
                                        bounds=bounds, approx_grad=False, disp=iprint, maxiter=maxiter)
        
        # params = np.reshape(result[0], (3*n_gauss, cube.shape[1]))   

        return result, params_tau, params_Tb


    def mean2vel(self, CRVAL, CDELT, CRPIX, mean):                                                                            
        return [(CRVAL + CDELT * (mean[i] - CRPIX)) for i in range(len(mean))]       


    def init_bounds_spectrum(self, n_gauss, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig):
        bounds_inf = np.zeros(3*n_gauss)
        bounds_sup = np.zeros(3*n_gauss)
        
        bounds_inf[0::3] = lb_amp
        bounds_inf[1::3] = lb_mu
        bounds_inf[2::3] = lb_sig
        
        bounds_sup[0::3] = ub_amp
        bounds_sup[1::3] = ub_mu
        bounds_sup[2::3] = ub_sig
        
        return [(bounds_inf[i], bounds_sup[i]) for i in np.arange(len(bounds_sup))]


    def init_bounds(self, cube, params, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig):
        bounds_inf = np.zeros(params.shape)
        bounds_sup = np.zeros(params.shape)

        for i in range(cube.shape[1]):
            for k in np.arange(params.shape[0]/3):
                bounds_A = [lb_amp, ub_amp]
                bounds_mu = [lb_mu, ub_mu]
                bounds_ampma = [lb_sig, ub_sig]

                bounds_inf[int(0+(3*k)),i] = bounds_A[0]
                bounds_inf[int(1+(3*k)),i] = bounds_mu[0]
                bounds_inf[int(2+(3*k)),i] = bounds_ampma[0]
                
                bounds_sup[int(0+(3*k)),i] = bounds_A[1]
                bounds_sup[int(1+(3*k)),i] = bounds_mu[1]
                bounds_sup[int(2+(3*k)),i] = bounds_ampma[1]
            
        return [(bounds_inf.ravel()[i], bounds_sup.ravel()[i]) for i in np.arange(len(bounds_sup.ravel()))]


    def f_g(self, pars, n_gauss, data, rms, lambda_Tb, lambda_tau, lambda_mu, lambda_sig):
        params = np.reshape(pars, (3*n_gauss, data.shape[1]))
        
        x = np.arange(data.shape[0])
        
        model = np.zeros(data.shape)
        dF_over_dB = np.zeros((params.shape[0], data.shape[0], data.shape[1]))
        product = np.zeros((params.shape[0], data.shape[0], data.shape[1]))
        deriv = np.zeros((params.shape[0], data.shape[1]))

        for i in np.arange(data.shape[1]):
            for k in np.arange(n_gauss):
                model[:,i] += self.gaussian(x, params[0+(k*3),i], params[1+(k*3),i], params[2+(k*3),i])
                
                dF_over_dB[0+(k*3),:,i] += (1. 
                                            * np.exp((-(x - params[1+(k*3),i])**2)/(2.* (params[2+(k*3),i])**2)))
                dF_over_dB[1+(k*3),:,i] += (params[0+(k*3),i] * (x - params[1+(k*3),i]) / (params[2+(k*3),i])**2 
                                            * np.exp((-(x - params[1+(k*3),i])**2)/(2.* (params[2+(k*3),i])**2)))
                dF_over_dB[2+(k*3),:,i] += (params[0+(k*3),i] * (x - params[1+(k*3),i])**2 / (params[2+(k*3),i])**3 
                                            * np.exp((-(x - params[1+(k*3),i])**2)/(2.* (params[2+(k*3),i])**2)))                

        F = model - data   
        
        F[:,0] /= rms[0]
        F[:,1] /= rms[1]
        
        for i in np.arange(data.shape[1]):
            for v in np.arange(data.shape[0]):
                if i ==0 :
                    product[:,v,i] = lambda_Tb * dF_over_dB[:,v,i] * F[v,i]
                else:
                    product[:,v,i] = lambda_tau * dF_over_dB[:,v,i] * F[v,i]

                        
        deriv = np.sum(product, axis=1)

        J = (0.5 * lambda_Tb * np.sum(F[:,0]**2)) + (0.5 * lambda_tau * np.sum(F[:,1]**2))

        R_mu = 0.5 * lambda_mu * np.sum((params[1::3,1] / params[1::3,0] - 1.)**2)
        R_sig = 0.5 * lambda_sig * np.sum((params[2::3,1] / params[2::3,0] - 1.)**2)
        
        deriv[1::3,0] = deriv[1::3,0] - (lambda_mu * params[1::3,1] / params[1::3,0]**2.
                                         * (params[1::3,1] / params[1::3,0] - 1.))
        
        deriv[1::3,1] = deriv[1::3,1] + (lambda_mu / params[1::3,0]
                                         * (params[1::3,1] / params[1::3,0] - 1.))
        
        deriv[2::3,0] = deriv[2::3,0] - (lambda_sig *  params[2::3,1] / params[2::3,0]**2.
                                         * (params[2::3,1] / params[2::3,0] - 1.))
        
        deriv[2::3,1] = deriv[2::3,1] + (lambda_sig / params[2::3,0]
                                         * (params[2::3,1] / params[2::3,0] - 1.))
        
        return J + R_mu + R_sig, deriv.ravel()

    
    def f_g_spectrum(self, params, n_gauss, data):
        x = np.arange(data.shape[0])
        
        model = np.zeros(data.shape[0])
        dF_over_dB = np.zeros((params.shape[0], data.shape[0]))
        product = np.zeros((params.shape[0], data.shape[0]))
        deriv = np.zeros((params.shape[0]))
        
        for k in np.arange(n_gauss):
            model += self.gaussian(x, params[0+(k*3)], params[1+(k*3)], params[2+(k*3)])
            
            dF_over_dB[0+(k*3),:] += (1. 
                                      * np.exp((-(x - params[1+(k*3)])**2)/(2.* (params[2+(k*3)])**2)))
            dF_over_dB[1+(k*3),:] += (params[0+(k*3)] * (x - params[1+(k*3)]) / (params[2+(k*3)])**2 
                                      * np.exp((-(x - params[1+(k*3)])**2)/(2.* (params[2+(k*3)])**2)))
            dF_over_dB[2+(k*3),:] += (params[0+(k*3)] * (x - params[1+(k*3)])**2 / (params[2+(k*3)])**3 
                                      * np.exp((-(x - params[1+(k*3)])**2)/(2.* (params[2+(k*3)])**2)))                
    
        F = model - data                
                
        for v in np.arange(data.shape[0]):
            product[:,v] = dF_over_dB[:,v] * F[v]
                        
        deriv = np.sum(product, axis=1)
        
        J = 0.5*np.sum(F**2)
        
        return J, deriv.ravel()


    def init_spectrum(self, params, n_gauss, data, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig, 
                      iprint, amp_fact_init, sig_init, maxiter):
        for i in np.arange(n_gauss):
            n = i+1
            x = np.arange(data.shape[0])
            bounds = self.init_bounds_spectrum(n, lb_amp, ub_amp, lb_mu, ub_mu, lb_sig, ub_sig)
            model = np.zeros(data.shape[0])
            
            for k in np.arange(n):
                model += self.gaussian(x, params[0+(k*3)], params[1+(k*3)], params[2+(k*3)])

            residual = model - data

            xx = np.zeros((3*n,1))
            for p in np.arange(3*n):
                xx[p] = params[p]
                
            xx[1+(i*3)] = np.where(residual == np.min(residual))[0][0]   
            xx[0+(i*3)] = data[int(xx[1+(i*3)])] * amp_fact_init
            xx[2+(i*3)] = sig_init

            result = optimize.fmin_l_bfgs_b(self.f_g_spectrum, xx, args=(n, data), 
                                        bounds=bounds, approx_grad=False, disp=iprint, maxiter=maxiter)
    
            for p in np.arange(3*n):
                params[p] = result[0][p]
            
        return params


    def gaussian(self, x, amp, mu, sig):
        return amp * np.exp(-((x - mu)**2)/(2. * sig**2))


    def model_spectrum(self, params, data, n_gauss):
        x = np.arange(data.shape[0])
        model = np.zeros(len(x))        
        
        for k in np.arange(n_gauss):
            model += self.gaussian(x, params[0+(k*3)], params[1+(k*3)], params[2+(k*3)])

        return model


    def model(self, params, data, n_gauss):
        x = np.arange(data.shape[0])
        model = np.zeros(data.shape)
        
        for i in np.arange(data.shape[1]):
            for k in np.arange(n_gauss):
                model[:,i] += self.gaussian(x, params[0+(k*3),i], params[1+(k*3),i], params[2+(k*3),i])
            
        return model


if __name__ == '__main__':    
    print("lbfgs_abs module")
    core = lbfgs_abs(np.zeros(30), np.zeros(30))
    
    path="/mnt/raid-cita/amarchal/21SPONGE/"
    # filename = "all_sponge_sources_table_tighter.fits"
    filename = "sponge_pepsi_challenge_results.fits"

    name = '3C225A'
    # name = '3C154'

    cat = fits.getdata(path+filename)       
    data_s = pytabs.Table(cat)
    idx_absline = np.where(data_s["NAMES"] == name)[0][0]

    # v = data_s[idx_absline]["VEL"][600:1200]
    # Tb = data_s[idx_absline]["TB"][600:1200] / 100
    # tau = data_s[idx_absline]["TAU"][600:1200]
    # rms_Tb= np.std(Tb[10:30]) /100
    # rms_tau = np.std(tau[10:30])
    v = data_s[idx_absline]["VEL"]

    rms_Tb= np.std(data_s[idx_absline]["TB"][370:400])
    rms_tau = np.std(data_s[idx_absline]["TAU"][50:80])

    Tb = data_s[idx_absline]["TB"] / np.nansum(data_s[idx_absline]["TB"]) * 100#(1. / rms_Tb)
    tau = data_s[idx_absline]["TAU"] / np.nansum(data_s[idx_absline]["TAU"]) * 100#(1. / rms_tau)

    # Tb = Tb[600:1200]
    # tau = tau[600:1200]
    # v = v[600:1200]

    #Channel spacing
    dv = np.diff(v)[0]

    #hdr
    hdr=fits.Header()
    hdr["CDELT3"] = dv
    hdr["CRPIX3"] = 0
    hdr["CRVAL3"] = v[0]*1.e3
    
    #parameters                                                                                                                                                               
    amp_fact_init = 2./3.
    sig_init = 2.
    iprint_init = 1
    iprint = 1
    maxiter_init = 200
    maxiter = 200
    n_gauss = 18
    lambda_Tb = 1        
    lambda_tau = 1
    lambda_mu = 0
    lambda_sig = 0
    lb_amp = 0.
    ub_amp = np.max(Tb)
    lb_mu = 1
    ub_mu = len(tau)
    lb_sig = 1
    ub_sig = 30
    
    core = lbfgs_abs(Tb=Tb, tau=tau, hdr=hdr)
    # core = lbfgs_abs(Tb=Tb, tau=tau, rms_Tb=rms_Tb, rms_tau=rms_tau, hdr=hdr)
    
    result, params_tau, params_Tb = core.run(n_gauss=n_gauss,
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
                                             iprint_init=iprint_init)
    
    print("J =",result[1])

    stop
    
    #Compute model                                                                                                                                                            
    cube = np.moveaxis(np.array([Tb,tau]),0,1)
    params = np.reshape(result[0], (3*n_gauss, cube.shape[1]))
    vfield_Tb = core.mean2vel(hdr["CRVAL3"]*1.e-3, hdr["CDELT3"], hdr["CRPIX3"], 
                              params[1::3,0])
    vfield_tau = core.mean2vel(hdr["CRVAL3"]*1.e-3, hdr["CDELT3"], hdr["CRPIX3"], 
                               params[1::3,1])
    
    model_cube = core.model(params, cube, n_gauss)

    
    #Plot                                                                                                                                                                                                                                                             
    pvalues = np.logspace(-1, 0, n_gauss)
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
        for k in np.arange(n_gauss):
            line = core.gaussian(x, params[0+(k*3),i], params[1+(k*3),i], 
                                 params[2+(k*3),i])
            if i == 1:
                ax2.plot(v, -line, color=plt.cm.viridis(pvalues[k]), linewidth=2.)
            else:
                ax1.plot(v, line, color=plt.cm.viridis(pvalues[k]), linewidth=2.)

    ax1.set_ylabel(r'T$_{B}$ [K]', fontsize=16)
    ax2.set_ylabel(r'$- \tau$', fontsize=16)
    ax2.set_xlabel(r'v [km s$^{-1}$]', fontsize=16)
