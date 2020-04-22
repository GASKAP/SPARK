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
    def __init__(self, rho, T, vz, thin=False):
        """
        Synthetic observation of 21cm line emission
        author: A. Marchal
        
        Parameters
        ----------
        
        Returns:
        --------
        """    
        super(synth, self).__init__()
        self.rho = rho *u.g*u.cm**-3
        self.T = T *u.K
        self.vz = vz * 1.e-5 *u.km*u.s**-1
        self.thin = thin

        # Constant
        self.m_h = 1.6737236e-27 *u.kg
        self.C = 1.82243e18 *u.K**-1 *u.cm**-2 / (u.km * u.s**-1)
        self.box_size = 40 *u.pc
        self.resolution = 1024
        self.dz = self.box_size /self.resolution
        self.dz_cm = self.dz.to(u.cm)

    def gen(self, vmin=-40, vmax=40, dv=0.8, T_lim=[0,np.inf]):        
        # Cut temperature field 
        Tk_lim_inf = T_lim[0]
        Tk_lim_sup = T_lim[1]

        idx_phase = np.where((self.T.value > Tk_lim_inf) & (self.T.value < Tk_lim_sup))
        
        rho_cube_phase = np.zeros((rho_cube.shape[0], rho_cube.shape[1], rho_cube.shape[2]))
        T_cube_phase = np.zeros((rho_cube.shape[0], rho_cube.shape[1], rho_cube.shape[2]))
        vz_cube_phase = np.zeros((rho_cube.shape[0], rho_cube.shape[1], rho_cube.shape[2]))
        
        rho_cube_phase[idx_phase] = self.rho.value[idx_phase]
        T_cube_phase[idx_phase] = self.T.value[idx_phase]
        vz_cube_phase[idx_phase] = self.vz.value[idx_phase]

        # Preliminary calculation
        Delta2 = ((const.k_B.value * T_cube_phase / self.m_h.value)) * 1.e-6 #km.s-1
        n = rho_cube_phase/(self.m_h.value*1.e3)
        n_Delta = n / np.sqrt(Delta2)

        # Spectral range
        u = np.arange(vmin,vmax+reso, reso)

        map_u = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
        for i in np.arange(T_cube_phase.shape[1]):
            for j in np.arange(T_cube_phase.shape[2]):
                map_u[:,i,j] = u

        if self.thin == True:
            Tb_thin_fast = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
            for i in tqdm(range(len(u))):
                dI = n_Delta * np.exp(- (u[i] - (vz_cube_phase))**2 / (2.*Delta2))
                dI[np.where(dI != dI)] = 0.
                Tb_thin_fast[i] = 1./(self.C.value * np.sqrt(2.*np.pi)) * np.sum(dI,0) * self.dz_cm

            return Tb_thin_fast

        else:
            Tb = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
            tau_in_front = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
            
            for i in tqdm(range(T_cube_phase.shape[0])):    
                Tb_z = np.zeros((len(u), T_cube_phase.shape[1], T_cube_phase.shape[2]))
                tau_z = 1. / (self.C.value * np.sqrt(2.*np.pi)) * n_Delta[i] / T_cube_phase[i] * np.exp(- (map_u - (vz_cube_phase[i]))**2 
                                                                                               / (2.*Delta2[i])) * self.dz_cm
                idx_nonzero = ~np.isnan(tau_z[0])
                Tb_z[:,idx_nonzero] = T_cube_phase[i,idx_nonzero] * (1. - np.exp(-1.*tau_z[:,idx_nonzero])) * np.exp(-1.*tau_in_front[:,idx_nonzero])
                
                tau_in_front[:,idx_nonzero] += tau_z[:,idx_nonzero]
                
                Tb += Tb_z 
            return Tb


if __name__ == '__main__':    
    # Open data 
    path = '/home/amarchal/SPARK/data/'
    
    hdu_list_rho = fits.open(path + 'rho_cube_sample.fits')
    hdu_list_T = fits.open(path + 'T_cube_sample.fits')
    hdu_list_vz = fits.open(path + 'vz_cube_sample.fits')
    
    #Velocity range and channel spacing
    vmin = -40 #km.s-1
    vmax = 40 #km.s-1
    reso = 0.8 #km.s-1
    
    rho_cube = hdu_list_rho[0].data #g.cm-3
    T_cube = hdu_list_T[0].data #K
    vz_cube = hdu_list_vz[0].data #m.s-1

    core = synth(rho=rho_cube, T=T_cube, vz=vz_cube, thin=True)
    foo = core.gen(vmin=-40, vmax=40, dv=0.8)
