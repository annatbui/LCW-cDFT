#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------
    LCW-cDFT for Solvation
    Copyright (C) 2023  Anna T. Bui

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
-------------------------------------------------------------------------------

Created October 2022. Last updated November 2023.
Author: Anna T. Bui
Email: btb32@cam.ac.uk

LCW-cDFT for Solvation Program.
Calculate solvent density profiles and solvation free energies
for non-polar spherical solutes.

Paper: A classical density functional theory for solvation across length scale
Author: Anna T. Bui, Stephen J. Cox
Available at: *** ADD URL ***

Supported fluids and state points:
    SPC/E water
    RPBE-D3 water
    mW water 
    mW water

Supported external potentials for spherical geometry are:
    Hard Wall (HW)
    Lennard-Jones (LJ)
    Attractive potential (SLJ)
-------------------------------------------------------------------------------
"""
from IPython import display
import argparse
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


#####################
# GLOBAL CONVERSION #
#####################
# Scientific constants
kB         = 1.38064852e-23  # unit is J/K
NA          = 6.0221409e23  # unit is 1/mol

m_convert     = 1e3*1e24/(NA*NA) # from kJ mol^-2 cm^3 angstrom^2 to J angstrom^5
a_convert     = 1e3*1e24/(NA*NA) # from kJ cm^3 mol^-2 to # J angstrom^3
d_convert     = 1.0              # from angstrom to angstrom
gamma_convert = 1e-23


def get_arguments():
    # adds description of the program.
    parser = argparse.ArgumentParser(
        description="Code for cDFT."
    )

    parser.add_argument(
        "-in",
        "--input_file",
        required=True,
        type=str,
        default=None,
        help="the relative path to input file"
    )

    parser.add_argument(
        "-out",
        "--output_file",
        required=False,
        type=str,
        default="final.out",
        help="the relative path to output file"
    )
    return parser.parse_args()


def load_input(filename):
    '''Load input files and extract parameter'''
    with open(filename) as fh:
        for line in fh:
            line = line.partition('#')[0]
            line = line.rstrip()
            words = line.split()
            if len(words) > 0:
                if   words[0] == 'bulk_density':
                    rho_bulk    = float(words[-1])
                elif words[0] == 'temperature':
                    T           = float(words[-1])
                elif words[0] == 'chemical_potential':
                    delta_mu    = float(words[-1])
                elif words[0] == 'liquid_coex_density':
                    liquid_coex = float(words[-1])
                elif words[0] == 'vapor_coex_density':
                    vapor_coex  = float(words[-1])
                elif words[0] == 'surface_tension':
                    gamma       = float(words[-1])
                elif words[0] == 'interfacial_thickness':
                    d           = float(words[-1])
                elif words[0] == 'lambda':
                    coarse_grain_length = float(words[-1])
                elif words[0] == 'a':
                    a           = float(words[-1])
                elif words[0] == 'dcf_kspace':
                    dcf_file    = words[-1]
                elif words[0] == 'HS_radius':
                    HS_RADIUS   = float(words[-1])
                elif words[0] == 'initial_guess':
                    initial_guess = words[-1]
    
            
    return  rho_bulk, T, delta_mu, liquid_coex, vapor_coex, \
            gamma, d, coarse_grain_length, a, dcf_file, HS_RADIUS, initial_guess 


def get_data(path_to_data):
    '''
    Load direct correlation function
    '''
    data = np.loadtxt(path_to_data, skiprows=1)
    return data[:, 0], data[:, 1]


def place(ax):
  ax.tick_params(direction="in", which="minor", length=3)
  ax.tick_params(direction="in", which="major", length=5, labelsize=13)
  ax.grid(which="major", ls="dashed", dashes=(1, 3), lw=1, zorder=0)

def plot_test(function):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.2))
    ax.plot(r_, function, lw=1.6, color='blue')
    place(ax)
    
def HS_solute(r, radius):
    '''
    Solute external potential
    '''
    energy = r*0+1e100
    energy[r >= radius] = 0
    return energy  

def get_rFT(h, r):
    '''
    Radial Fourier Transform from real to reciprocal space
    '''
    result = []
    for k in k_:
        integrand = 4 * np.pi *r * h * np.sin(k*r) / k
        result.append(integrate.simpson(integrand, r))
    h_k = np.array(result) 
    return h_k


def get_invrFT(f, k):
    '''
    Radial Inverse Fourier Transform from reciprocal to real space
    '''
    result = []
    for r in r_:
        integrand = k * f * np.sin(k*r) / (r * 2 * np.pi * np.pi) 
        result.append(integrate.simpson(integrand, k))
    f_r = np.array(result) 
    f_r[4000:] = f_r[4000]
    return f_r 


def Gaussian_in_kspace(k, sigma):
    return np.exp(-0.5*np.power(k,2)*np.power(sigma,2))


def Gaussian_in_rspace(r, sigma):
    return np.exp(-r**2 /(2 * sigma**2)) / np.power((sigma * np.sqrt(2 * np.pi)),3)

def w(n):
    energy = 2 * m * np.power(n-liquid_coex,2) * np.power(n-vapor_coex,2)/ (np.power(d,2)*np.power(liquid_coex-vapor_coex,2)) - n*delta_mu
    return energy

def w_prime(n):
    first_prefactor = 4 * m / np.power(d,2) * np.power(liquid_coex-vapor_coex, 2)
    first_bracket   = np.power(n-liquid_coex,2) * (n-vapor_coex) + np.power(n-vapor_coex,2) * (n-liquid_coex)
    second_term     = - delta_mu
    return first_prefactor * first_bracket + second_term
    


def compute_full_density(rho_slow, rho_guess, ratio):
    '''
    Compute full density by minimisation of the functional
    '''

    # initial guess
    rho_trial = rho_guess
    rho_old   = np.zeros(r_.shape)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_ylim(-0.01,4)
    
    place(ax)

    # iterative loop
    while np.allclose(rho_trial,rho_old, rtol=1e-4, atol=1e-5) is False:

        ax.plot(r_, rho_trial/rho_bulk)
        display.display(fig)
        display.clear_output(wait=True)
        
        rho_old = rho_trial

        
        delta_rho_rho_slow_r = (rho_old - rho_slow) * rho_slow 
        delta_rho_rho_slow_k = get_rFT(delta_rho_rho_slow_r, r_)
        
        pre_gamma_k =  delta_rho_rho_slow_k * c_k
        pre_gamma_r =  get_invrFT(pre_gamma_k, k_)
        
        gamma = pre_gamma_r * rho_slow/ np.power(rho_bulk, 2)        
        
        # compute RHS
        rho_new = rho_slow * np.exp(-beta*HS_solute(r_, HS_RADIUS) + ratio * gamma)
        
        # update to new density
        rho_trial = rho_old * (1-alpha_full) + rho_new * alpha_full
    
    rho_final = rho_trial
    
    return rho_final


def compute_slow_density(rho_full, rho_guess):
    '''
    Compute slow density according to vDW with unbalanced force from given full density, guess from previous iteration
    '''
    
    # Coarse grain the full density
    rho_full_bar_k = get_rFT(rho_full, r_) * Gaussian_in_kspace(k_, coarse_grain_length)
    rho_full_bar_r = get_invrFT(rho_full_bar_k, k_)
    

    # initial guess
    rho_trial = rho_guess
    rho_old = np.zeros(r_.shape)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_ylim(-1,4)
    place(ax)
    
    # iterative loop
    i = 0
    while np.allclose(rho_trial,rho_old, rtol=5e-4, atol=5e-5) is False:
        
        ax.plot(r_, rho_trial/rho_bulk)
        display.display(fig)
        display.clear_output(wait=True)
        
        rho_old = rho_trial
        
           
        rho_slow_bar_k = get_rFT(rho_old, r_) * Gaussian_in_kspace(k_, coarse_grain_length)
        rho_slow_bar_r = get_invrFT(rho_slow_bar_k, k_)        
        
        first_term = a*np.power(coarse_grain_length,2)*rho_full_bar_r/m 
        second_term = 0.5 * (w_prime(rho_old) - 0*w_prime(rho_bulk))*np.power(coarse_grain_length,2)/ m
        third_term = (1 -(a*np.power(coarse_grain_length,2)/m ))*rho_slow_bar_r
        
        rho_new = first_term - second_term + third_term
        
         # Update to new density
        rho_trial = rho_old * (1 - alpha_slow) + rho_new * alpha_slow
      
        i = i + 1
        
    rho_final = rho_trial   
    
    # return new slowly varying density
    return rho_final


def free_energy_large(rho_s, rho_f):
    '''
    Returns Free energy from van der Waals functional + unbalanced energy between two densities
    '''
    
    #  van der Waals functional
    local_term  = w(rho_s) - w(rho_bulk)*0
    local_term[rho_s > rho_bulk] = 0
    integrand   = r_ * r_ * local_term
    free_energy_local =  4 * np.pi * integrate.simpson(integrand, r_)
    
    gradient_term = 0.5 * m * np.power(np.gradient(rho_s), 2)
    gradient_term[rho_s > rho_bulk] = 0
    integrand   = r_ * r_ * gradient_term
    free_energy_gradient =  4 * np.pi * integrate.simpson(integrand, r_)
    
    #  unbalanced energy
    delta_rho_bar_k = get_rFT(rho_f - rho_s, r_) * Gaussian_in_kspace(k_, coarse_grain_length)
    delta_rho_bar_r = get_invrFT(delta_rho_bar_k, k_)
    integrand       = r_ * r_ * (-2 * a * delta_rho_bar_r) * rho_s
    integrand[rho_s > rho_bulk] = 0
    free_energy_u   = 4 * np.pi * integrate.simpson(integrand, r_)   
    
   
    return free_energy_local*1e-3*NA, free_energy_gradient*1e-3*NA, free_energy_u*1e-3*NA


def free_energy_small(rho_f, rho_s):
    '''
    Return the energy to insert a solute
    '''
    
    # ideal term
    rho_s[rho_s==0] = 1e-23
    ratio           = rho_f/rho_s
    ratio[ratio==0] = 1
    integrand       =  r_ * r_ * (rho_f * np.log(ratio) - rho_f + rho_s)
    free_energy_id  = 4 * np.pi * kB * T * integrate.simpson(integrand, r_)
    
    # external term
    integrand       = r_ * r_ * HS_solute(r_, HS_RADIUS) * rho_f
    free_energy_ext = 4 * np.pi * integrate.simpson(integrand, r_)
    
    # excess term
    
    delta_rho_rho_slow_r = (rho_f - rho_s) * rho_s 
    delta_rho_rho_slow_k = get_rFT(delta_rho_rho_slow_r, r_)
    
    pre_gamma_k          =  delta_rho_rho_slow_k * c_k
    pre_gamma_r          =  get_invrFT(pre_gamma_k, k_)
    
    gamma                = pre_gamma_r * rho_s / np.power(rho_bulk, 2)
    
    integrand            =  r_ * r_ * (rho_f - rho_s)  * gamma
    free_energy_exc      = -0.5 * kB * T * 4 * np.pi * integrate.simpson(integrand, r_)
    
    return free_energy_id*1e-3*NA, free_energy_ext*1e-3*NA, free_energy_exc*1e-3*NA
    
def write_out_data(filename, full_density_final, slow_density_final):
    '''
    Write out data
    '''
    header_text = ''' Output from LCW-MDFT minimisation \n \n############## \n## DENSITY ### \n############## \n r [angtrom] | full_density [angstrom^-3] | slow_density [angstrom^-3] \n '''
    
    
    np.savetxt(filename, np.c_[r_,full_density_final,slow_density_final], header=header_text)
    
    F_local, F_gradient, F_u = free_energy_large(slow_density_final, full_density_final)
    F_id, F_ext, F_exc       = free_energy_small(full_density_final, slow_density_final)
    
    F_large = F_local + F_gradient + F_u
    F_small = F_id + F_ext + F_exc - F_u
    
    F_solv = F_small + F_large
    F_solv_area = 1e3 * F_solv/(1e-3 * NA * 4 * np.pi * 1e-20 * np.power(HS_RADIUS,2))
    
    with open(filename, "a") as myfile:
        myfile.write("#\n################## \n")
        myfile.write("### PARAMETERS ### \n")
        myfile.write("################## \n#\n")
        myfile.write("# water model                                = SPC/E\n")
        myfile.write("# rho_bulk [angstrom^-3]                     = {}\n".format(rho_bulk))
        myfile.write("# a [kJ cm^3 mol^-2]                         = {}\n".format(a/a_convert))
        myfile.write("# m [kJ mol^-2 cm^3 angstrom^2]              = {}\n".format(m/m_convert))
        myfile.write("# lambda [angstrom]                          = {}\n".format(coarse_grain_length))
        myfile.write("# d  [angstrom]                              = {}\n".format(d))
        myfile.write("# liquid density [angstrom^-3]               = {}\n".format(liquid_coex))
        myfile.write("# gas density [angstrom^-3]                  = {}\n".format(vapor_coex))
        myfile.write("# surface tension [mN/m^2]                   = {}\n".format(gamma))
        myfile.write("# DCF approximation                          = interpolation\n")
        
        myfile.write("#\n############## \n")
        myfile.write("### SOLUTE ### \n")
        myfile.write("############## \n#\n")        
        myfile.write("# HS radius [angstrom]                       = {}\n".format(HS_RADIUS))
        
        myfile.write("#\n################### \n")
        myfile.write("### FREE ENERGY ### \n")
        myfile.write("################### \n#\n")
        myfile.write("# Local van der Waals [kJ/mol]               = {}\n".format(F_local))
        myfile.write("# Gradient van der Waals [kJ/mol]            = {}\n".format(F_gradient))
        myfile.write("# Unbalanced energy [kJ/mol]                 = {}\n".format(F_u))
        myfile.write("# Total large length scale term [kJ/mol]     = {}\n".format(F_large))        
        myfile.write("# \n")
        myfile.write("# Ideal term [kJ/mol]                        = {}\n".format(F_id))
        myfile.write("# External term [kJ/mol]                     = {}\n".format(F_ext))
        myfile.write("# Unbalanced term [kJ/mol]                   = {}\n".format(-F_u))
        myfile.write("# Excess term [kJ/mol]                       = {}\n".format(F_exc))
        myfile.write("# Total small length scale term [kJ/mol]     = {}\n".format(F_small))
        myfile.write("# \n")
        myfile.write("# Total free energy of solvation [kJ/mol]    = {}\n".format(F_solv))
        myfile.write("# Free energy of solvation per area [mJ/m^2] = {}\n".format(F_solv_area))
        
################
# MAIN PROGRAM #
################

if __name__ == "__main__":
    '''
    MAIN FUNCTION
    '''
    
    # start by getting arguments 
    args = get_arguments()
    path_to_input = args.input_file
    path_to_output = args.output_file
    
    # essenial inputs
    rho_bulk, T, delta_mu, liquid_coex, vapor_coex, \
    gamma, d, coarse_grain_length, a, dcf_file, HS_RADIUS, initial_guess = load_input(path_to_input)
    alpha_full = 0.05
    alpha_slow = 0.15
    
    # prepare grid space
    k_cutoff = 2000
    r_cutoff = 5000
    k_, c_k = get_data(dcf_file)
    r_ = np.linspace(5e-3,50,5000)
   
    k_, c_k = k_[:k_cutoff], c_k[:k_cutoff]
    r_ = r_[:r_cutoff]
    dr = r_[1] - r_[0]
    r_extend = np.arange(r_[-1]+dr,50,dr)
    r_ = np.concatenate([r_, r_extend])
    
    # unit conversion
    beta = 1/(kB * T)
    delta_mu = delta_mu * kB * T
    gamma = gamma * gamma_convert
    d = d * d_convert
    a = a * a_convert
    m = 3 * d * gamma / np.power(liquid_coex-vapor_coex,2)


    
    # FIRST ITERATION INITIALISATION
    if initial_guess != 'bulk':
        distance = float(initial_guess)
        rho_guess = 0.5*((liquid_coex + vapor_coex)+(liquid_coex - vapor_coex)*np.tanh((r_-HS_RADIUS+distance)/d))
    else:
        rho_guess = rho_bulk
    
    
    full_density_guess = rho_guess * np.exp(-beta*HS_solute(r_, HS_RADIUS))
    slow_density_guess = np.ones(r_.shape) * rho_guess
    
    slow_density_old   = slow_density_guess
 
    full_density_new   = compute_full_density(slow_density_guess, full_density_guess, 0.5)
    slow_density_guess_k = get_rFT(full_density_new, r_) * Gaussian_in_kspace(k_, 1)
    slow_density_guess_r = get_invrFT(slow_density_guess_k, k_)
        
    slow_density_new   = compute_slow_density(full_density_new, slow_density_guess_r)
    slow_density_trial = slow_density_new
    slow_density_old = 0*slow_density_trial
    
    # ITERATION LOOP: GUESS SLOW AND FULL DENSITY
    i = 1
    while np.allclose(slow_density_trial, slow_density_old,  rtol=8e-4, atol=5e-5) is False:
        
        slow_density_old = slow_density_trial
        full_density_new = compute_full_density(slow_density_old, full_density_new, 1)
        slow_density_new = compute_slow_density(full_density_new, slow_density_new)
        slow_density_trial = slow_density_new
    
    slow_density_final = slow_density_new
    full_density_final = full_density_new
    
    # WRITE OUT DATA
    write_out_data(path_to_output, full_density_final, slow_density_final)
    

    

    

    
