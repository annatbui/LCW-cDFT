#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------
    
    LCW-cDFT
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

    Created October 2022. Last updated Sep 2024.
    Author: Anna T. Bui
    Email: btb32@cam.ac.uk

    LCW-cDFT for Solvation Program.
        Calculate solvent density profiles and solvation free energies
        for non-polar spherical solutes.

    Paper: A classical density functional theory for solvation across length scales
        Authors: Anna T. Bui, Stephen J. Cox
        Available at: https://doi.org/10.1063/5.0223750

    Supported fluids and state points:
        SPC/E water, 300 K
        RPBE-D3 water, 300 K
        mW water, 300 K 
        mW water, 426 K

    Supported external potentials for spherical geometry are:
        Hard sphere
        Lennard-Jones
        Attractive potential 
        
-------------------------------------------------------------------------------
"""
from IPython import display
import argparse
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

params = {"axes.labelsize": 14,
          "axes.titlesize": 25,}
plt.rcParams["axes.linewidth"] = 1
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
plt.rcParams['figure.dpi'] = 100
plt.rcParams.update(params)


# Scientific constants
kB            = 1.38064852e-23   # J/K
NA            = 6.0221409e23     # 1/mol

# Unit conversion
m_convert     = 1e3*1e24/(NA*NA) # from kJ mol^-2 cm^3 AA^2 to J AA^5
a_convert     = 1e3*1e24/(NA*NA) # from kJ cm^3 mol^-2 to # J AA^3
c_convert     = 1e9/NA           # from J mol^-1 nm^9 to J AA^9
d_convert     = 1e15/NA          # from J mol^-1 nm^15 to J AA^15
kcal_to_J     = 4184
infty         = 1e100
J_to_kJmol    = 1e-3 * NA

def get_arguments():
    '''
    Get argument from command line.
    '''
    parser = argparse.ArgumentParser(
        description="Enter input and output files for LCW-cDFT"
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

def place(ax):
    '''
    Make axis looks pretty
    '''
    ax.tick_params(direction="in", which="minor", length=3)
    ax.tick_params(direction="in", which="major", length=5, labelsize=13)
    ax.grid(which="major", ls="dashed", dashes=(1, 3), lw=1, zorder=0)
    
    
def load_input(filename):
    '''
    Load input files and extract parameters
    Return all neccessary input for program
    '''
    # Default parameters
    rho_bulk            = 0.03323521
    temperature         = 300
    delta_mu            = 1e-3
    liquid_coex         = 0.033234
    vapor_coex          = 4.747e-7
    SG_c                = 1.096
    SG_d                = 0
    SG_m_min            = 1000
    SG_m_max            = 1400
    SG_rho_s_min        = 0.0
    SG_rho_s_max        = 0.03323521
    cg_length           = 0.85
    a                   = 200
    dcf_file            = '../parameters/dcf_ck_spce_rho_u.txt'
    initial_guess       = 'bulk'
    alpha_full          = 0.05
    alpha_slow          = 0.15
    rtol                = 5e-5
    atol                = 5e-6 
    k_cutoff            = 2000
    max_FT              = 25
    r_max               = 30
    r_min               = 5e-3
    dr                  = 0.01
    solute_type         = 'HS'
    HS_radius           = 2.0
    LJ_sigma            = 1.0
    LJ_epsilon          = 1.0
    ATT_epsilon         = 1.0
    ATT_sigma           = 1.0
    ATT_radius          = 1.0

    
    
    with open(filename) as fh:
        for line in fh:
            line = line.partition('#')[0]
            line = line.rstrip()
            words = line.split()
            if len(words) > 0:
                if   words[0] == 'bulk_density':
                    rho_bulk    = float(words[-1])
                elif words[0] == 'solute_type':
                    solute_type = words[-1]
                elif words[0] == 'temperature':
                    temperature = float(words[-1])
                elif words[0] == 'chemical_potential':
                    delta_mu    = float(words[-1])
                elif words[0] == 'liquid_coex_density':
                    liquid_coex = float(words[-1])
                elif words[0] == 'vapor_coex_density':
                    vapor_coex  = float(words[-1])
                elif words[0] == 'c':
                    SG_c       = float(words[-1])
                elif words[0] == 'd':
                    SG_d       = float(words[-1])
                elif words[0] == 'm_min':
                    SG_m_min  = float(words[-1])
                elif words[0] == 'm_max':
                    SG_m_max  = float(words[-1])
                elif words[0] == 'rho_s_min':
                    SG_rho_s_min = float(words[-1])
                elif words[0] == 'rho_s_max':
                    SG_rho_s_max = float(words[-1])
                elif words[0] == 'lambda':
                    cg_length = float(words[-1])
                elif words[0] == 'a':
                    a           = float(words[-1])
                elif words[0] == 'dcf_kspace':
                    dcf_file    = words[-1]
                elif words[0] == 'initial_guess':
                    initial_guess = words[-1]    
                elif words[0] == 'HS_radius':
                    HS_radius   = float(words[-1])
                elif words[0] == 'LJ_sigma':
                    LJ_sigma   = float(words[-1])
                elif words[0] == 'LJ_epsilon':
                    LJ_epsilon   = float(words[-1])
                elif words[0] == 'ATT_sigma':
                    ATT_sigma   = float(words[-1])
                elif words[0] == 'ATT_epsilon':
                    ATT_epsilon   = float(words[-1])
                elif words[0] == 'ATT_radius':
                    ATT_radius   = float(words[-1])
                elif words[0] == 'r_max':
                    r_max = float(words[-1])
                elif words[0] == 'r_min':
                    r_min = float(words[-1])
                elif words[0] == 'max_FT':
                    max_FT = float(words[-1])
                elif words[0] == 'dr':
                    dr = float(words[-1])                
            
    return  rho_bulk, temperature, delta_mu, liquid_coex, vapor_coex, \
            SG_c, SG_d, SG_m_min, SG_m_max, SG_rho_s_min, SG_rho_s_max, cg_length, a, dcf_file, \
            initial_guess, alpha_full, alpha_slow, rtol, atol, \
            k_cutoff, dr, r_min, r_max, max_FT, solute_type, \
            HS_radius, LJ_sigma, LJ_epsilon, ATT_epsilon, ATT_sigma, ATT_radius



def get_dcf(path_to_data):
    '''
    Load direct correlation function
    '''
    data = np.loadtxt(path_to_data, skiprows=1)
    return data[:, 0], data[:, 1]



def ATT_solute(r, epsilon_sf, sigma_s, Rs):
    '''
    Lennard Jones wall
        epsilon: in kcal mol^-1 
        sigma: in angstrom
    '''
    rmin = sigma_s*(2/5)**(1/6)
    rp = rmin + r
    power3 = (1/(rp + Rs))**3 - (1/(rp - Rs))**3
    power9 = (1/(rp - Rs))**9 - (1/(rp + Rs))**9
    power2 = (1/(rp - Rs))**2 - (1/(rp + Rs))**2
    power8 = (1/(rp + Rs))**8 - (1/(rp - Rs))**8
    
    energy = epsilon_sf * (2*np.power(sigma_s,9)*power9/15 \
                        + 3*np.power(sigma_s,9)*power8/(20*rp) \
                        + np.power(sigma_s,3)*power3 \
                        +  3*np.power(sigma_s,3)*power2/(2*rp) )
    energy[r <  Rs] = infty
    return energy * kcal_to_J / NA 


def LJ_solute(r, epsilon, sigma):
    '''
    Lennard Jones solute
        epsilon: in kcal mol^-1 
        sigma: in angstrom
    '''
    rc = sigma + 30
    power6 = (sigma/r)**6
    power12 = power6**2
    shift = 4*epsilon*((sigma/rc)**12 - (sigma/rc)**6)
    energy = 4*epsilon*(power12-power6) - shift
    energy[r >= rc] = 0
    return energy * kcal_to_J / NA


def HS_solute(r, radius):
    '''
    Solute external potential
    '''
    energy = r*0 + infty
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
    f_r[array_len_last:] = f_r[array_len_last]
    return f_r 


def Gaussian_kspace(k, sigma):
    '''
    Gaussian function in reciprocal space
    '''
    return np.exp(-0.5*np.power(k,2)*np.power(sigma,2))


def w(n):
    '''
    Local grand potential density
    '''
    first_term = 0.5 * SG_c * np.power(n-liquid_coex,2) * np.power(n-vapor_coex,2)
    second_term = 0.25* SG_d * np.power(n-liquid_coex,4) * np.power(n-vapor_coex,4)
    second_term[n>liquid_coex]=0
    second_term[n<vapor_coex]=0
    third_term = - n*delta_mu
    energy = first_term+second_term+third_term
    return energy

def w_prime(n):
    '''
    Derivative of local grand potential density wrt density
    '''
    first_prefactor = SG_c
    first_bracket   = np.power(n-liquid_coex,2) * (n-vapor_coex) \
                    + np.power(n-vapor_coex,2) * (n-liquid_coex)
    second_prefactor = SG_d
    second_bracket   = np.power(n-liquid_coex,4) * np.power(n-vapor_coex,3) \
                    + np.power(n-vapor_coex,4) * np.power(n-liquid_coex,3)
    second_bracket[n>liquid_coex]=0
    second_bracket[n<vapor_coex]=0    
    third_term     = - delta_mu
    return first_prefactor*first_bracket + second_prefactor*second_bracket + third_term


def update_full_dens(rho_slow, rho_guess, ratio):
    '''
    Compute full density by minimisation of the functional
    '''

    # initial guess
    rho_trial = rho_guess
    rho_old   = np.zeros(r_.shape)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.set_ylim(-0.05,3.05)
    ax.set_xlim(-1,31)
    ax.set_ylabel(r'$\rho(z)/\rho_{\mathrm{u}}$')
    ax.set_xlabel(r'$z$ [$\mathrm{\AA}$]')
    place(ax)

    # iterative loop
    while np.allclose(rho_trial,rho_old, rtol, atol) is False:

        ax.plot(r_ , rho_trial/rho_bulk)
        display.display(fig)
        display.clear_output(wait=True)
          
        rho_old = rho_trial

        
        delta_rho_rho_slow_r = (rho_old - rho_slow) * rho_slow 
        delta_rho_rho_slow_k = get_rFT(delta_rho_rho_slow_r, r_)
        
        pre_gamma_k =  delta_rho_rho_slow_k * c_k
        pre_gamma_r =  get_invrFT(pre_gamma_k, k_)
        
        gamma = pre_gamma_r * rho_slow/ np.power(rho_bulk, 2)        
        
        # compute RHS
        rho_new = rho_slow * np.exp(-beta*external_potential + ratio * gamma)
        
        # update to new density
        rho_trial = rho_old * (1-alpha_full) + rho_new * alpha_full
    
    rho_final = rho_trial
    
    return rho_final


def update_slow_dens(rho_full, rho_guess):
    '''
    Compute slow density according to vdW with unbalanced force from given 
    full density, guess from previous iteration
    '''
    
    edge_solute_index = np.argmax(rho_full > rho_guess)
    #edge_solute_index = np.argmax(rho_full > rho_guess)
    
    # Coarse grain the full density
    rho_full_bar_k = get_rFT(rho_full, r_) * Gaussian_kspace(k_, cg_length)
    rho_full_bar_r = get_invrFT(rho_full_bar_k, k_)
    

    # initial guess
    rho_trial = rho_guess
    rho_old = np.zeros(r_.shape)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.set_ylim(-0.05,3.05)
    ax.set_xlim(-1,31)
    ax.set_ylabel(r'$\rho_{\mathrm{s}}(z)/\rho_{\mathrm{u}}$')
    ax.set_xlabel(r'$z$ [$\mathrm{\AA}$]')

    place(ax)  
    
    # iterative loop

    while np.allclose(rho_trial,rho_old, rtol, atol) is False:
        
        ax.plot(r_ , rho_trial/rho_bulk)
        display.display(fig)
        display.clear_output(wait=True)
 
        rho_old = rho_trial


        rho_s_star = rho_old[edge_solute_index]
        SG_m = SG_m_min + (SG_m_max - SG_m_min) * (rho_s_star - SG_rho_s_min) /(SG_rho_s_max - SG_rho_s_min)
        #if SG_m < SG_m_min:
         #   SG_m = SG_m_min
           
        rho_slow_bar_k = get_rFT(rho_old, r_) * Gaussian_kspace(k_, cg_length)
        rho_slow_bar_r = get_invrFT(rho_slow_bar_k, k_)        
        
        first_term = a*np.power(cg_length,2)*rho_full_bar_r/SG_m
        second_term = 0.5 * w_prime(rho_old)*np.power(cg_length,2)/ SG_m
        third_term = (1 -(a*np.power(cg_length,2)/SG_m ))*rho_slow_bar_r
        
        rho_new = first_term - second_term + third_term
        
        
        rho_new[rho_new < vapor_coex] = vapor_coex
        rho_new[rho_new > rho_bulk] = rho_bulk
        rho_new[r_> edge+15] = rho_bulk
        
         # Update to new density
        rho_trial = rho_old * (1 - alpha_slow) + rho_new * alpha_slow
      
    rho_final = rho_trial  
    SG_m_final = SG_m 
    
    # return new slowly varying density
    return rho_final, SG_m_final


def free_energy_large(rho_s, rho_f, SG_m):
    '''
    Returns Free energy from van der Waals functional 
    + unbalanced energy between two densities
    '''
    
    #  van der Waals functional
    local_term  = w(rho_s) - w(rho_s*0+rho_bulk)
    local_term[rho_s > rho_bulk] = 0
    integrand   = r_ * r_ * local_term
    free_energy_local =  4 * np.pi * integrate.simpson(integrand, r_)
    
    gradient_term = 0.5 * SG_m * np.power(np.gradient(rho_s, r_), 2)
    gradient_term[rho_s > liquid_coex] = 0
    gradient_term[rho_s < vapor_coex] = 0
    gradient_term[r_> edge+5] = 0
    gradient_term[r_< edge-5] = 0
    integrand   = r_ * r_ * gradient_term
    free_energy_gradient =  4 * np.pi * integrate.simpson(integrand, r_)
    

    #  unbalanced energy
    delta_rho_bar_k = get_rFT(rho_f - rho_s, r_) * Gaussian_kspace(k_, cg_length)
    delta_rho_bar_r = get_invrFT(delta_rho_bar_k, k_)
    psi = (-2 * a * delta_rho_bar_r)
    psi = psi-psi[-1]
    integrand       = r_ * r_ * psi * rho_s
    integrand[r_ > edge+12] = 0
    free_energy_u   = 4 * np.pi * integrate.simpson(integrand, r_)   
    
   
    return free_energy_local*J_to_kJmol, free_energy_gradient*J_to_kJmol,\
            free_energy_u*J_to_kJmol


def free_energy_small(rho_f, rho_s):
    '''
    Return the energy to insert a solute
    '''
    
    # ideal term
    rho_s[r_ < edge - 5] = 0
    rho_s[rho_s==0] = 1e-100
    ratio           = rho_f/rho_s
    ratio[ratio==0] = 1
    integrand       =  r_ * r_ * (rho_f * np.log(ratio) - rho_f + rho_s)
    free_energy_id  = 4 * np.pi * kB * temperature * integrate.simpson(integrand, r_)
    
    # external term
    integrand       = r_ * r_ * external_potential * rho_f
    free_energy_ext = 4 * np.pi * integrate.simpson(integrand, r_)
    
    # excess term
    
    delta_rho_rho_slow_r = (rho_f - rho_s) * rho_s 
    delta_rho_rho_slow_k = get_rFT(delta_rho_rho_slow_r, r_)
    
    pre_gamma_k          =  delta_rho_rho_slow_k * c_k
    pre_gamma_r          =  get_invrFT(pre_gamma_k, k_)
    
    gamma                = pre_gamma_r * rho_s / np.power(rho_bulk, 2)
    
    integrand            =  r_ * r_ * (rho_f - rho_s)  * gamma
    free_energy_exc      = -0.5 * kB * temperature * 4 * np.pi *\
                            integrate.simpson(integrand, r_)

    #  unbalanced energy
    delta_rho_bar_k = get_rFT(rho_f - rho_s, r_) * Gaussian_kspace(k_, cg_length)
    delta_rho_bar_r = get_invrFT(delta_rho_bar_k, k_)
    psi = (-2 * a * delta_rho_bar_r)
    psi = psi-psi[-1]
    

    rho_bar_k = get_rFT(rho_f, r_) * Gaussian_kspace(k_, cg_length)
    rho_bar_r = get_invrFT(rho_bar_k, k_)    
    
    integrand1       = r_ * r_ * psi * rho_bar_r 
    integrand1[r_ > edge+12] = 0
    integrand2       = r_ * r_ * psi * rho_f
    integrand2[r_ > edge+12] = 0
    
    free_energy_u_rho  = 4 * np.pi * integrate.simpson(integrand2, r_) 
    free_energy_u_rho_bar   = 4 * np.pi * integrate.simpson(integrand1, r_)  
    return free_energy_id*J_to_kJmol, free_energy_ext*J_to_kJmol, \
            free_energy_exc*J_to_kJmol, free_energy_u_rho*J_to_kJmol, \
            free_energy_u_rho_bar*J_to_kJmol   


header_text = '''-------------------------------------------------------------------------------
Output file from LCW-cDFT

Copyright (C) 2023 Anna T. Bui
-------------------------------------------------------------------------------

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

Paper: A classical density functional theory for solvation across length scales
    Authors: Anna T. Bui, Stephen J. Cox
    Available at: https://doi.org/10.1063/5.0223750

-------------------------------------------------------------------------------

DENSITY PROFILES

r [AA]   |  full_density [AA^-3]  |  slow_density [AA^-3] 
'''

def write_out_data(filename, full_dens_final, slow_dens_final, SG_m):
    '''
    Write out data
    '''

    F_local, F_gradient, F_u_large = free_energy_large(slow_dens_final, full_dens_final, SG_m)
    F_id, F_ext, F_exc, F_u_small, F_u_small_bar = free_energy_small(full_dens_final, slow_dens_final)
    
    F_large = F_local + F_gradient 
    F_small = F_id + F_ext + F_exc 
    
    F_solv = F_gradient + F_local + F_id + F_ext + F_exc
    
    per_area = 1e3 /(1e-3 * NA * 4 * np.pi * 1e-20 * np.power(edge,2))
    
    F_local_area = F_local * per_area
    F_gradient_area = F_gradient * per_area
    
    F_solv_area = F_solv * per_area
     
    footer_text = '''

-------------------------------------------------------------------------------
PARAMETERS OF SOLVENTS
-------------------------------------------------------------------------------

bulk_density [AA^-3]                    = {:.8f}
temperature [K]                         = {:.1f}

liquid_coex_density [AA^-3]             = {:.8f}
gas_coex_density [AA^-3]                = {:.8f}
c [J mol^-1 nm^9]                       = {:.5f}                               
d [J mol^-1 nm^15]                      = {:.5f}    
m_min [kJ mol^-2 cm^3 AA^2]             = {:.2f}
m_max [kJ mol^-2 cm^3 AA^2]             = {:.2f}
m [kJ mol^-2 cm^3 AA^2]                 = {:.2f}

a [kJ cm^3 mol^-2]                      = {:.2f}
lambda [AA]                             = {:.3f}

-------------------------------------------------------------------------------
EXTERNAL POTENTIAL
-------------------------------------------------------------------------------

{}

-------------------------------------------------------------------------------
FREE ENERGY OF SOLVATION    
-------------------------------------------------------------------------------

Local van der Waals [kJ/mol]            = {:.6f}
Gradient van der Waals [kJ/mol]         = {:.6f}
Local van der Waals per area [mJ/m^2]   = {:.6f}
Gradient van der Waals per area [mJ/m^2]= {:.6f}
Combined large length scale [kJ/mol]    = {:.6f}

Ideal term [kJ/mol]                     = {:.6f}
External term [kJ/mol]                  = {:.6f}
Excess term [kJ/mol]                    = {:.6f}
Combined small length scale [kJ/mol]    = {:.6f}

Unbalanced rho energy [kJ/mol]          = {:.6f}
Unbalanced rho_bar energy [kJ/mol]      = {:.6f}
Unbalanced rho_s energy [kJ/mol]        = {:.6f}

Solvation free energy [kJ/mol]          = {:.6f}
Solvation free energy per area [mJ/m^2] = {:.6f}

-----------------------------------END OF OUTPUT------------------------------'''.format( \
        rho_bulk, temperature, \
        liquid_coex, vapor_coex, SG_c/c_convert, SG_d/d_convert, \
        SG_m_min/m_convert, SG_m_max/m_convert, SG_m/m_convert, \
        a/a_convert, cg_length, external_potential_text, \
        F_local, F_gradient, F_local_area, F_gradient_area, F_large, \
        F_id, F_ext, F_exc, F_small, \
        F_u_small, F_u_small_bar, F_u_large, \
        F_solv, F_solv_area)
    
    
    np.savetxt(filename, np.c_[r_,full_dens_final,slow_dens_final], \
                fmt='%.10e', header=header_text, footer=footer_text)
    

    

if __name__ == "__main__":
    '''
    Main program for minimising LCW-cDFT 
    Solve for the equilibrium full density and
    slowly varying density self-consistently.
    '''
    
    # start by getting arguments 
    args = get_arguments()
    path_to_input = args.input_file
    path_to_output = args.output_file
    
    # essenial inputs
    rho_bulk, temperature, delta_mu, liquid_coex, vapor_coex, \
    SG_c, SG_d,  SG_m_min, SG_m_max, SG_rho_s_min, SG_rho_s_max, cg_length, a, dcf_file, \
    initial_guess, alpha_full, alpha_slow, rtol, atol, \
    k_cutoff, dr, r_min, r_max, max_FT, solute_type, \
    HS_radius, LJ_sigma, LJ_epsilon, ATT_epsilon, ATT_sigma, ATT_radius \
    = load_input(path_to_input)

    # prepare grid space
    k_, c_k = get_dcf(dcf_file)
    array_len = int(r_max/dr)
    array_len_last = int(max_FT/dr) # for FT
    r_ = np.linspace(r_min,r_max,array_len)
    k_, c_k = k_[:k_cutoff], c_k[:k_cutoff]
    

    # unit conversion
    beta = 1/(kB * temperature)
    delta_mu = delta_mu * kB * temperature
    SG_c = SG_c * c_convert
    SG_m_min = SG_m_min * m_convert
    SG_m_max = SG_m_max * m_convert
    SG_d = SG_d * d_convert
    a = a * a_convert
   
    # external potential
    if solute_type == 'HS':
        external_potential = HS_solute(r_, HS_radius)
        external_potential_text = 'HS solute, R [AA] = {}'.format(HS_radius)
        edge = HS_radius
    elif solute_type == 'LJ':
        external_potential = LJ_solute(r_, LJ_epsilon, LJ_sigma)
        external_potential_text = 'LJ solute, sigma [AA] = {}, \
                                \n epsilon [kcal mol^-1] = {}'.format(LJ_sigma, LJ_epsilon)
        edge = LJ_sigma
    elif solute_type == 'ATT':
        edge = ATT_radius
        external_potential_text = 'ATT solute, radius [AA] = {}, \
                                epsilon [kcal mol^-1] = {}, \
                                sigma [AA] = {}'.format( \
                                ATT_radius, ATT_epsilon, ATT_sigma)
        external_potential = ATT_solute(r_, ATT_epsilon, ATT_sigma, ATT_radius)
    
    # FIRST ITERATION INITIALISATION
    if initial_guess != 'bulk':
        distance = float(initial_guess)
        d = 0.5* np.sqrt(SG_m_min/SG_c) / (liquid_coex-vapor_coex)
        rho_guess = 0.5*((rho_bulk + vapor_coex)+(rho_bulk - vapor_coex) \
                         *np.tanh((r_-edge+distance)/(2.5*d)))
    else:
        rho_guess = rho_bulk
    
    
    # start with an initial guess for both the full and slow density
    full_dens_guess = rho_guess * np.exp(-beta*external_potential)
    slow_dens_guess = np.ones(r_.shape) * rho_guess
    
    # The first cycle
    full_dens_new   = update_full_dens(slow_dens_guess, full_dens_guess, 0.5)
    slow_dens_guess_k = get_rFT(full_dens_new, r_) * Gaussian_kspace(k_, 1.5)
    slow_dens_guess_r = rho_guess #get_invrFT(slow_dens_guess_k, k_)
        
    slow_dens_new, SG_m_new   = update_slow_dens(full_dens_new, slow_dens_guess_r)
    slow_dens_trial = slow_dens_new
    slow_dens_old   = 0*slow_dens_trial
    
    # Iterative loop: update both full and slow density if not converged
    while np.allclose(slow_dens_trial, slow_dens_old,  rtol, atol) is False:
        
        slow_dens_old = slow_dens_trial
        
        # Iterative loop: update full density if not converged
        full_dens_new = update_full_dens(slow_dens_old, full_dens_new, 1)
        # Iterative loop: update slow density if not converged
        slow_dens_new, SG_m_new = update_slow_dens(full_dens_new, slow_dens_new)
        slow_dens_trial = slow_dens_new
    
    # final converged densities
    slow_dens_final = slow_dens_new
    SG_m_final = SG_m_new
    full_dens_final = full_dens_new
    
    # write output
    write_out_data(path_to_output, full_dens_final, slow_dens_final, SG_m_final)
    

    

    
