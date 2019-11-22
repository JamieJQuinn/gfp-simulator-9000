"""Provides physical and nondimensionalised parameters"""

import math

# Parameters
cell_radius = 5e-6 # metres
membrane_thickness_real = 0.5e-6 # From approximate wavelength of light (metres)
D = 30e-12 # Diffusion constant of GFP (m^2 per s)
kp_real = 5.6e5 # association constant (M^-1 s^-1)
km_real = 0.069 # dissociation constant (s^-1)
total_gfp = 1e6
total_rac = 1e7

N0 = 6.0221409e23 # Avogadro's (mole^-1)

# Volumes
cell_volume = 0.5*4.0/3.0*math.pi*math.pow(cell_radius,3) * 1e3 # litres
membrane_area = 0.5*4.0*math.pi*math.pow(cell_radius, 2) + math.pi*math.pow(cell_radius, 2)
# membrane_volume = 0.5*4.0/3.0*math.pi*(math.pow(cell_radius, 3) - math.pow(cell_radius-membrane_thickness, 3)) * 1e3 #litres

# Concentrations
conc_gfp_real = total_gfp/(N0*cell_volume) # Molars
conc_rac_real = total_rac/(N0*membrane_area) # Mole per m^2

# Nondimensionalise
M0 = conc_gfp_real # typical concentration (Molars)
L0 = 1e-5 # typical length (metres)
T0 = L0*L0/D # Diffusion time (seconds)

membrane_thickness = membrane_thickness_real/L0

conc_gfp = conc_gfp_real / M0
conc_rac = conc_rac_real / (1e3*M0*L0)

kp = kp_real * M0 * T0
km = km_real * T0
