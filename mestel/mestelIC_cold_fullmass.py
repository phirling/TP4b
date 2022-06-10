import numpy as np
import scipy.special as sci
from multiprocessing import Pool, cpu_count
import emcee
import h5py
import write_gadget as wg
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d
from tqdm import tqdm

# This script generates a realization with q = inf (sigma_r = 0), corresponding
# to purely circular orbits.
# Sampling is performed without MCMC, via inverse transform sampling for the radii
# the velocities are all equal to v0 and purely tangential.
# (This disk is completely unstable if self gravity is enabled)

### Model parameters
G = 4.299581e+04       # kpc / 10^10 solar mass * (km/s)^2
r0 = 16.               # Scale Radius
v0 = 200 #200               # Circular velocity

### Sampling parameters
N = 40000000 #500 Use 500 for orbits, 1e7 for density # Number of active particles in simulation
N = int(N)

### Cut functions (Zang 1976, Toomre 1989, see De Rijcke et al. 2019)
#r_inner = 5.0 #10.0
#R_cut = 40.0*r_inner   # Hard cut (set to much larger than active disk)
r_inner = G/v0**2
r_outer = 11.5*r_inner
R_cut = 30.0*r_inner

### IC File
fname = 'mestel.hdf5'  # Name of the ic file (dont forget .hdf5)
box_size = 2000.0      # Size of simulation box
periodic_bdry = False  # Use periodic boundary conditions or not

# Compute other quantities from parameters
Sigma0 = v0**2 / (2.0*np.pi*G*r0) # Scale Surface density 

# Disk Patch
# Angles between which to sample
theta_min = -np.pi #-np.pi / 6#-2*np.pi/3
theta_max = np.pi  #np.pi / 6#+2*np.pi/3

# Radii between which to sample
r_min = 0  #4 for orbits
r_max = 30 #12 for orbits

### Print parameters
print('\n############################################################################################# \n')
print('Generating Cold Full-mass Mestel Disk with the following parameters:\n')
print('-- Model Parameters --')
print('Circular velocity:                                v0 = ' + str(v0) + 'km/s')
print('Characteristic Radius:                            r0 = ' + str(r0) + ' kpc')
print('')
print('-- Truncation --')
print('Hard Cut Radius:                               R_cut = '+str(R_cut))
print('')
print('-- Sampling --')
print('Number of active particles:                   N_part = {:.1e}'.format(N))
print('')
print('-- Output --')
print('Output file: ' + fname)
print('\n############################################################################################# \n')

# Inverse transform sampling
def m_of_r(r):
    return v0**2*r/G

def r_of_m(m):
    return G*m/v0**2

# Sample Radii
#m_cut = m_of_r(R_cut)
#m_rand = m_cut * np.random.uniform(size=N)
m_rand = np.random.uniform(m_of_r(r_min),m_of_r(r_max),size=N)
r_rand = r_of_m(m_rand)

### Convert samples to ICs
# Positions
#theta_rand = np.random.uniform(0.0,2*np.pi,N)
theta_rand = np.random.uniform(theta_min,theta_max,N)
x = r_rand*np.cos(theta_rand)
y = r_rand*np.sin(theta_rand)
z = np.zeros(N)

X = np.array([x,y,z]).transpose()

# Velocities
v_x = -np.sin(theta_rand) * v0
v_y = np.cos(theta_rand) * v0
v_z = np.zeros(N)

V = np.array([v_x,v_y,v_z]).transpose()

# Masses
#m = m_cut / N
m = 1.0

# Write IC file
X_full = X
V_full = V
M_full = m*np.ones(N)
IDs    = np.arange(N)

print(IDs)
print('Writing IC file...')
with h5py.File(fname,'w') as f:
    wg.write_header(
        f,
        boxsize=box_size,
        flag_entropy = 0,
        np_total = [0,N,0,0,0,0],
        np_total_hw = [0,0,0,0,0,0]
    )
    wg.write_runtime_pars(
        f,
        periodic_boundary = periodic_bdry
    )
    wg.write_units(
        f,
        length = 3.086e21,
        mass = 1.988e43,
        time = 3.086e21 / 1.0e5,
        current = 1.0,
        temperature = 1.0
    )

    wg.write_block(
        f,
        part_type = 1,
        pos = X_full,
        vel = V_full,
        mass = M_full,
        ids = IDs,
        int_energy = np.zeros(N),
        smoothing = np.ones(N)
    )
print('done.')
