import numpy as np
import scipy.special as sci
from multiprocessing import Pool, cpu_count
import emcee
import h5py
from swiftsimio import Writer
import unyt
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d
from tqdm import tqdm

### Model parameters
G = 4.299581e+04       # kpc / 10^10 solar mass * (km/s)^2
q = 11.4               # Dispersion parameter
v0 = 200.              # Circular velocity
r0 = 20*G/v0**2      # Scale Radius

### Sampling parameters
ndim = 3               # r, vr, vt
nwalkers = 32          # Number of MCMC walkers FIXME: for parallel maybe reduce?
nsamples = 300000        # Number of samples to be drawn for each walker
scatter_coeff = 0.1    # Amplitude of random scatter for walker initial pos
burnin = 100           # Number of burn-in steps
N = nwalkers*nsamples  # Number of active particles in simulation
display_results = True # Plot the sampling results as a check

### Cut functions (Zang 1976, Toomre 1989, see De Rijcke et al. 2019)
r_inner = G/v0**2
r_outer = 11.5*r_inner
nu       = 4
mu       = 5
R_cut = 30.0*r_inner   # Hard cut (set to much larger than active disk)
chi     = 0.5            # Global Mass coefficient (e.g. 0.5 for "Half-Mass" Disk) --> Add External potential with r0' = 1/chi * r0

## Missing Mass sampling
nwalkers_mm = 32
nsamples_mm = 30000
N_mm = nwalkers_mm * nsamples_mm # Number of unresponsive background particles to compensate missing mass
active_id_start = 10000000 # Particle ID below which the particles are unresponsive (see https://swift.dur.ac.uk/docs/GettingStarted/special_modes.html)

### Parallelism
use_parallel = False   # Use parallel sampling
nb_threads = 6         # Number of parallel threads to use

### IC File
fname = 'mestel.hdf5'  # Name of the ic file (dont forget .hdf5)
pickle_ics = 0         # Optional: Pickle ics (pos,vel,mass) for future use
box_size = 2000.0      # Size of simulation box
periodic_bdry = False  # Use periodic boundary conditions or not

# Compute other quantities from parameters
Sigma0 = v0**2 / (2.0*np.pi*G*r0) # Scale Surface density 
sig = v0 / np.sqrt(1.0+q) # Velocity dispersion
sig2 = v0**2 / (1.0+q)
F0 = Sigma0 / ( 2.0**(q/2.) * np.sqrt(np.pi) * r0**q * sig**(q+2.0) * sci.gamma((1+q)/2.0)) # DF Normalization factor
# Precompute quantities for efficiency during sampling
lnF0 = np.log(F0)
routv0mu = (r_outer * v0)**mu
lnroutv0mu = np.log(routv0mu)
rinv0nu = (r_inner * v0)**nu

### Print parameters
print('\n############################################################################################# \n')
print('Generating Mestel Disk with the following parameters:\n')
print('-- Model Parameters --')
print('Circular velocity:                                v0 = ' + str(v0) + 'km/s')
print('Characteristic Radius:                            r0 = ' + str(r0) + ' kpc')
print('Dispersion parameter:                              q = ' + str(q))
print('')
print('-- Truncation --')
print('Inner cutoff radius:                         r_inner = '+str(r_inner)+' kpc, Exponent: nu = '+str(nu))
print('Outer cutoff radius:                         r_outer = '+str(r_outer)+' kpc, Exponent: mu = '+str(mu))
print('Hard Cut:                                      R_cut = '+str(R_cut))
print('')
print('-- Sampling --')
print('Number of active particles:                   N_part = {:.1e}'.format(N))
print('Number of background particles (missing mass): N_bkg = {:.1e}'.format(N_mm))
print('')
print('-- Output --')
print('Output file: ' + fname)
print('\n############################################################################################# \n')

### Helper Functions
# Binding potential / Binding energy
def relative_potential(r):
    return -v0**2 * np.log(r/r0)
def relative_energy(r,vr,vt):
    return relative_potential(r) - 0.5*(vr**2 + vt**2)

### Log of Distribution Function (De Rijcke et al. 2019, Sellwood 2012)
def log_prob(x):
    if (x[0] <= 0.0 or x[0] > R_cut ): return -np.inf
    elif x[2] < 0.0: return -np.inf
    else:
        rvt = x[0]*x[2]
        rvtn = rvt**nu
        # FIXME: divide DF by r to obtain correct density, why?
        # Truncated Disk
        return np.log(rvtn) - np.log(rvtn + rinv0nu) + lnF0 + q*np.log(rvt) + relative_energy(x[0],x[1],x[2])/sig2 + lnroutv0mu - np.log(routv0mu + rvt**mu) + np.log(x[0])
        # Untruncated Disk
        #return lnF0 + q*np.log(rvt) + relative_energy(x[0],x[1],x[2])/sig2 + np.log(x[0])
    

### Initialize Walkers
startpos = np.array([r0,0.0,v0])
p0 = startpos + scatter_coeff * np.random.randn(nwalkers, len(startpos))

### Sample DF
print('Sampling  ' + "{:d}".format(N) + ' active particles...')
if use_parallel:
    with Pool(processes=nb_threads) as pool: 
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        state = sampler.run_mcmc(p0, burnin)
        sampler.reset()
        sampler.run_mcmc(state, nsamples,progress=True)
    samples = sampler.get_chain(flat=True)
    
else:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    state = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    sampler.run_mcmc(state, nsamples,progress=True)
    samples = sampler.get_chain(flat=True)
     
### Convert samples to ICs
# Positions
theta_rand = np.random.uniform(0.0,2*np.pi,N)
x = samples[:,0]*np.cos(theta_rand)
y = samples[:,0]*np.sin(theta_rand)
z = np.zeros(N)

X = np.array([x,y,z]).transpose()

# Velocities
v_x = np.cos(theta_rand) * samples[:,1] - np.sin(theta_rand) * samples[:,2]
v_y = np.sin(theta_rand) * samples[:,1] + np.cos(theta_rand) * samples[:,2]
v_z = np.zeros(N)

V = np.array([v_x,v_y,v_z]).transpose()

'''
### Compute Missing mass
The distribution function that was sampled so far contains the 2 truncation functions, and hence
does not generate a surface density equal to the Mestel density. To compensate for this, the
"missing mass density" is calculated numerically and sampled to produce another set of particles
that will be static (or "unresponsive") in the simulation and act merely as a kind of background
potential.

The mass density of the truncated model (which is not analytical) also must be known to calculate
the total mass and hence the masses of the (active) particles, this is done below.
'''
print('Computing mass density of truncated disk, missing mass density and integrate to get total mass...')
# Truncated Distribution Function (same as above, without log)
def DF(vr,vt,r):
    L = r*vt
    return F0 * L**(nu+q) * routv0mu / ((rinv0nu+L**nu)*(routv0mu+L**mu)) * np.exp(relative_energy(r,vr,vt)/sig2)

# Truncated DF marginalized (integrated) over vr (analytically), for efficiency
def DF_intover_vr(vt,r):
    L = r*vt
    return F0 * routv0mu * r**(nu-1) * vt**(nu+q) * np.exp(-0.5*vt**2/sig2) / ((rinv0nu + L**nu)*(routv0mu + L**mu)) * np.sqrt(2*np.pi)*sig / r0**(-1-q)

# Numerically integrate over vr to produce Sigma(r)
def density_truncated(r):
    return quad(DF_intover_vr,0.0,np.inf,args=(r,))[0]

# Numerically integrate over vr and r to produce M(r)
def integrand(vt,r):
    return 2*np.pi*r*DF_intover_vr(vt,r)
def m_of_r_truncated(r):
    return dblquad(integrand,0.0,r,0.0,np.inf)[0]

# Mestel density
def density_mestel(r):
    return v0**2/(2.0*np.pi*G*r)

# Total Mass at R_cut --> mass of particles
# Theoretical total mass of Mestel model
Mtot = v0**2 * R_cut / G

# Active Disk
M = m_of_r_truncated(R_cut)
m = M / N

# Missing Mass
M_mm = Mtot - M
m_mm = M_mm / N_mm

print('Total Mass of untruncated disk: ' + "{:.2e}".format(Mtot*1e10) + ' solar masses.')
print('Total Mass of truncated disk:   ' + "{:.2e}".format(M*1e10) + ' solar masses.')
print('Missing Mass:                   ' + "{:.2e}".format(M_mm*1e10) + ' solar masses.')

# Create more efficient truncated density via interpolation
nb_radii_interp = 500
interp_rspace = np.linspace(0.0,R_cut,nb_radii_interp)
interp_dens = np.empty(nb_radii_interp)
for i,r in enumerate(interp_rspace):
    interp_dens[i] = density_truncated(r)
density_truncated_interp = interp1d(interp_rspace,interp_dens)

# Missing mass density
def missing_mass_density(r):
    return density_mestel(r) - density_truncated_interp(r)

# Sample missing mass
def logprob_mm(r):
    if r <= 0.0 or r > R_cut : return -np.inf
    else: return np.log(missing_mass_density(r)) + np.log(r)
print('Sampling ' + "{:d}".format(N_mm) + ' background particles to compensate for missing mass...')
sampler_mm = emcee.EnsembleSampler(nwalkers_mm, 1, logprob_mm)
startpos_mm = np.array(r0)
p0_mm = startpos_mm + scatter_coeff * np.random.randn(nwalkers_mm, 1)
state_mm = sampler_mm.run_mcmc(p0_mm, burnin)
sampler_mm.reset()
sampler_mm.run_mcmc(state_mm, nsamples_mm,progress=True)
samples_mm = sampler_mm.get_chain(flat=True)

# Convert to cartesian
r_mm = samples_mm[:,0]
theta_rand_mm = np.random.uniform(0.0,2*np.pi,N_mm)
x_mm = r_mm*np.cos(theta_rand_mm)
y_mm = r_mm*np.sin(theta_rand_mm)
z_mm = np.zeros(N_mm)

X_mm = np.array([x_mm,y_mm,z_mm]).transpose()

# Global mass coefficient
m *= chi
m_mm *= chi

# Merge Active & Passive particles to write IC file
X_full = np.concatenate((X,X_mm))
V_full = np.concatenate((V,np.zeros((N_mm,3))))
M_full = np.concatenate((m*np.ones(N),m_mm*np.ones(N_mm)))
IDs    = np.concatenate((np.arange(active_id_start,N+active_id_start),np.arange(N_mm)))
print(IDs)

### Write to hdf5
print('Writing IC file...')
galactic_units = unyt.UnitSystem(
    "galactic",
    unyt.kpc,
    unyt.unyt_quantity(1e10, units=unyt.Solar_Mass),
    unyt.unyt_quantity(1.0, units=unyt.s * unyt.kpc / unyt.km).to(unyt.Gyr),
)
wrt = Writer(galactic_units, box_size * unyt.kpc)
wrt.dark_matter.coordinates = X_full * unyt.kpc
wrt.dark_matter.velocities = V_full * (unyt.km / unyt.s)
wrt.dark_matter.masses = M_full * 1e10 * unyt.msun
wrt.dark_matter.particle_ids =IDs
wrt.write(fname)
print('done.')
epsilon0 = np.sqrt(2.0*G*Mtot*r0)/v0 / np.sqrt(N+N_mm)
print('Recommended Softening length (times conventional factor): ' + "{:.4e}".format(epsilon0) + ' kpc')

### Testing
if display_results:
    print('Plotting Sampling Results...')
    import matplotlib.pyplot as plt
    def mass_ins(r,rs,ms):
        return ms*(rs <= r).sum()
    def density(r,mins):
        return np.diff(mins) / np.diff(r) / (2.*np.pi*r[1:])
    def density_mestel(r):
        return v0**2/(2.*np.pi*G*r)

    rsp = np.logspace(np.log10(r_inner/4),np.log10(R_cut-5),300)
    mins_active = np.empty(len(rsp))
    mins_passive = np.empty(len(rsp))
    for i,r in enumerate(rsp):
        mins_active[i] = mass_ins(r,samples[:,0],m)
    for i,r in enumerate(rsp):
        mins_passive[i] = mass_ins(r,samples_mm[:,0],m_mm)
    dens_active = density(rsp,mins_active)
    dens_passive = density(rsp,mins_passive)
    
    fs = 20
    ms = 2
    lw = 2.5
    plt.figure(figsize=(10,10))
    plt.loglog(rsp[1:],dens_active,'s',ms=ms,label='Sampled Mass Density (truncated)')
    plt.loglog(rsp[1:],dens_passive,'^',ms=ms,label='Sampled Missing Mass Density')
    plt.loglog(rsp[1:],dens_active + dens_passive,'o',ms=2.5,label='Total Sampled Mass Density')
    plt.loglog(rsp,chi*density_truncated_interp(rsp),'--',lw=lw,label='Theoretical Density (truncated)')
    plt.loglog(rsp,chi*density_mestel(rsp),lw=lw,c='black',label='Theoretical Total density (untruncated)')
    plt.loglog(rsp,chi*missing_mass_density(rsp),'-.',label='Theoretical Missing Mass Density')
    plt.legend(fontsize=fs-7)
    plt.xlabel(r'$r$ [kpc]',fontsize=fs)
    plt.ylabel(r'$\rho(r)$ [$10^{10} M_{\odot}$ / kpc$^3$]',fontsize=fs)
    plt.title('Consistency Check',fontsize=fs)
    plt.show()
