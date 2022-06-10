import numpy as np
from matplotlib.colors import Normalize, LogNorm, PowerNorm
from matplotlib.animation import FuncAnimation
import argparse
import h5py

parser = argparse.ArgumentParser(description="Make film of surface density of snapshots, in order")

# Snapshot Parameters
parser.add_argument("files",nargs='+',help="snapshot files to be imaged")
parser.add_argument("-shift", type=float, default=1000.0, help="Shift applied to particles in params.yml")
parser.add_argument("-aid",type=int,default=10000000,help="Index below which not to plot particles")

# Histogram Parameters
parser.add_argument("-nbins",type=int,default=500,help="Number of bins in each dimension (x,y)")
parser.add_argument("-lim",type=float,default=14,help="Limits of histogram (in kpc)")
parser.add_argument("-interp",type=str,default='none',help="Interpolation used ('none','kaiser','gaussian',...). Default: none")
parser.add_argument("--polar",action='store_true',help="Pot a r-phi histogram rather than an image")

# Color Normalization Parameters
parser.add_argument("-cmap",type=str,default='YlGnBu_r',help="Colormap (try YlGnBu_r, Magma !)")
parser.add_argument("-norm",type=str,default='log',help='Color Norm to use')
parser.add_argument("-gamma",type=float,help="Exponent of power law norm (use -norm power)")
parser.add_argument("-cmin",type=float,default=0,help=
                    """Minimum Physical value in the Histogram.
                    This effectively sets the contrast of the images.""")
parser.add_argument("-cmax",type=float,default=-1,help="Maximum physical value in the Histogram.")

# Figure Parameterrs
parser.add_argument("--notex",action='store_true')
parser.add_argument("-nrows",type=int,default=2,help="Number of rows in figure")
parser.add_argument("-ncols",type=int,default=3,help="Number of columns in figure")
parser.add_argument("-figheight",type=float,default=13,help="Height of figure in inches")
parser.add_argument("-figwidth",type=float,default=10,help="Width of figure in inches")
parser.add_argument("--savefig",action='store_true',help="Save film as .mp4")

# Corotation Parameters
parser.add_argument("-rcr",type=float,default=0.0,
    help="Radius of the corotating origin, set to 0 for no corotation (default)")
parser.add_argument("-acr",type=float,default=0,
    help="Initial angle of the corotating origin")
parser.add_argument("-vcr",type=float,default=200,
    help="Circular velocity of the corotation")
parser.add_argument("--set_origin",action='store_true',
    help="Set the origin of the figure to the corotating origin (default: no)")

args = parser.parse_args()
fnames = args.files

# Convenience
lim = args.lim
active_id_start = int(args.aid)

# Configure histogram color norm
if args.cmin <= 0 and args.norm == 'log': args.cmin = 1.
if args.norm == 'power':
    norm = PowerNorm(gamma=args.gamma,vmin=args.cmin,vmax=args.cmax)
elif args.norm == 'log':
    # The color range is later automatically adjusted to >0 if log norm
    norm = LogNorm(clip=True,vmin=args.cmin,vmax=args.cmax)
elif args.norm == 'linear':
    norm = Normalize(vmin=args.cmin,vmax=args.cmax) # Equivalent to linear
else:
    raise NameError("Unknown color norm: " + str(args.norm))

# Configure Pyplot
from matplotlib import pyplot as plt
if not args.notex: plt.rcParams.update({"text.usetex": True,'font.size':22,'font.family': 'serif'})
else: plt.rcParams.update({'font.size':15})

# Set up corotation
corot = args.rcr > 0
if corot:
    omega = -args.vcr / args.rcr
    x0, y0 = 0.0,0.0
    if args.set_origin:
        x0 = args.rcr*np.cos(args.acr)
        y0 = args.rcr*np.sin(args.acr)
    def process_pos(X,Y,t):
        x = np.cos(omega*t)*X - np.sin(omega*t)*Y - x0
        y = np.sin(omega*t)*X + np.cos(omega*t)*Y - y0
        return x,y


# Set up data treatment for cartesian/polar
if not args.polar:
    def hist(x,y):
        data = np.histogram2d(x,y,bins=args.nbins,range=[[-lim, lim], [-lim, lim]])[0]
        data[data==0] = args.cmin
        return data
else:
    def hist(x,y):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)
        #data = np.histogram2d(phi,r,bins=args.nbins,range=[[0,lim],[-np.pi,np.pi]])[0]
        data = np.histogram2d(phi,r,bins=args.nbins,range=[[-np.pi,np.pi],[0,lim]])[0]
        data[data==0] = args.cmin
        return data

# Methods for axes appearance
fs = 25
def style(ax):
    if not args.polar:
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        ax.set_xlabel('x [kpc]',fontsize=fs)
        ax.set_ylabel('y [kpc]',fontsize=fs)
        ax.set_aspect('equal')
    else:
        ax.set_ylabel('$r$ [kpc]',fontsize=fs)
        ax.set_xlabel('$\phi$ [rad]',fontsize=fs)
        #ax.set_aspect(lim/(2*np.pi))
        ax.set_aspect((2*np.pi)/lim)
        ax.invert_xaxis()

cmax = 0. # The max of the color range is computed across all snapshots
fig, ax = plt.subplots(args.nrows,args.ncols,figsize=(args.figwidth,args.figheight))
ax = ax.flatten()
ims = []
# Initialize plot
if not args.polar:
    ext = (-lim,lim,-lim,lim)
else:
    #ext = (0,lim,-np.pi,np.pi)
    ext = (-np.pi,np.pi,0,lim)

for i,fname in enumerate(fnames):
    # Extract data from snapshot
    f = h5py.File(fname, "r")
    pos = np.array(f["DMParticles"]["Coordinates"]) - args.shift
    IDs = np.array(f["DMParticles"]["ParticleIDs"])
    pos = pos[IDs>=active_id_start]
    t = f["Header"].attrs["Time"][0]
    t_myr = t * f["Units"].attrs["Unit time in cgs (U_t)"][0]/ 31557600.0e6
    x = pos[:,0]
    y = pos[:,1]

    # Process Data
    if corot:
        x,y = process_pos(x,y,t)

    # Update image
    data = hist(x,y)
    
    ims.append(ax[i].imshow(data.T,
              interpolation = args.interp, norm = norm,
              extent = ext, cmap = args.cmap,origin='lower'))
    # Make image
    style(ax[i])

    # Find global max
    cmax = max(cmax,np.amax(data)) 
    ax[i].set_title(r'$t=$ ' + "{:.2f}".format(t_myr) + ' Myr',fontsize=args.figwidth)

norm.vmax = cmax
fig.tight_layout()
# Show & Save figure
if args.savefig:
    fig.savefig('image.eps',bbox_inches = 'tight')
else:
    plt.show()
