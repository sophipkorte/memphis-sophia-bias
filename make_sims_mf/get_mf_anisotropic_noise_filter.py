import gc
import os
import numpy as np
import healpy as hp
from orphics import maps, stats, mpi, io
import falafel.utils as futils
from falafel import qe
from lenspyx.utils import camb_clfile
import pickle

import mpi4py.rc
mpi4py.rc.threads = False
import MPI_error_handler
import argparse

args = argparse.Namespace()

#############################################################
# PARAMETERS
#############################################################
args.lmax_len = 3000
args.nside = 2048
args.nsims = 200
lmin = 600

args.mask_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits'
args.obs_file = '/home/spk24f/masks/LAT_hits_2048_galcoords.fits'
args.hits_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits'
args.output_dir = '/data/data01/sophia/meanfield-maps/lensing_anoise/'
os.makedirs(args.output_dir, exist_ok=True)
output_path = lambda x: os.path.join(args.output_dir, x)

amp_arcmin = 5.3
args.nobs = hp.read_map(args.obs_file)

#############################################################
# LOAD MASK AND CLS
#############################################################
mask = hp.read_map(args.mask_file)
fsky = np.mean(mask**2)

cls_path = "/home/spk24f/gitreps/lenspyx/lenspyx/data/cls/"
cl_len = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cl_gradlens = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_gradlensedCls.dat'))

cl_gradlens_order = {
    "TT": cl_gradlens["tt"][:args.lmax_len+1],
    "TE": cl_gradlens["te"][:args.lmax_len+1],
    "EE": cl_gradlens["ee"][:args.lmax_len+1],
    "BB": cl_gradlens["bb"][:args.lmax_len+1]
}

cl_len_order = {
    "TT": cl_len["tt"][:args.lmax_len+1],
    "TE": cl_len["te"][:args.lmax_len+1],
    "EE": cl_len["ee"][:args.lmax_len+1],
    "BB": cl_len["bb"][:args.lmax_len+1]
}

#############################################################
# ANISOTROPIC NOISE FUNCTION
#############################################################
def add_anisotropic_noise(noise_scale, map_size):
    noiset = np.random.normal(scale=noise_scale, size=map_size)
    return noiset

#############################################################
# HELPER FUNCTIONS
#############################################################
def smooth_cls(cl, points=300):
    bin_edges = np.linspace(2, len(cl), points).astype(int)
    cents, cls = bandedcls(cl, bin_edges)
    cls = maps.interp(cents, cls, fill_value="extrapolate")(np.arange(len(cl)))
    return cls

def bandedcls(cl, _bin_edges):
    ls = np.arange(cl.size)
    binner = stats.bin1D(_bin_edges)
    cents, bls = binner.bin(ls, cl)
    return cents, bls

#############################################################
# PRECOMPUTE NOISE SCALE (once, outside loop)
#############################################################
npix = 12 * args.nside**2
Omegapix = 4 * np.pi / npix
sigma2_muK2_sr = (amp_arcmin * np.pi / 180 / 60)**2
sigma2_pixel = sigma2_muK2_sr / Omegapix
nobs = args.nobs
hits_safe = np.copy(nobs)
hits_safe[hits_safe <= 0] = np.nan
noise_scale = np.sqrt(sigma2_pixel / hits_safe)
del nobs, hits_safe  # no longer needed

##############################################################
# LOAD FILTERED CLs
##############################################################
with open('/home/spk24f/gitreps/lenspyx/examples/partsky_anisotropic_noise_nofg_noNILC/tcl_filter_anoise.pkl', 'rb') as f:  
    tcl = pickle.load(f)

cl_tt_masked_smooth = tcl["TT"][:args.lmax_len+1]

#############################################################
# MPI SETUP
#############################################################
comm, rank, my_tasks = mpi.distribute(args.nsims)
s = stats.Stats(comm)

px = qe.pixelization(nside=args.nside)

for sim in my_tasks:
    print(f"Rank {rank} processing sim {sim}")

    # 1. Generate Gaussian CMB from lensed Cls
    np.random.seed(sim * 3)
    gauss_map = hp.synfast(cl_len_order['TT'], nside=args.nside, lmax=args.lmax_len)

    # 2. Add independent anisotropic noise (in-place)
    np.random.seed(sim * 3 + 1)
    noiset = np.random.normal(scale=noise_scale, size=npix)
    gauss_map += noiset
    del noiset

    # 3. Apply mask (in-place)
    gauss_map *= mask
    np.nan_to_num(gauss_map, copy=False)

    # 4. Get alms
    tlm_masked = hp.map2alm(gauss_map, lmax=args.lmax_len)
    del gauss_map

    # 5. Same filtering as data
    #alreay loaded 

    # 6. Filter (same procedure as data reconstruction)
    Xdat0 = qe.filter_alms(tlm_masked,
                           cl_gradlens_order["TT"] / cl_tt_masked_smooth[:args.lmax_len+1],
                           lmin, args.lmax_len)
    Xdat = qe.filter_alms(tlm_masked,
                          1. / cl_tt_masked_smooth[:args.lmax_len+1],
                          lmin, args.lmax_len)
    del tlm_masked

    # 7. Run QE
    recon = qe.qe_temperature_only(px=px, Xalm=Xdat0, Yalm=Xdat, mlmax=args.lmax_len)
    del Xdat0, Xdat

    # 8. Stack
    unnorm_rkalm_g = futils.change_alm_lmax(recon.astype(np.complex128), args.lmax_len)
    del recon

    s.add_to_stack('rf', unnorm_rkalm_g.real)
    s.add_to_stack('if', unnorm_rkalm_g.imag)
    del unnorm_rkalm_g
    gc.collect()

with io.nostdout():
    s.get_stacks()

if rank == 0:
    mf_alm_unnorm = s.stacks['rf'] + 1j * s.stacks['if']
    hp.write_alm("mf_phi_alm_2048_200sims_unnorm_anoise_filter.fits",mf_alm_unnorm,overwrite=True)
    print("Mean field saved.")
 


















































