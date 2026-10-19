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
args.mask_gal_file = '/home/spk24f/masks/mask_binary_gal_cut_n2048_3deg_apo.fits'
planck_gal_060_file = '/home/spk24f/masks/Planck/Planck_GAL_060_n2048_3deg_apo.fits'
args.obs_file = '/home/spk24f/masks/LAT_hits_2048_galcoords.fits'
args.hits_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits'
args.output_dir = '/data/data01/sophia/meanfield-maps/lensing_anoise_beam/'
os.makedirs(args.output_dir, exist_ok=True)
output_path = lambda x: os.path.join(args.output_dir, x)

def parse_args(existing_args):
    parser = argparse.ArgumentParser(description="Apply extra mask for galactic foregrounds")
    parser.add_argument("--extra_mask",action="store_true",help='mask galactic plane')
    parser.add_argument("--fg_type",type=str,default='d10',help='foreground type')
    parser.add_argument("--freq",type=int,default=93,help='frequency in GHz')
    parser.add_argument("--planck_mask",action="store_true",help='apply Planck mask GAL060')
    return parser.parse_args(namespace=existing_args)

args = parse_args(args)

amp_arcmin = 5.3
beam_fwhm = 2.2  # arcmin
args.nobs = hp.read_map(args.obs_file)

#############################################################
# LOAD MASK AND CLS
#############################################################
mask = hp.read_map(args.mask_file)
mask_gal = hp.read_map(args.mask_gal_file)
mask_planck = hp.read_map(planck_gal_060_file)

if args.planck_mask:
    mask2 = mask * mask_planck
    suffix = '_GAL060'
else:
    mask2 = mask * mask_gal
    suffix = ""

if args.extra_mask == True:
    npix = mask2.size
    pmap = 4*np.pi / npix
    w2 = np.sum((mask2**2)*pmap) /np.pi / 4.
    w4 = np.sum((mask2**4)*pmap) /np.pi / 4.
    w3 = np.sum((mask2**3)*pmap) /np.pi / 4.
    use_mask = mask2
    end = '_extra_mask'

else: 
    npix = mask.size
    pmap = 4*np.pi / npix
    w2 = np.sum((mask**2)*pmap) /np.pi / 4.
    w4 = np.sum((mask**4)*pmap) /np.pi / 4.
    w3 = np.sum((mask**3)*pmap) /np.pi / 4.
    use_mask = mask
    end = '_lat_mask'

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
#PRECOMPUTE BEAM FUNCTION (once, outside loop)
beam = maps.gauss_beam(beam_fwhm, np.arange(args.lmax_len+1))

##############################################################
# LOAD FILTERED CLs
##############################################################
with open(f'/home/spk24f/gitreps/lenspyx/examples/partsky_lensing_anoise_fg_noNILC/{args.fg_type}/f{args.freq}/tcl_filter_anoise_beam_{args.fg_type}_{args.freq}{end}{suffix}.pkl', 'rb') as f:  
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
    gauss_map_alm = hp.synalm(cl_len_order['TT'], lmax=args.lmax_len)
    gauss_map_beam = hp.almxfl(gauss_map_alm, beam)
    gauss_map = hp.alm2map(gauss_map_beam, nside=args.nside, lmax=args.lmax_len)

# 2. Add independent anisotropic noise (in-place)
    np.random.seed(sim * 3 + 1)
    noiset = np.random.normal(scale=noise_scale, size=npix)
    gauss_map += noiset
    del noiset

    # 3. Apply mask (in-place)
    gauss_map *= use_mask
    np.nan_to_num(gauss_map, copy=False)

    # 4. Get alms
    tlm_masked_beam = hp.map2alm(gauss_map, lmax=args.lmax_len)
    tlm_masked = hp.almxfl(tlm_masked_beam,1./beam)
    del gauss_map,tlm_masked_beam

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
    hp.write_alm(f"mf_phi_alm_2048_200sims_unnorm_anoise_filter_beam_{args.fg_type}_{args.freq}{end}{suffix}.fits",mf_alm_unnorm,overwrite=True)
    print("Mean field saved.")
 


















































