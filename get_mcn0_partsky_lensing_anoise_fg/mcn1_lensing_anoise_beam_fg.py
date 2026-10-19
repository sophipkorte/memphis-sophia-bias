import gc
import os
import numpy as np
import healpy as hp
import pickle
from orphics import maps, stats, mpi, io
from falafel import qe, utils 
from lenspyx.utils import camb_clfile
from solenspipe.bias import mcn1
from solenspipe import get_qfunc
import falafel.utils as futils

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import MPI_error_handler
import argparse

print("imports ok")

############################################################
#ASK FOR EXTRA ARGUMENTS 
############################################################
parser = argparse.ArgumentParser(prog='MCN0',description='mcn0 with foregrounds and mask')

parser.add_argument("--extra_mask",action="store_true",help='mask galactic plane')
parser.add_argument("--fg_type",type=str,default='d10',help='foreground type')
parser.add_argument("--freq",type=int,default=93,help='frequency channel')
parser.add_argument("--planck_mask",action="store_true",help='use Planck mask GAL060')

args = parser.parse_args()

#############################################################
# PARAMETERS
#############################################################
lmax_len = 3000
lmin = 600
nside = 2048
ests = 'TT'
amp_arcmin = 5.3
beam_fwhm = 2.2
SO_FREQ = [93, 145]

# mcn1 does i=i+1, so indices go from 1 to nsims
# If you have 100 sims (0-99), set nsims=99 so indices stay in range (1-99)
nsims = 99

mask_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits'
obs_file = '/home/spk24f/masks/LAT_hits_2048_galcoords.fits'
mask_gal_file = '/home/spk24f/masks/mask_binary_gal_cut_n2048_3deg_apo.fits'
mask_planck_file = '/home/spk24f/masks/Planck/Planck_GAL_060_n2048_3deg_apo.fits'
fg_base_dir = '/data/data01/sophia/fg-bias-sims/'

def get_output_dir(base_dir, fg_type, extra_mask):
    if extra_mask:
        mask_tag = 'extra_mask'
    else:
        mask_tag = 'lat_mask'
    d = os.path.join(base_dir, fg_type, mask_tag)
    os.makedirs(d, exist_ok=True)
    return d

#############################################################
# LOAD MASK
#############################################################
mask = hp.read_map(mask_file)
mask_gal = hp.read_map(mask_gal_file)
mask_planck = hp.read_map(mask_planck_file)

if args.planck_mask:
    mask2 = mask * mask_planck
    suffix = '_GAL060'
else: 
    mask2 = mask * mask_gal
    suffix = ''

if args.extra_mask == True:
    use_mask = mask2
    end = '_extra_mask'

else: 
    use_mask = mask
    end = '_lat_mask'

#############################################################
# PRECOMPUTE NOISE SCALE (once)
#############################################################
npix = 12 * nside**2
Omegapix = 4 * np.pi / npix
sigma2_muK2_sr = (amp_arcmin * np.pi / 180 / 60)**2
sigma2_pixel = sigma2_muK2_sr / Omegapix
nobs = hp.read_map(obs_file)
hits_safe = np.copy(nobs)
hits_safe[hits_safe <= 0] = np.nan
noise_scale = np.sqrt(sigma2_pixel / hits_safe)
del nobs, hits_safe

#############################################################
#PRECOMPUTE BEAM
#############################################################
beam = maps.gauss_beam(beam_fwhm,np.arange(lmax_len+1))

#############################################################
# LOAD CLS
#############################################################
cls_path = "/home/spk24f/gitreps/lenspyx/lenspyx/data/cls/"
cl_gradlens = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_gradlensedCls.dat'))
cl_len = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))

cl_gradlens_order = {
    "TT": cl_gradlens["tt"][:lmax_len+1],
    "TE": cl_gradlens["te"][:lmax_len+1],
    "EE": cl_gradlens["ee"][:lmax_len+1],
    "BB": cl_gradlens["bb"][:lmax_len+1]
}

cl_len_order = {
    "TT": cl_len["tt"][:lmax_len+1],
    "TE": cl_len["te"][:lmax_len+1],
    "EE": cl_len["ee"][:lmax_len+1],
    "BB": cl_len["bb"][:lmax_len+1]
}

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

def power(a, b):
    return hp.alm2cl(a, b)

def get_gauss_fg_alm_name(freq, fg_type, sim, fsky=None):
    tag = f'{freq}_{fg_type}_{sim}'
    if fsky is not None:
        tag = f'{tag}_{fsky:.2f}'
    return f'gauss_fg_alm_{tag}.fits'

#############################################################
# LOAD NORMALIZATION (computed with anisotropic noise)
#############################################################
Als_dict = {}
with open(f'/home/spk24f/gitreps/lenspyx/examples/partsky_lensing_anoise_fg_noNILC/{args.fg_type}/f{args.freq}/lensing_partksy_anoise_beam_{args.fg_type}_{args.freq}_norm{end}{suffix}.pkl', 'rb') as f:
    Als_dict = pickle.load(f)
Als = Als_dict[ests]

#############################################################
# QE SETUP
#############################################################
px = qe.pixelization(nside=nside)
qfunc = get_qfunc(px, cl_gradlens_order, lmax_len, ests, Al1=Als)

#############################################################
# LOAD MEAN FIELD (computed with anisotropic noise)
#############################################################
mf_file_unnorm = f'/home/spk24f/gitreps/sophia-bias/make_sims_mf/mf_phi_alm_2048_200sims_unnorm_anoise_filter_beam_{args.fg_type}_{args.freq}{end}{suffix}.fits'

mf_alm = hp.read_alm(mf_file_unnorm)
mf_alm = hp.almxfl(mf_alm, Als[0])  # normalize

#############################################################
# LOAD FILTER FROM DATA
#############################################################
tcl_filter = {}
with open(f'/home/spk24f/gitreps/lenspyx/examples/partsky_lensing_anoise_fg_noNILC/{args.fg_type}/f{args.freq}/tcl_filter_anoise_beam_{args.fg_type}_{args.freq}{end}{suffix}.pkl', 'rb') as f:
    tcl_filter = pickle.load(f)

#############################################################
# QFUNC WITH MEAN FIELD SUBTRACTION
#############################################################
def qfunc_mf(x, y):
    result = qfunc(x, y)
    result[0] = result[0] - mf_alm
    return result

#############################################################
# GET_KMAP FUNCTION
# MCN1 uses sets 0,1,2,3
# set 0 (S): unique CMB, unique phi
# set 1 (S'): unique CMB, unique phi
# set 2 (Sphi): unique CMB, shared phi with set 3
# set 3 (Sphi'): unique CMB, shared phi with set 2
#############################################################
def get_kmap(seed, tcl_len_order,args):
    icov, setidx, simidx = seed[:3]
    fname = f"/home/spk24f/gitreps/sophia-bias/make_sims/sims/set{setidx}/tlm_len_{simidx:04d}.fits"
    fg_dir = get_output_dir(fg_base_dir,args.fg_type, args.extra_mask)
    fg_name = os.path.join(fg_dir,get_gauss_fg_alm_name(args.freq, args.fg_type, simidx))
    tlm = hp.read_alm(fname)
    fg_alm = hp.read_alm(fg_name)
    tlm_beam = hp.almxfl(tlm+fg_alm,beam)
    del tlm, fg_alm

    # Convert to map
    tmap = hp.alm2map(tlm_beam, nside=nside, lmax=lmax_len)
    del tlm_beam

    # Add independent anisotropic noise
    # Unique seed per (icov, set, sim) so noise is independent
    np.random.seed(icov * 10**8 + setidx * 10**6 + simidx * 10**3 + 5)
    noiset = np.random.normal(scale=noise_scale, size=npix)
    tmap += noiset
    del noiset

    # Apply mask
    tmap *= mask
    np.nan_to_num(tmap, copy=False)

    # Get alms and deconvolve
    tlm_masked_beam = hp.map2alm(tmap, lmax=lmax_len)
    tlm_masked = hp.almxfl(tlm_masked_beam,1./beam)
    del tmap,tlm_masked_beam

    # Filter (same as data)
    alms_for_falafel = np.array([tlm_masked, np.zeros_like(tlm_masked), np.zeros_like(tlm_masked)])
    del tlm_masked
    Xdat = utils.isotropic_filter(alms_for_falafel, tcl_len_order, lmin, lmax_len)
    del alms_for_falafel

    gc.collect()
    return [Xdat[0], Xdat[1], Xdat[2]]

def get_kmap_wrapped(seed):
    return get_kmap(seed, tcl_filter, args)

#############################################################
# RUN MCN1
#############################################################
mcn1_result = mcn1(
    icov=0,
    get_kmap=get_kmap_wrapped,
    power=power,
    nsims=nsims,
    qfunc1=qfunc_mf,
    verbose=True
)

#############################################################
# SAVE OUTPUT
#############################################################
np.save(f"mcn1_anoise_{ests}_filter_beam_{args.freq}_{args.fg_type}{end}{suffix}.npy", mcn1_result)
print("MCN1 bias shape:", mcn1_result.shape)
print("MCN1 mean:", np.mean(mcn1_result, axis=0))