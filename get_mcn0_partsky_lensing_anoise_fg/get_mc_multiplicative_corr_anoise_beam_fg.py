#falafel imports 
from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,lensing as plensing,curvedsky, utils, enplot
from enlib import bench
import numpy as np
import os,sys
import healpy as hp
from falafel import qe,utils
import pytempura
from solenspipe import get_qfunc

#lenspyx imports 
import os
import lenspyx
import pickle
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl
from lenspyx import lensing 
import matplotlib.pyplot as plt
import argparse
import gc

import mpi4py.rc
mpi4py.rc.threads = False
import MPI_error_handler

############################################################
#ASK FOR EXTRA ARGUMENTS 
############################################################
parser = argparse.ArgumentParser(prog='MC multiplicative',description='mc multiplicative correction with foregrounds and mask')

parser.add_argument("--extra_mask",action="store_true",help='mask galactic plane')
parser.add_argument("--fg_type",type=str,default='d10',help='foreground type')
parser.add_argument("--freq",type=int,default=93,help='frequency channel')

args = parser.parse_args()

##############################################################
#FUNCTIONS
##############################################################
def phitokappa(arrayCL):
    clresult = np.zeros(shape=(len(arrayCL),))
    l = len(arrayCL)
    for i in range(l):
        clresult[i] = (i*(i+1)/2)**2 * arrayCL[i]
    return clresult


def get_gauss_fg_alm_name(freq, fg_type, sim, fsky=None):
    tag = f'{freq}_{fg_type}_{sim}'
    if fsky is not None:
        tag = f'{tag}_{fsky:.2f}'
    return f'gauss_fg_alm_{tag}.fits'

#############################################################
# PARAMETERS
#############################################################
lmax_len = 3000
lmin = 600
nside = 2048
nsims = 98
amp_arcmin = 5.3
ests = 'TT'
beam_fwhm = 2.2
SO_FREQ = [93, 145]

#############################################################
# LOAD EVERYTHING (same as your reconstruction code)
#############################################################
# mask, noise_scale, Als, cl_gradlens_order, tcl_filter, mf_alm
# ... (same loading code as your MCN0/MCN1 scripts)

mask_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits'
obs_file = '/home/spk24f/masks/LAT_hits_2048_galcoords.fits'
mask_gal_file = '/home/spk24f/masks/mask_binary_gal_cut_n2048_3deg_apo.fits'
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
mask2 = mask * mask_gal

if args.extra_mask == True:
    use_mask = mask2
    end = '_extra_mask'

else: 
    use_mask = mask
    end = '_lat_mask'

npix = use_mask.size
pmap = 4 * np.pi / npix
w2 = np.sum((use_mask**2) * pmap) / np.pi / 4.
w4 = np.sum((use_mask**4) * pmap) / np.pi / 4.

#############################################################
# PRECOMPUTE NOISE SCALE (once)
#############################################################
numpix = 12 * nside**2
Omegapix = 4 * np.pi / numpix
sigma2_muK2_sr = (amp_arcmin * np.pi / 180 / 60)**2
sigma2_pixel = sigma2_muK2_sr / Omegapix
nobs = hp.read_map(obs_file)
hits_safe = np.copy(nobs)
hits_safe[hits_safe <= 0] = np.nan
noise_scale = np.sqrt(sigma2_pixel / hits_safe)
del nobs, hits_safe

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
# LOAD NORMALIZATION (computed with anisotropic noise)
#############################################################
Als_dict = {}
with open(f'/home/spk24f/gitreps/lenspyx/examples/partsky_lensing_anoise_fg_noNILC/{args.fg_type}/lensing_partksy_anoise_beam_{args.fg_type}_norm{end}.pkl', 'rb') as f:
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
mf_file_unnorm = f'/home/spk24f/gitreps/sophia-bias/make_sims_mf/mf_phi_alm_2048_200sims_unnorm_anoise_filter_beam_{args.fg_type}{end}.fits'

mf_alm = hp.read_alm(mf_file_unnorm)
mf_alm = hp.almxfl(mf_alm, Als[0])  # normalize

#############################################################
#LOAD OBSERVED DATA (for filtering)
#############################################################
tcl_filter = {}
with open(f'/home/spk24f/gitreps/lenspyx/examples/partsky_lensing_anoise_fg_noNILC/{args.fg_type}/tcl_filter_anoise_beam_{args.fg_type}{end}.pkl', 'rb') as f:
    tcl_filter = pickle.load(f)

cls_dict = {}
with open(f'/home/spk24f/gitreps/lenspyx/examples/partsky_lensing_anoise_fg_noNILC/{args.fg_type}/cls_dict_anoise_beam_{args.fg_type}{end}.pkl', 'rb') as f:
    cls_dict = pickle.load(f)    

acls = cls_dict["recon"]
icls = cls_dict["input"]
cls = cls_dict["cross"]
ells = np.arange(len(acls))

##############################################################
#LOAD MCN0 and MCN1 (if available)
##############################################################
try:
    mcn0_file = f'/home/spk24f/gitreps/sophia-bias/get_mcn0_partsky_lensing_anoise_fg/mcn0_anoise_TT_filter_beam_{args.freq}_{args.fg_type}{end}.npy'
    mcn0_raw = np.load(mcn0_file)
    mcn0_mean = np.mean(mcn0_raw, axis=0)
    mcn0_phi = mcn0_mean[0] 
    has_mcn0 = True
except:
    has_mcn0 = False

if has_mcn0:
    mcn0_kappa = (ells * (ells + 1.) / 2)**2 * mcn0_phi / w4
else: 
    mcn0_kappa = None

try:
    mcn1_file = f'/home/spk24f/gitreps/sophia-bias/get_mcn0_partsky_lensing_anoise_fg/mcn1_anoise_TT_filter_beam_{args.freq}_{args.fg_type}{end}.npy'
    mcn1_raw = np.load(mcn1_file)
    mcn1_mean = np.mean(mcn1_raw, axis=0)
    mcn1_phi = mcn1_mean[0] 
    has_mcn1 = True
except:
    has_mcn1 = False

if has_mcn1:
    mcn1_kappa = (ells * (ells + 1.) / 2)**2 * mcn1_phi / w4


############################################################
#PRECOMPUTE BEAM
############################################################
beam = maps.gauss_beam(beam_fwhm,np.arange(lmax_len+1))

#############################################################
# MPI SETUP
#############################################################

comm, rank, my_tasks = mpi.distribute(nsims)
s = stats.Stats(comm)

for sim in my_tasks:
    sim_idx = sim + 1

    # 1. Load lensed CMB
    fname = f"/data/data01/sophia/bias-sims/set2/tlm_len_{sim_idx:04d}.fits" #CHANGED SET2
    fg_dir = get_output_dir(fg_base_dir,args.fg_type, args.extra_mask)
    fg_name = os.path.join(fg_dir,get_gauss_fg_alm_name(args.freq, args.fg_type, sim_idx))
    tlm = hp.read_alm(fname)
    fg_alm = hp.read_alm(fg_name)
    tlm_beam = hp.almxfl(tlm+fg_alm,beam)
    del tlm, fg_alm

    # 2. Add noise
    np.random.seed(abs(hash((0, 0, sim_idx, 'noise'))) % (2**32))
    tmap = hp.alm2map(tlm_beam, nside=nside, lmax=lmax_len)
    del tlm_beam
    noiset = np.random.normal(scale=noise_scale, size=npix)
    tmap += noiset
    del noiset

    # 3. Mask
    tmap *= mask
    np.nan_to_num(tmap, copy=False)

    # 4. Filter (same as data)
    tlm_masked_beam = hp.map2alm(tmap, lmax=lmax_len)
    tlm_masked = hp.almxfl(tlm_masked_beam,1./beam)
    del tmap,tlm_masked_beam
    alms_for_falafel = np.array([tlm_masked, np.zeros_like(tlm_masked), np.zeros_like(tlm_masked)])
    del tlm_masked
    filtered = utils.isotropic_filter(alms_for_falafel, tcl_filter, lmin, lmax_len)
    del alms_for_falafel

    # 5. QE + MF subtraction (qfunc already includes Als normalization)
    recon = qfunc([filtered[0], filtered[1], filtered[2]],
                  [filtered[0], filtered[1], filtered[2]])
    del filtered
    recon[0] = recon[0] - mf_alm

    # 6. Convert to kappa (NO extra Als multiplication)
    rkalm = plensing.phi_to_kappa(recon[0].astype(np.complex128))
    rkalm = utils.change_alm_lmax(rkalm.astype(np.complex128), lmax_len)
    del recon

    # 7. Auto spectrum with w4 correction
    cl_auto = hp.alm2cl(rkalm, rkalm) / w4
    del rkalm

    # 8. Subtract MCN0 and MCN1 (same as you do for data)
    cl_debiased = cl_auto - mcn0_kappa
    if has_mcn1:
        cl_debiased = cl_debiased - mcn1_kappa
    del cl_auto

    s.add_to_stack('debiased', cl_debiased)
    del cl_debiased
    gc.collect()

with io.nostdout():
    s.get_stacks()

if rank == 0:
    mean_debiased = s.stacks['debiased']

    # MC correction factor
    # icls = input kappa Cl (theory)

    MC_corr = icls / mean_debiased

    np.save(f'mc_corr_anoise_beam_{args.freq}_{args.fg_type}{end}.npy', MC_corr)
    np.save(f'mean_debiased_anoise_beam_{args.freq}_{args.fg_type}{end}.npy', mean_debiased)
    print("MC correction saved.")
