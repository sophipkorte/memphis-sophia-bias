#falafel imports 
from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,lensing as plensing,curvedsky, utils, enplot
from enlib import bench
import numpy as np
import os,sys
import healpy as hp
from falafel import qe,utils as futils
from solenspipe import get_qfunc

import os
import lenspyx
import pickle
from lenspyx.utils import camb_clfile
import matplotlib.pyplot as plt
import gc

import mpi4py.rc
mpi4py.rc.threads = False
import MPI_error_handler

##############################################################
# FUNCTIONS
##############################################################
def phitokappa(arrayCL):
    clresult = np.zeros(shape=(len(arrayCL),))
    l = len(arrayCL)
    for i in range(l):
        clresult[i] = (i*(i+1)/2)**2 * arrayCL[i]
    return clresult

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
# PARAMETERS
#############################################################
lmax_len = 3000
lmin = 600
nside = 2048
nsims = 98
amp_arcmin = 5.3
ests = 'TT'
beam_fwhm = 2.2

#############################################################
# LOAD EVERYTHING
#############################################################
mask_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits'
obs_file = '/home/spk24f/masks/LAT_hits_2048_galcoords.fits'

mask = hp.read_map(mask_file)
npix = 12 * nside**2
pmap = 4 * np.pi / npix
w2 = np.sum((mask**2) * pmap) / np.pi / 4.
w4 = np.sum((mask**4) * pmap) / np.pi / 4.

Omegapix = 4 * np.pi / npix
sigma2_muK2_sr = (amp_arcmin * np.pi / 180 / 60)**2
sigma2_pixel = sigma2_muK2_sr / Omegapix
nobs = hp.read_map(obs_file)
hits_safe = np.copy(nobs)
hits_safe[hits_safe <= 0] = np.nan
noise_scale = np.sqrt(sigma2_pixel / hits_safe)
del nobs, hits_safe

cls_path = "/home/spk24f/gitreps/lenspyx/lenspyx/data/cls/"
cl_gradlens = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_gradlensedCls.dat'))
cl_len = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cl_unl = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))

cl_gradlens_order = {
    "TT": cl_gradlens["tt"][:lmax_len+1],
    "TE": cl_gradlens["te"][:lmax_len+1],
    "EE": cl_gradlens["ee"][:lmax_len+1],
    "BB": cl_gradlens["bb"][:lmax_len+1]
}

Als_dict = {}
with open(f'/home/spk24f/gitreps/lenspyx/examples/partsky_anisotropic_noise_nofg_noNILC/filter_run_beam/lensing_partksy_anoise_beam_norm.pkl', 'rb') as f:
    Als_dict = pickle.load(f)
Als = Als_dict[ests]

px = qe.pixelization(nside=nside)
qfunc = get_qfunc(px, cl_gradlens_order, lmax_len, ests, Al1=Als)

mf_alm = hp.read_alm('/home/spk24f/gitreps/sophia-bias/make_sims_mf/mf_phi_alm_2048_200sims_unnorm_anoise_filter_beam.fits')
mf_alm = hp.almxfl(mf_alm, Als[0])

tcl_filter = {}
with open('/home/spk24f/gitreps/lenspyx/examples/partsky_anisotropic_noise_nofg_noNILC/filter_run_beam/tcl_filter_anoise_beam.pkl', 'rb') as f:
    tcl_filter = pickle.load(f)

#############################################################
# LOAD INPUT KAPPA, MCN0, MCN1, MC CORRECTION
#############################################################
cls_dict = {}
with open('/home/spk24f/gitreps/lenspyx/examples/partsky_anisotropic_noise_nofg_noNILC/filter_run_beam/cls_dict_anoise_beam.pkl', 'rb') as f:
    cls_dict = pickle.load(f)
icls = cls_dict["input"]

mcn0_raw = np.load('/home/spk24f/gitreps/sophia-bias/get_mcn0_partsky_lensing_anoise/mcn0_anoise_TT_filter_beam.npy')
mcn0_mean = np.mean(mcn0_raw, axis=0)
mcn0_kappa = phitokappa(mcn0_mean[0]) / w4

mcn1_raw = np.load('/home/spk24f/gitreps/sophia-bias/get_mcn0_partsky_lensing_anoise/mcn1_anoise_TT_filter_beam.npy')
mcn1_mean = np.mean(mcn1_raw, axis=0)
mcn1_kappa = phitokappa(mcn1_mean[0]) / w4

MC_corr_raw = np.load('mc_corr_anoise_beam.npy')
MC_corr_smooth = smooth_cls(MC_corr_raw, points=100)
MC_corr_safe = np.copy(MC_corr_smooth) 
MC_corr_safe[MC_corr_safe < 0.1] = np.nan 
MC_corr_safe[MC_corr_safe > 10.0] = np.nan 

##############################################################
#PRECOMPUTE BEAM 
###############################################################
beam = maps.gauss_beam(beam_fwhm,np.arange(lmax_len+1))

#############################################################
# MPI SETUP
#############################################################
comm, rank, my_tasks = mpi.distribute(nsims)

# Each rank collects its spectra
local_debiased = []
local_corrected = []

for sim in my_tasks:
    sim_idx = sim + 1
    print(f"Rank {rank} processing sim {sim_idx}", flush=True)

    # 1. Load lensed CMB
    fname = f"/data/data01/sophia/bias-sims/set2/tlm_len_{sim_idx:04d}.fits" #CHANGED TO SET2
    tlm = hp.read_alm(fname)
    tlm_beam = hp.almxfl(tlm,beam)

    # 2. Add noise
    np.random.seed(abs(hash((0, 0, sim_idx, 'noise'))) % (2**32))
    tmap = hp.alm2map(tlm_beam, nside=nside, lmax=lmax_len)
    del tlm,tlm_beam
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
    filtered = futils.isotropic_filter(alms_for_falafel, tcl_filter, lmin, lmax_len)
    del alms_for_falafel

    # 5. QE + MF subtraction
    recon = qfunc([filtered[0], filtered[1], filtered[2]],
                  [filtered[0], filtered[1], filtered[2]])
    del filtered
    recon[0] = recon[0] - mf_alm

    # 6. Convert to kappa
    rkalm = plensing.phi_to_kappa(recon[0].astype(np.complex128))
    rkalm = futils.change_alm_lmax(rkalm.astype(np.complex128), lmax_len)
    del recon

    # 7. Auto spectrum
    cl_auto = hp.alm2cl(rkalm, rkalm) / w4
    del rkalm

    # 8. Debias
    cl_debiased = cl_auto - mcn0_kappa - mcn1_kappa
    del cl_auto

    # 9. Apply MC correction
    cl_corrected = cl_debiased * MC_corr_raw #CHANGED FROM SAFE TO RAW...

    # 10. Save both versions
    local_debiased.append(cl_debiased.copy())
    local_corrected.append(cl_corrected.copy())
    del cl_debiased, cl_corrected
    gc.collect()

#############################################################
# GATHER ALL SPECTRA ACROSS RANKS
#############################################################
all_debiased = utils.allgatherv(local_debiased, comm)
all_corrected = utils.allgatherv(local_corrected, comm)

if rank == 0: 
    all_debiased = np.array(all_debiased)
    all_corrected = np.array(all_corrected)

    #########################################################
    # SAVE RAW SPECTRA
    #########################################################
    np.save('/home/spk24f/gitreps/sophia-bias/make_sims_mf/errorbars_anoise_beam/all_debiased_spectra_anoise_beam_set2.npy', all_debiased)
    np.save('/home/spk24f/gitreps/sophia-bias/make_sims_mf/errorbars_anoise_beam/all_corrected_spectra_anoise_beam_set2.npy', all_corrected)

    #########################################################
    # COMPUTE MEAN AND STD
    #########################################################
    mean_debiased = np.mean(all_debiased, axis=0)
    std_debiased = np.std(all_debiased, axis=0)

    mean_corrected = np.mean(all_corrected, axis=0)
    std_corrected = np.std(all_corrected, axis=0)

    np.save('/home/spk24f/gitreps/sophia-bias/make_sims_mf/errorbars_anoise_beam/mean_debiased_anoise_beam_set2.npy', mean_debiased)
    np.save('/home/spk24f/gitreps/sophia-bias/make_sims_mf/errorbars_anoise_beam/std_debiased_anoise_beam_set2.npy', std_debiased)
    np.save('/home/spk24f/gitreps/sophia-bias/make_sims_mf/errorbars_anoise_beam/mean_corrected_anoise_beam_set2.npy', mean_corrected)
    np.save('/home/spk24f/gitreps/sophia-bias/make_sims_mf/errorbars_anoise_beam/std_corrected_anoise_beam_set2.npy', std_corrected)

   