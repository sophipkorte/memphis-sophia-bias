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

#############################################################
# PARAMETERS
#############################################################
lmax_len = 3000
lmin = 600
nside = 2048
nsims = 98
amp_arcmin = 5.3
ests = 'TT'

#############################################################
# LOAD EVERYTHING (same as your reconstruction code)
#############################################################
# mask, noise_scale, Als, cl_gradlens_order, tcl_filter, mf_alm
# ... (same loading code as your MCN0/MCN1 scripts)

mask_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits'
obs_file = '/home/spk24f/masks/LAT_hits_2048_galcoords.fits'

mask = hp.read_map(mask_file)

npix = 12 * nside**2
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

Als_dict = {}
with open(f'/home/spk24f/gitreps/lenspyx/examples/partsky_anisotropic_noise_nofg_noNILC/filter_run/lensing_partksy_anoise_norm.pkl', 'rb') as f:
    Als_dict = pickle.load(f)
Als = Als_dict[ests]

px = qe.pixelization(nside=nside)
qfunc = get_qfunc(px, cl_gradlens_order, lmax_len, ests, Al1=Als)

mf_alm = hp.read_alm('/home/spk24f/gitreps/sophia-bias/make_sims_mf/mf_phi_alm_2048_200sims_unnorm_anoise_filter.fits')
mf_alm = hp.almxfl(mf_alm, Als[0])

tcl_filter = {}
with open('/home/spk24f/gitreps/lenspyx/examples/partsky_anisotropic_noise_nofg_noNILC/filter_run/tcl_filter_anoise.pkl', 'rb') as f:
    tcl_filter = pickle.load(f)

#############################################################
# MPI SETUP
#############################################################
comm, rank, my_tasks = mpi.distribute(nsims)
s = stats.Stats(comm)

for sim in my_tasks:
    sim_idx = sim + 1  # match your sim indexing

    # 1. Load lensed CMB alm
    fname = f"/data/data01/sophia/bias-sims/set0/tlm_len_{sim_idx:04d}.fits"
    tlm = hp.read_alm(fname)

    # 2. Load corresponding input phi alm
    # You need to have saved these when generating sims!
    # If not saved, regenerate with same seed
    fname_phi = f"/data/data01/sophia/bias-sims/set0/plm_{sim_idx:04d}.fits"
    plm = hp.read_alm(fname_phi)

    # 3. Add anisotropic noise (same as MCN0/MCN1)
    np.random.seed(abs(hash((0, 0, sim_idx, 'noise'))) % (2**32))
    noiset = np.random.normal(scale=noise_scale, size=npix)
    tmap = hp.alm2map(tlm, nside=nside, lmax=lmax_len)
    del tlm
    tmap += noiset
    del noiset

    # 4. Apply mask
    tmap *= mask
    np.nan_to_num(tmap, copy=False)

    # 5. Get alms and filter (same as data)
    tlm_masked = hp.map2alm(tmap, lmax=lmax_len)
    del tmap
    alms_for_falafel = np.array([tlm_masked, np.zeros_like(tlm_masked), np.zeros_like(tlm_masked)])
    del tlm_masked
    filtered = utils.isotropic_filter(alms_for_falafel, tcl_filter, lmin, lmax_len)
    del alms_for_falafel

    # 6. Run QE and subtract mean field
    recon = qfunc([filtered[0], filtered[1], filtered[2]],
                  [filtered[0], filtered[1], filtered[2]])
    del filtered
    recon[0] = recon[0] - mf_alm

    # 7. Normalize with analytical A_L
    rkalm = plensing.phi_to_kappa(recon[0].astype(np.complex128)) #changed here, double norm
    rkalm = utils.change_alm_lmax(rkalm.astype(np.complex128), lmax_len)
    del recon

    # 8. Input kappa
    ikalm = plensing.phi_to_kappa(plm.astype(np.complex128))
    ikalm = utils.change_alm_lmax(ikalm.astype(np.complex128), lmax_len)
    del plm

    # 9. Compute cross and input auto
    cl_cross = hp.alm2cl(rkalm, ikalm)
    cl_input = hp.alm2cl(ikalm, ikalm)
    del rkalm, ikalm

    s.add_to_stack('cross', cl_cross)
    s.add_to_stack('input', cl_input)
    del cl_cross, cl_input
    gc.collect()

with io.nostdout():
    s.get_stacks()

if rank == 0:
    mean_cross = s.stacks['cross']
    mean_input = s.stacks['input']

    # MC Response
    R_L = mean_cross / mean_input

    np.save('mc_response_anoise.npy', R_L)
    print("MC response saved.")
    print("R_L:", R_L)