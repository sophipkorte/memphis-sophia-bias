import gc
import os
import numpy as np
import healpy as hp
import pickle
from orphics import maps, stats, mpi, io
from falafel import qe, utils
from lenspyx.utils import camb_clfile
from solenspipe.bias import mcrdn0
from solenspipe import get_qfunc
import falafel.utils as futils

import mpi4py.rc
mpi4py.rc.threads = False
import MPI_error_handler

#############################################################
# PARAMETERS
#############################################################
lmax_len = 3000
lmin = 600
nside = 2048
ests = 'TT'
amp_arcmin = 5.3
beam_fwhm = 2.2

# IMPORTANT: mcrdn0 does i=i+2, so indices go from 2 to nsims+1
# If you have 100 sims (0-99), set nsims=98 so indices stay in range (2-99)
nsims = 98

mask_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits'
obs_file = '/home/spk24f/masks/LAT_hits_2048_galcoords.fits'

#############################################################
# LOAD MASK
#############################################################
mask = hp.read_map(mask_file)
fsky = np.mean(mask**2)

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

############################################################
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

#############################################################
# LOAD NORMALIZATION (computed with anisotropic noise)
#############################################################
Als_dict = {}
with open(f'/home/spk24f/gitreps/lenspyx/examples/partsky_anisotropic_noise_nofg_noNILC/filter_run_beam/lensing_partksy_anoise_beam_norm.pkl', 'rb') as f:
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
mf_alm = hp.read_alm('/home/spk24f/gitreps/sophia-bias/make_sims_mf/mf_phi_alm_2048_200sims_unnorm_anoise_filter_beam.fits')
mf_alm = hp.almxfl(mf_alm, Als[0])  # normalize

#############################################################
#LOAD OBSERVED DATA (for filtering)
#############################################################
tcl_filter = {}
with open('/home/spk24f/gitreps/lenspyx/examples/partsky_anisotropic_noise_nofg_noNILC/filter_run_beam/tcl_filter_anoise_beam.pkl', 'rb') as f:
    tcl_filter = pickle.load(f)


#############################################################
# QFUNC WITH MEAN FIELD SUBTRACTION
#############################################################
def qfunc_mf(x, y):
    result = qfunc(x, y)
    result[0] = result[0] - mf_alm
    return result

#############################################################
# GET_KMAP FUNCTION (memory efficient, per-sim pseudo-Cl)
#############################################################
def get_kmap(seed,tcl_len_order):
    icov, setidx, simidx = seed[:3]
    fname = f"/home/spk24f/gitreps/sophia-bias/make_sims/sims/set{setidx}/tlm_len_{simidx:04d}.fits"
    tlm = hp.read_alm(fname)
    tlm_beam = hp.almxfl(tlm,beam)

    # Convert to map
    tmap = hp.alm2map(tlm_beam, nside=nside, lmax=lmax_len)
    del tlm,tlm_beam

    # Add independent anisotropic noise
    np.random.seed(abs(hash((icov, setidx, simidx, 'noise'))) % (2**32))
    noiset = np.random.normal(scale=noise_scale, size=npix)
    tmap += noiset
    del noiset

    # Apply mask
    tmap *= mask
    np.nan_to_num(tmap, copy=False)

    # Get alms & deconlvolve
    tlm_masked_beam = hp.map2alm(tmap, lmax=lmax_len)
    tlm_masked = hp.almxfl(tlm_masked_beam,1./beam)
    del tmap,tlm_masked_beam

    # Filter
    alms_for_falafel = np.array([tlm_masked, np.zeros_like(tlm_masked),np.zeros_like(tlm_masked)])
    Xdat = utils.isotropic_filter(alms_for_falafel, tcl_len_order, lmin, lmax_len)
    del tlm_masked
    del alms_for_falafel

    gc.collect()
    return [Xdat[0], Xdat[1], Xdat[2]]

def new_get_kmap_wrapped(seed): 
    return get_kmap(seed,tcl_filter)

#############################################################
# RUN MCN0
#############################################################
rdn0, mcn0_result = mcrdn0(
    icov=0,
    get_kmap=new_get_kmap_wrapped,
    power=power,
    nsims=nsims,
    qfunc1=qfunc_mf,
    Xdat=None,
    use_mpi=True,
    verbose=True,
    skip_rd=True
)

#############################################################
# SAVE OUTPUT
#############################################################
np.save(f"mcn0_anoise_{ests}_filter_beam.npy", mcn0_result)
print("MCN0 bias shape:", mcn0_result.shape)
print("MCN0 mean:", np.mean(mcn0_result, axis=0))