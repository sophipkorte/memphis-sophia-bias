from __future__ import print_function
from orphics import maps, io, cosmology, stats
from pixell import enmap, lensing as plensing, curvedsky, utils, enplot
from enlib import bench
import numpy as np
import os, sys
import healpy as hp
from falafel import qe, utils
import pytempura

# lenspyx imports
import os
import lenspyx
import pickle
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl
from lenspyx import lensing
import matplotlib.pyplot as plt
from solenspipe.bias import mcn1
from solenspipe import get_qfunc

#############################################################
# GET_KMAP FUNCTION
#############################################################
def smooth_cls(cl,points=300):
    """bin and interpolate a cl to smooth it"""
    bin_edges = np.linspace(2,len(cl),points).astype(int)
    cents,cls=bandedcls(cl,bin_edges)
    cls=maps.interp(cents,cls,fill_value="extrapolate")(np.arange(len(cl))) #added extrapolate here so that the smooth cl would not have zero and mess up the norm
    return cls

def bandedcls(cl,_bin_edges):
    ls=np.arange(cl.size)
    binner = stats.bin1D(_bin_edges)
    cents,bls = binner.bin(ls,cl)
    return cents,bls

def get_kmap(seed, tcl_len_order, lmin, lmax_len, mask, nside):
    """
    Returns filtered [Talm, Ealm, Balm] for solenspipe.
    Applies mask in map space before filtering.
    """
    icov, setidx, simidx = seed[:3]
    fname = f"/home/spk24f/gitreps/sophia-bias/make_sims/sims/set{setidx}/tlm_len_{simidx:04d}.fits"
    tlm = hp.read_alm(fname)

    # Convert to maps, apply mask, convert back to alms
    tmap = hp.alm2map(tlm, nside=nside, lmax=lmax_len)

    tmap_masked = tmap * mask
   
    tlm_masked = hp.map2alm(tmap_masked, lmax=lmax_len)
 
    alms_for_falafel = np.array([tlm_masked, np.zeros_like(tlm_masked), np.zeros_like(tlm_masked)])
    filtered = utils.isotropic_filter(alms_for_falafel, tcl_len_order, lmin, lmax_len)
    return [filtered[0], filtered[1], filtered[2]]


def new_get_kmap(seed,tcl_len_order,lmin,lmax_len, mask, nside):
    """
    Returns filtered [Talm, Ealm, Balm] for solenspipe.
    Applies mask in map space before filtering.
    """
    icov, setidx, simidx = seed[:3]
    fname = f"/home/spk24f/gitreps/sophia-bias/make_sims/sims/set{setidx}/tlm_len_{simidx:04d}.fits"
    tlm = hp.read_alm(fname)
    
    # Convert to maps, apply mask, convert back to alms
    tmap = hp.alm2map(tlm, nside=nside, lmax=lmax_len)
    
    tmap_masked = tmap * mask
    w2 = np.mean(mask**2)
       
    tlm_masked = hp.map2alm(tmap_masked, lmax=lmax_len)
    cl_tt_masked = hp.alm2cl(tlm_masked)
    cl_tt_masked_smooth = smooth_cls(cl_tt_masked/w2, points=300)
    tcl_len_order_smooth = {key: val.copy() for key, val in tcl_len_order.items()}
    tcl_len_order_smooth['TT'] = cl_tt_masked_smooth
     
    alms_for_falafel = np.array([tlm_masked, np.zeros_like(tlm_masked), np.zeros_like(tlm_masked)])
    filtered = utils.isotropic_filter(alms_for_falafel, tcl_len_order_smooth, lmin, lmax_len)
    return [filtered[0], filtered[1], filtered[2]]


###############################################
# DEFINE FUNCTIONS HERE
################################################
def power(a, b):
    return hp.alm2cl(a, b)


#####################################################
# HERE ARE SOME VALUES WE WILL NEED
#########################################################
lmax_len = 3000
lmin = 600
dlmax = 1000
lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax
nside = 2048

#############################################################
# DEFINE ESTIMATOR HERE
############################################################
ests = 'TT'

###########################################################
# LOAD MASK
###########################################################
mask_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits'
mask = hp.read_map(mask_file)
fsky = np.mean(mask**2)

###########################################################
# HERE ARE THE CLS WE WILL NEED
###########################################################
cls_path = "/home/spk24f/gitreps/lenspyx/lenspyx/data/cls/"
cl_gradlens = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_gradlensedCls.dat'))
cl_unl = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cl_len = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))

##############################################################
# REORDERING/GETTING THE RIGHT LMAX
##############################################################
cl_gradlens_order = {"TT": cl_gradlens["tt"][:lmax_len+1],
                     "TE": cl_gradlens["te"][:lmax_len+1],
                     "EE": cl_gradlens['ee'][:lmax_len+1],
                     "BB": cl_gradlens['bb'][:lmax_len+1]}

cl_len_order = {"TT": cl_len["tt"][:lmax_len+1],
                "TE": cl_len["te"][:lmax_len+1],
                "EE": cl_len["ee"][:lmax_len+1],
                "BB": cl_len["bb"][:lmax_len+1]}

##############################################################
# TOTAL CLS FOR FILTERING (lensed Cls / fsky to account for mask)
##############################################################
tcl_len_order = {key: val.copy() / fsky for key, val in cl_len_order.items()}

#########################################################
# READING IN NORMALIZATION
##########################################################
Als_dict = {}
with open(f'/home/spk24f/gitreps/lenspyx/examples/partsky_nonoise_nofg/n2048_3deg_apo/only_lensing_partsky_norm.pkl', 'rb') as f:
    Als_dict = pickle.load(f)

#############################################################
# CALL GET QFUNC TO DO THE QE
############################################################
Als = Als_dict[ests]
px = qe.pixelization(nside=nside)
qfunc = get_qfunc(px, cl_gradlens_order, lmax_len, ests, Al1=Als)

#############################################################
# LOAD MEAN FIELD
#############################################################
mf_alm = hp.read_alm(f'/home/spk24f/gitreps/sophia-bias/make_sims_mf/mf_phi_alm_2048_200sims_0.61_unnorm.fits')
# If unnormalized, normalize here:
mf_alm = hp.almxfl(mf_alm, Als[0])  # gradient component normalization

#############################################################
# DEFINE QFUNC WITH MEAN FIELD SUBTRACTION
#############################################################
def qfunc_mf(x, y):
    """QE with mean field subtracted"""
    result = qfunc(x, y) #I think this is wrong. 
    result[0] = result[0] - mf_alm  # subtract MF from gradient
    return result

###############################################################
# DEFINE A WRAPPER FUNCTION FOR GET_KMAP
###############################################################
def get_kmap_wrapped(seed):
    return get_kmap(seed, tcl_len_order, lmin=lmin, lmax_len=lmax_len, mask=mask, nside=nside)

def new_get_kmap_wrapped(seed): 
    return new_get_kmap(seed,cl_len_order,lmin=lmin,lmax_len=lmax_len,mask=mask,nside=nside)

#############################################################
# RUN MCN1
#############################################################
nsims = 90
mcn1_result = mcn1(
    icov=0,
    get_kmap=new_get_kmap_wrapped,
    power=power,
    nsims=nsims,
    qfunc1=qfunc_mf,
    verbose=True
)

#############################################################
# SAVE OUTPUT
#############################################################
np.save(f"mcn1_mask_{ests}_filter.npy", mcn1_result)
print("MCN1 bias shape:", mcn1_result.shape)
print("MCN1 mean:", np.mean(mcn1_result, axis=0))