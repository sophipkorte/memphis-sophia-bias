from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,lensing as plensing,curvedsky, utils, enplot
from enlib import bench
import numpy as np
import os,sys
import healpy as hp
from falafel import qe,utils
import pytempura

#lenspyx imports 
import os
import lenspyx
from lenspyx.utils import camb_clfile


#############################################################
# GET_KMAP FOR MEAN FIELD (with mask)
#############################################################
def get_kmap_mf(seed, tcl_len_order, lmin, lmax_len, nside, mask):
    """
    Load lensed CMB, apply mask, convert to alm, filter.
    """
    icov, setidx, simidx = seed[:3]
    fname = f"/home/spk24f/gitreps/sophia-bias/make_sims/sims/set{setidx}/tlm_len_{simidx:04d}.fits"
    tlm = hp.read_alm(fname)
    
    # Convert to map, apply mask, convert back to alm
    tmap = hp.alm2map(tlm, nside, lmax=lmax_len)
    tmap_masked = tmap * mask
    tlm_masked = hp.map2alm(tmap_masked, lmax=lmax_len)
    
    alms_for_falafel = np.array([tlm_masked, np.zeros_like(tlm_masked), np.zeros_like(tlm_masked)])
    filtered = utils.isotropic_filter(alms_for_falafel, tcl_len_order, lmin, lmax_len)
    return [filtered[0], filtered[1], filtered[2]]


#############################################################
# PARAMETERS
#############################################################
lmax_len = 3000
nside = 2048
lmin = 600
ests = 'TT'
nsims = 98  # number of sims to average over

#############################################################
# LOAD MASK
#############################################################
mask = hp.read_map("/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits")


#############################################################
# DEFINE ESTIMATOR HERE 
############################################################
ests = 'TT'

#############################################################
# LOAD CLS (same as before)
#############################################################
cls_path = cls_path = "/home/spk24f/gitreps/lenspyx/lenspyx/data/cls/"
cl_gradlens = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_gradlensedCls.dat')) 
cl_unl = camb_clfile(os.path.join(cls_path,'FFP10_wdipole_lenspotentialCls.dat'))
cl_len = camb_clfile(os.path.join(cls_path,'FFP10_wdipole_lensedCls.dat'))

##############################################################
# REORDERING/GETTING THE RIGHT LMAX 
##############################################################
cl_gradlens_order = {"TT": cl_gradlens["tt"], "TE": cl_gradlens["te"], "EE": cl_gradlens['ee'], "BB": cl_gradlens['bb']}
for key,item in cl_gradlens_order.items(): 
    cl_gradlens_order[key] = item[:lmax_len+1]

cl_len_order = {"TT": cl_len["tt"], "TE": cl_len["te"], "EE": cl_len['ee'], "BB": cl_len['bb']}
for key,item in cl_len_order.items(): 
    cl_len_order[key] = item[:lmax_len+1]

cl_unl_order = {"TT": cl_unl["tt"], "TE": cl_unl["te"], "EE": cl_unl['ee'], "BB": cl_unl['bb']}
for key,item in cl_unl_order.items(): 
    cl_unl_order[key] = item[:lmax_len+1]  


#############################################################
# QE SETUP (same as before)
#############################################################
# px = qe.pixelization(nside=nside)
# qfunc = get_qfunc(px, cl_gradlens_order, lmax_len, ests, Al1=Als)


#############################################################
# CALCULATE MEAN FIELD
#############################################################
comm, rank, my_tasks = mpi.distribute(nsims)

mf_accum = None
count = 0

for i in my_tasks:
    i = i + 2  # match the index offset convention
    seed = (0, 0, i)  # use set0 sims
    Xsim = get_kmap_mf(seed, cl_len_order, lmin, lmax_len, nside, mask)
    
    # Run QE on the sim with itself
    phi_sim = qfunc(Xsim, Xsim)
    
    if mf_accum is None:
        mf_accum = phi_sim.copy()
    else:
        mf_accum += phi_sim
    count += 1

# Gather across ranks and average
from pixell import utils as putils
all_mf = putils.allgatherv([mf_accum], comm)
all_counts = putils.allgatherv([count], comm)

mean_field = np.sum(all_mf, axis=0) / np.sum(all_counts)

#############################################################
# SAVE
#############################################################
np.save(f"mean_field_{ests}.npy", mean_field)
print("Mean field shape:", mean_field.shape)