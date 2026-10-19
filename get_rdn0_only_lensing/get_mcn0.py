from __future__ import print_function
import numpy as np
import os
import healpy as hp
import pickle
from falafel import qe, utils
from lenspyx.utils import camb_clfile
from solenspipe.bias import mcrdn0
from solenspipe import get_qfunc


#############################################################
# GET_KMAP FUNCTION
#############################################################
def get_kmap(seed,tcl_len_order,lmin,lmax_len):
    """
    Returns filtered [Talm, Ealm, Balm] for solenspipe.
    
    MCN0 only uses set=0 and set=1.
    mcrdn0 internally does i=i+2, so simidx starts at 2.
    Make sure your simulation files exist for these indices.
    """
    icov, setidx, simidx = seed[:3]
    fname = f"/home/spk24f/gitreps/sophia-bias/make_sims/sims/set{setidx}/tlm_len_{simidx:04d}.fits"
    tlm = hp.read_alm(fname) #for now doing only RDN0 for TT, so only reading Talm. If we want to do EE or TE, we will need to read the other alms as well.
    alms_for_falafel = np.array([tlm,np.zeros_like(tlm),np.zeros_like(tlm)]) #this is the array of alms that falafel expects (T,E,B)
    filtered = utils.isotropic_filter(alms_for_falafel, tcl_len_order, lmin, lmax_len)
    return [filtered[0], filtered[1], filtered[2]]


def power(a,b): 
    return hp.alm2cl(a,b)


#############################################################
# PARAMETERS
#############################################################
lmax_len = 3000
nside = 2048
ests = 'TT'
lmin = 600

# IMPORTANT: mcrdn0 internally does i=i+2, so sim indices go from 2 to nsims+1.
# If you generated 100 sims (indices 0-99), set nsims=98 here so indices stay in range (2-99).
nsims = 98

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
# LOAD NORMALIZATION
#############################################################
Als_dict = {}
with open(f'/home/spk24f/gitreps/lenspyx/examples/only_lensing_test/only_lensing_test_v1/{ests}/only_lensing_norm_{ests}.pkl', 'rb') as f:
    Als_dict = pickle.load(f)
Als = Als_dict[ests]

#############################################################
# QE SETUP
#############################################################
px = qe.pixelization(nside=nside)
qfunc = get_qfunc(px, cl_gradlens_order, lmax_len, ests, Al1=Als)

def get_kmap_wrapped(seed):
    return get_kmap(seed, cl_len_order, lmin=600, lmax_len=lmax_len)

#############################################################
# RUN MCN0
#############################################################

rdn0_result, mcn0_result = mcrdn0(
    icov=0,
    get_kmap=get_kmap_wrapped,
    power=power,
    nsims=nsims,
    qfunc1=qfunc,
    Xdat=None,
    use_mpi=True,
    verbose=True,
    skip_rd=True
)

#############################################################
# SAVE OUTPUT
#############################################################
np.save(f"mcn0_{ests}.npy", mcn0_result)
print("MCN0 bias shape:", mcn0_result.shape)
print("MCN0 mean:", np.mean(mcn0_result, axis=0))