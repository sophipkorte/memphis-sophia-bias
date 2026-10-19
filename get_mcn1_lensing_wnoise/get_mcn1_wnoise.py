from __future__ import print_function
from orphics import maps,io,cosmology,stats
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
import pickle
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl
from lenspyx import lensing 
import matplotlib.pyplot as plt
from solenspipe.bias import mcn1
from solenspipe import get_qfunc

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filter-whiteamplitude", type=float, default=5.3, help='White noise level [muK-arcmin]')
parser.add_argument("--filter-whitefwhm", type=float, default=2.2, help='White noise beam fwhm [arcmin]')

args = parser.parse_args()
 
#--- get_kmap ---
def get_kmap(seed,tcl_len_order,lmin,lmax_len,Nl):
    icov, setidx, simidx = seed[:3]
    fname = f"/home/spk24f/gitreps/sophia-bias/make_sims_mv/sims/set{setidx}/alm_len_{simidx:04d}.fits"
    alm = hp.read_alm(fname,hdu=[1,2,3]) #this is hopefully reading all fields (T,E,B)
    nalm = hp.synalm((Nl['TT'],None,Nl['EE'],Nl['BB']),lmax=lmax_len) #this is the noise alms
    alms_for_falafel = np.array([alm[0]+nalm[0],alm[1]+nalm[1],alm[2]+nalm[2]]) #this is the array of alms that falafel expects (T,E,B)
    Xdat = utils.isotropic_filter(alms_for_falafel,tcl_len_order,lmin,lmax_len)
    return [Xdat[0], Xdat[1], Xdat[2]] #return all fields for now

###############################################
#DEFINE FUNCTIONS HERE 
################################################
def phitokappa(arrayCL):
    clresult = np.zeros(shape=(len(arrayCL),1))
    l = len(arrayCL)
    print(l)
    for i in range(l):
        clresult[i] = (i*(i+1)/2)**2 * arrayCL[i]
    return clresult

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

def power(a,b): 
    return hp.alm2cl(a,b)

def get_white_noise(beam_fwhm,noise_t,noise_p=None,lmax=9000):
    ls = np.arange(lmax+1)
    if noise_p is None: noise_p = np.sqrt(2.)*noise_t
    nells = {}
    nells['TT'] = (noise_t*np.pi/180./60.)**2. / maps.gauss_beam(beam_fwhm,ls)**2.
    nells['EE'] = (noise_p*np.pi/180./60.)**2. / maps.gauss_beam(beam_fwhm,ls)**2.
    nells['BB'] = (noise_p*np.pi/180./60.)**2. / maps.gauss_beam(beam_fwhm,ls)**2.

    return nells

#####################################################
# HERE ARE SOME VALUES WE WILL NEED 
#########################################################
lmax_len = 3000 # desired lmax of the lensed field. #!! careful here, lmax of ilc is 799...
lmin = 600
dlmax = 1000 # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
epsilon = 1e-6 # target accuracy of the output maps (execution time has a fairly weak dependence on this)
lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax
nside = 2048

#############################################################
# DEFINE ESTIMATOR HERE 
############################################################
ests = 'TT'

###########################################################
# HERE ARE THE CLS WE WILL NEED 
###########################################################
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


#########################################################
# READING IN NORMALIZATION (FOR THE ONLY LENSING CASE)
##########################################################
Als_dict = {}
with open(f'/home/spk24f/gitreps/lenspyx/examples/only_lensing_wnoise_test/{ests}/only_lensing_norm_{ests}.pkl','rb') as f: 
    Als_dict = pickle.load(f)

#############################################################
# CALL GET QFUNC TO DO THE QE 
############################################################
Als = Als_dict[ests]
px = qe.pixelization(nside=nside)
qfunc = get_qfunc(px,cl_gradlens_order,lmax_len,ests,Al1=Als)

#############################################################
# GET THE NOISE POWER SPECTRUM
#############################################################
noise_dict = get_white_noise(args.filter_whitefwhm,args.filter_whiteamplitude,lmax=lmax_len)
tcl_len_order = {key: val.copy() for key, val in cl_len_order.items()}
tcl_len_order['TT'] += noise_dict['TT']
tcl_len_order['EE'] += noise_dict['EE']
tcl_len_order['BB'] += noise_dict['BB']

###############################################################
#DEFINE A WRAPPER FUNCTION FOR GET_KMAP SO THAT IT CAN BE USED IN MCN1
############################################################### 
def get_kmap_wrapped(seed):
    return get_kmap(seed, tcl_len_order, lmin=lmin, lmax_len=lmax_len, Nl=noise_dict)


#--- Run MCN1 ---
nsims = 90
mcn1_result = mcn1(
    icov=0,
    get_kmap=get_kmap_wrapped,
    power=power,
    nsims=nsims,
    qfunc1=qfunc,
    verbose=True
)

#--- Save/print output ---
np.save(f"mcn1_{ests}.npy", mcn1_result)
print("MCN1 bias shape:", mcn1_result.shape)
print("MCN1 bias:", mcn1_result)