import os
import argparse
import numpy as np
import healpy as hp
from lenspyx.lenspyx.utils import camb_clfile
from pixell import enmap,lensing as plensing,curvedsky as cs
from orphics import maps,stats,mpi,io
import pytempura 
import falafel.utils as futils 
from falafel import qe
import pickle

import mpi4py.rc
mpi4py.rc.threads = False
import MPI_error_handler

args = argparse.Namespace()

""" for stage gauss sims meanfield """

args.lmax_len = 3000 # desired lmax of the lensed field. #!! careful here, lmax of ilc is 799...
dlmax = 1000  # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
epsilon = 1e-6 # target accuracy of the output maps (execution time has a fairly weak dependence on this)
args.lmax_unl, args.mmax_unl = args.lmax_len + dlmax, args.lmax_len + dlmax
args.nside = 2048

sims_ids = np.arange(200) #low for now
args.nsims = len(sims_ids)

args.mask_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits' 
args.output_dir = '/data/data01/sophia/meanfield-maps/'
cls_path = "/home/spk24f/gitreps/lenspyx/lenspyx/data/cls/"
cl_gradlens = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_gradlensedCls.dat')) 
cl_unl = camb_clfile(os.path.join(cls_path,'FFP10_wdipole_lenspotentialCls.dat'))
cl_len = camb_clfile(os.path.join(cls_path,'FFP10_wdipole_lensedCls.dat'))


args.output_dir = '/data/data01/sophia/meanfield-maps/lensing_wnoise/'

#al norm: qe-temp-Als-anoise.pkl
output_path = lambda x: os.path.join(args.output_dir, x)

ests = ['TT']

mask = hp.read_map(args.mask_file)
fsky = np.mean(mask**2)

cl_gradlens_order = {"TT": cl_gradlens["tt"], "TE": cl_gradlens["te"], "EE": cl_gradlens['ee'], "BB": cl_gradlens['bb']}
for key,item in cl_gradlens_order.items(): 
    cl_gradlens_order[key] = item[:args.lmax_len+1]

cl_len_order = {"TT": cl_len["tt"], "TE": cl_len["te"], "EE": cl_len['ee'], "BB": cl_len['bb']}
for key,item in cl_len_order.items(): 
    cl_len_order[key] = item[:args.lmax_len+1]

cl_unl_order = {"TT": cl_unl["tt"], "TE": cl_unl["te"], "EE": cl_unl['ee'], "BB": cl_unl['bb']}
for key,item in cl_unl_order.items(): 
    cl_unl_order[key] = item[:args.lmax_len+1]


def get_gauss_hp_map_name(args, sim, fsky=None):
    
    tag = f'{args.nside}_{sim}'

    if fsky is not None:
        tag = f'{tag}_{fsky:.2f}'

    return f'hp_gauss_map_{tag}.fits'

def get_gauss_alm_name(args, sim, fsky=None):
    
    tag = f'{args.nside}_{sim}'

    if fsky is not None:
        tag = f'{tag}_{fsky:.2f}'

    return f'gauss_alm_{tag}.fits'

def get_mf_alm_name(args, fsky=None,norm=True):
    
    tag = f'{args.nside}_{args.nsims}sims'

    if fsky is not None:
        tag = f'{tag}_{fsky:.2f}'
    if norm==False: 
        tag = f'{tag}_unnorm'

    return f'mf_phi_alm_{tag}.fits'
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

def get_white_noise(beam_fwhm, noise_t, noise_p=None, lmax=3000):
    ls = np.arange(lmax + 1)
    if noise_p is None:
        noise_p = np.sqrt(2.) * noise_t
    nells = {}
    nells['TT'] = (noise_t * np.pi / 180. / 60.)**2. / maps.gauss_beam(beam_fwhm, ls)**2.
    nells['EE'] = (noise_p * np.pi / 180. / 60.)**2. / maps.gauss_beam(beam_fwhm, ls)**2.
    nells['BB'] = (noise_p * np.pi / 180. / 60.)**2. / maps.gauss_beam(beam_fwhm, ls)**2.
    return nells


#need to load all cl (t and pol) to use on norm

comm,rank,my_tasks = mpi.distribute(args.nsims)
s = stats.Stats(comm)

# Decide on a geometry for the intermediate operations
px = qe.pixelization(nside=2048) #CHANGED HERE
lmin = 600

for sim in my_tasks:
    
    print(f"Rank {rank} processing sim {sim}")

    tlm = hp.read_alm(output_path(get_gauss_alm_name(args, sim=sim, fsky=fsky))) 

    cltt_mask = hp.alm2cl(tlm,lmax=args.lmax_len,mmax=args.lmax_len) #ADDED THIS
    cltt_mask_smooth = smooth_cls(cltt_mask / fsky, points=300) #ADDED THIS

    #Filtering 
    Xdat0 = qe.filter_alms(tlm,cl_gradlens_order["TT"]/cltt_mask_smooth[:args.lmax_len+1],lmin,args.lmax_len) #CHANGED HERE
    Xdat = qe.filter_alms(tlm,1./cltt_mask_smooth[:args.lmax_len+1],lmin,args.lmax_len) #ADDED THIS

    recon = qe.qe_temperature_only(px=px,Xalm=Xdat0,Yalm=Xdat,mlmax=args.lmax_len) #CHANGED HERE XALM YALM AND MLMAX

    #unnorm_rkalms_g = plensing.phi_to_kappa(recon[0].astype(np.complex128)) #WILL NOT CONVERT TO KAPPA HERE. GET PHI, SEND TO MEMPHIS, NORMALIZE THEN CONVERT TO KAPPA 
    unnorm_rkalms_g = recon #CHANGED HERE
    unnorm_rkalm_g = futils.change_alm_lmax(unnorm_rkalms_g.astype(np.complex128),args.lmax_len)

    #s.add_to_stack(f'rf', rkalm_g.real)
    #s.add_to_stack(f'if', rkalm_g.imag)
    s.add_to_stack(f'rfc', unnorm_rkalm_g.real)
    s.add_to_stack(f'ifc', unnorm_rkalm_g.imag)

with io.nostdout():
    s.get_stacks()

if rank==0:

    #mf_alm = s.stacks[f'rf'] + 1j * s.stacks[f'if']
    mf_alm_unnorm = s.stacks[f'rfc'] + 1j * s.stacks[f'ifc']

    #mf_name = get_mf_name(args)
    """hp.write_alm(output_path(get_mf_alm_name(args,fsky,norm=True)), mf_alm,overwrite=True)"""
    #hp.write_alm(get_mf_alm_name(args,fsky,norm=False), mf_alm_unnorm, overwrite=True)
    hp.write_alm("mf_phi_alm_2048_200sims_0.61_unnorm_wnoise.fits", mf_alm_unnorm, overwrite=True)
    #hp.write_alm(output_path(mf_name[1]), mf_alm_curl, overwrite = True)"""

    

    

    

    









