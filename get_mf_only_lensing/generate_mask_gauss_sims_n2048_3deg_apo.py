# read nilc t map with the residual foregrounds and anisotropic noise 
# get power spectrum from this map? -> I should probably get all the power spectra once and save it pkl and just keep using the same.
# synfast a bunch of maps with the same power spectrum 
# apply the same mask I used to make the t map that went into nilc 
# run the synfast * mask maps through falafel
# get a bunch of phi (or kappa) reconstruction alms 
# get the mean of the alms? what kind of mean is this? 
# subtract the mean from the original reconstruction 
# initially use nside 512, not a lot of sims
# then can try bigger nside, then bigger sims numb

import os
import argparse
import numpy as np
import healpy as hp
from pixell import enmap, curvedsky as cs 
from orphics import maps,stats,mpi,io
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

sims_ids = np.arange(400) #low for now
args.nsims = len(sims_ids)

args.mask_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.png' 
args.cl_file = "/pscratch/sd/s/sophiapk/lenspyx/lenspyx/data/cls/FFP10_wdipole_lenspotentialCls.dat"
args.output_dir = '/pscratch/sd/s/sophiapk/dustbias-meanfield-maps/my-output/mf_rkalm_v3_400sims/n2048_3deg_apo/'
args.input_dir = '/pscratch/sd/s/sophiapk/dustbias-meanfield-maps/my-input/n2048_3deg_apo/'

output_path = lambda x: os.path.join(args.output_dir, x)
input_path = lambda x: os.path.join(args.input_dir,x)

#mapc3 = hp.read_map(args.c3_map_file) #have to add pol = true and save pol cl
mask = hp.read_map(args.mask_file)

ell,cltt = np.loadtxt(args.cl_file,usecols=(0,1),unpack=True) 
#cltt = hp.anafast(mapc3,lmax = args.lmax_unl, mmax = args.lmax_unl) 

with open(input_path('unl_cltt_meanfield.pkl'),'wb') as f: 
    pickle.dump(cltt,f) 

fsky = np.mean(mask**2)


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

comm,rank,my_tasks = mpi.distribute(args.nsims)
s = stats.Stats(comm)

for sim in my_tasks:

    np.random.seed(sim)

    print(f"Rank {rank} processing sim {sim}")
    gauss_map = hp.sphtfunc.synfast(cltt, nside=args.nside) # randomly generated gaussian dust map
    gauss_map_mask = gauss_map * mask 
    #hp.write_map(output_path(get_gauss_hp_map_name(args, sim,fsky)), gauss_map_mask) 

    # measure and save 2pt
    # Convert the map to alms and compute the power spectrum
    gauss_map_mask_alms = hp.map2alm(gauss_map_mask, lmax=args.lmax_len)
    #foreground_cls = cs.alm2cl(foreground_alms) / w2_masks[fsky]

    hp.write_alm(output_path(get_gauss_alm_name(args, sim=sim, fsky=fsky)), gauss_map_mask_alms)


    









