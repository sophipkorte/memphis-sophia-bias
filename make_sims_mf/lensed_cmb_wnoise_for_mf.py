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

########################################
#PARAMETERS FOR LENSED CMB WITH NOISE SIMS
########################################

args.lmax_len = 3000 # desired lmax of the lensed field. #!! careful here, lmax of ilc is 799...
dlmax = 1000  # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
epsilon = 1e-6 # target accuracy of the output maps (execution time has a fairly weak dependence on this)
args.lmax_unl, args.mmax_unl = args.lmax_len + dlmax, args.lmax_len + dlmax
args.nside = 2048

noise_T_ukarcmin = 5.3
beam_fwhm = 2.2  # arcmin

sims_ids = np.arange(200) #low for now
args.nsims = len(sims_ids)

######################################################
#FILE PATHS FOR MASK, CL, AND OUTPUT DIRECTORY
args.mask_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits' 
args.cl_file = "/home/spk24f/gitreps/lenspyx/lenspyx/data/cls/FFP10_wdipole_lensedCls.dat"
args.output_dir = '/data/data01/sophia/meanfield-maps/lensing_wnoise/'
#############################################################

##########################################################
#CREATE OUTPUT DIRECTORY IF IT DOESN'T EXIST
#########################################################
output_path = lambda x: os.path.join(args.output_dir, x)


####################################################
#LOAD MASK AND COMPUTE FSKY, LOAD CLTT
#########################################################
mask = hp.read_map(args.mask_file)
fsky = np.mean(mask**2)

ell, cltt_raw = np.loadtxt(args.cl_file, usecols=(0,1), unpack=True)
ell = ell.astype(int)
cltt = np.zeros(len(cltt_raw) + 1)
cltt[ell[0]:ell[0]+len(cltt_raw)] = cltt_raw

########################################################
#DEFINE FUNCTIONS TO GET GAUSSIAN MAP AND ALM NAMES
######################################################

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

def get_white_noise(beam_fwhm, noise_t, noise_p=None, lmax=3000):
    ls = np.arange(lmax + 1)
    if noise_p is None:
        noise_p = np.sqrt(2.) * noise_t
    nells = {}
    nells['TT'] = (noise_t * np.pi / 180. / 60.)**2. / maps.gauss_beam(beam_fwhm, ls)**2.
    nells['EE'] = (noise_p * np.pi / 180. / 60.)**2. / maps.gauss_beam(beam_fwhm, ls)**2.
    nells['BB'] = (noise_p * np.pi / 180. / 60.)**2. / maps.gauss_beam(beam_fwhm, ls)**2.
    return nells

noise_dict = get_white_noise(beam_fwhm, noise_T_ukarcmin, lmax=args.lmax_len)
cl_total_TT = cltt[:args.lmax_len+1] + noise_dict['TT']

comm,rank,my_tasks = mpi.distribute(args.nsims)
s = stats.Stats(comm)

for sim in my_tasks:

    np.random.seed(sim)

    print(f"Rank {rank} processing sim {sim}")
    gauss_map = hp.sphtfunc.synfast(cl_total_TT, nside=args.nside) # randomly generated gaussian dust map
    gauss_map_mask = gauss_map * mask 
    #hp.write_map(output_path(get_gauss_hp_map_name(args, sim,fsky)), gauss_map_mask) 

    # measure and save 2pt
    # Convert the map to alms and compute the power spectrum
    gauss_map_mask_alms = hp.map2alm(gauss_map_mask, lmax=args.lmax_len)
    #foreground_cls = cs.alm2cl(foreground_alms) / w2_masks[fsky]

    hp.write_alm(output_path(get_gauss_alm_name(args, sim=sim, fsky=fsky)), gauss_map_mask_alms)


    









