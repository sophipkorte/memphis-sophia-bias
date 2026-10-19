import os
import argparse
import numpy as np
import healpy as hp
from pixell import enmap, curvedsky as cs
from orphics import maps, stats, mpi, io
import pickle
import pysm3
import pysm3.units as u
import gc

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import MPI_error_handler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

args = argparse.Namespace()

""" Generate Gaussian realizations of pysm foregrounds on SO frequency channels """

args.lmax_len = 3000
dlmax = 1000
args.lmax_unl, args.mmax_unl = args.lmax_len + dlmax, args.lmax_len + dlmax
args.nside = 2048

sims_ids = np.arange(100)
args.nsims = len(sims_ids)

args.mask_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits'
args.mask_gal_file = '/home/spk24f/masks/mask_binary_gal_cut_n2048_3deg_apo.fits'
args.output_dir = '/data/data01/sophia/fg-bias-sims/'
output_path = lambda x: os.path.join(args.output_dir, x)

def get_output_dir(base_dir, fg_type, extra_mask):
    if extra_mask:
        mask_tag = 'extra_mask'
    else:
        mask_tag = 'lat_mask'
    d = os.path.join(base_dir, fg_type, mask_tag)
    os.makedirs(d, exist_ok=True)
    return d

SO_FREQ = [93, 145]

pysm_str = {'d9': ['d9'],
            'd10': ['d10'],
            'd12': ['d12']}

"""
            'low': ['d9', 's4'],
            'medium': ['d10', 's5'],
            'high': ['d12', 's7']"""

def parse_args(existing_args):
    parser = argparse.ArgumentParser(description="Apply extra mask for galactic foregrounds")
    parser.add_argument("--extra_mask", action="store_true", help='mask galactic plane')
    return parser.parse_args(namespace=existing_args)

args = parse_args(args)

###########################################################
# FUNCTIONS
###########################################################

def smooth_cls(cl, points=300):
    """bin and interpolate a cl to smooth it"""
    bin_edges = np.linspace(2, len(cl), points).astype(int)
    cents, cls = bandedcls(cl, bin_edges)
    cls = maps.interp(cents, cls, fill_value="extrapolate")(np.arange(len(cl)))
    return cls

def bandedcls(cl, _bin_edges):
    ls = np.arange(cl.size)
    binner = stats.bin1D(_bin_edges)
    cents, bls = binner.bin(ls, cl)
    return cents, bls

def get_sky_model(args, str_fg):
    '''
        args.nside: 2048
        args.freq: frequency in GHz
    '''
    sky_dust = pysm3.Sky(nside=args.nside, preset_strings=list(str_fg))
    map_freqGHz = sky_dust.get_emission(args.freq * u.GHz)
    sky_hp_maps = map_freqGHz.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(args.freq * u.GHz))
    return sky_hp_maps

def get_gauss_fg_alm_name(freq, fg_type, sim, fsky=None):
    tag = f'{freq}_{fg_type}_{sim}'
    if fsky is not None:
        tag = f'{tag}_{fsky:.2f}'
    return f'gauss_fg_alm_{tag}.fits'

def get_fg_cls_name(freq, fg_type):
    return f'fg_cl_{freq}_{fg_type}.pkl'

###########################################################
# MASK INFO
#RANK 0 ONLY
###########################################################

if rank == 0:
    print("Rank 0: setting up masks and measuring foreground spectra...")

    mask = hp.read_map(args.mask_file)
    mask_gal = hp.read_map(args.mask_gal_file)
    mask2 = mask * mask_gal

    if args.extra_mask:
        npix = mask2.size
        pmap = 4 * np.pi / npix
        w2 = np.sum((mask2**2) * pmap) / np.pi / 4.
        w4 = np.sum((mask2**4) * pmap) / np.pi / 4.
        use_mask = mask2
        end = '_extra_mask'
    else:
        npix = mask.size
        pmap = 4 * np.pi / npix
        w2 = np.sum((mask**2) * pmap) / np.pi / 4.
        w4 = np.sum((mask**4) * pmap) / np.pi / 4.
        use_mask = mask
        end = '_lat_mask'

    print(f"fsky: {w2}") 

    fg_cls_dict = {}  # keyed by (freq, fg_type)

    for freq in SO_FREQ:
        args.freq = freq
        for fg_type, fg_components in pysm_str.items():
            key = f'{freq}_{fg_type}'
            fg_dir = get_output_dir(args.output_dir, fg_type, args.extra_mask)
            cls_file = os.path.join(fg_dir, get_fg_cls_name(freq, fg_type))

            if os.path.exists(cls_file):
                print(f"Loading existing fg cls for {key}")
                with open(cls_file, 'rb') as f:
                    fg_cls_dict[key] = pickle.load(f)
            else:
                print(f"Measuring fg cls for {key}")
                hp_map = get_sky_model(args, fg_components)

                # TT only — index 0 is temperature
                fg_t_map = np.array(hp_map[0]) 
                raw_cl= hp.anafast(fg_t_map, lmax=args.lmax_len)
        
                fg_cls_dict[key] = raw_cl

                with open(cls_file, 'wb') as f:
                    pickle.dump(raw_cl, f)

                del hp_map, fg_t_map
                gc.collect()

    print("All foreground cls measured/loaded.")

else:
    fg_cls_dict = None
    use_mask = None
    w2 = None

###########################################################
# BROADCAST: rank 0 sends data to all ranks
###########################################################

fg_cls_dict = comm.bcast(fg_cls_dict, root=0)
use_mask = comm.bcast(use_mask, root=0)
w2 = comm.bcast(w2, root=0)

comm.Barrier()

if rank == 0:
    print("All ranks received data. Starting sim generation.")


###########################################################
# STEP 2: Generate Gaussian realizations from measured cls
###########################################################

_, _, my_tasks = mpi.distribute(args.nsims)

for freq in SO_FREQ:
    for fg_idx, fg_type in enumerate(pysm_str.keys()):
        key = f'{freq}_{fg_type}'
        cl_fg = fg_cls_dict[key]
        fg_dir = get_output_dir(args.output_dir, fg_type, args.extra_mask)

        print(f"Rank {rank}: generating sims for {key}")

        for sim in my_tasks:
            np.random.seed(sim + freq * 10000 + fg_idx * 100000+3)

            print(f"  Rank {rank} sim {sim} for {key}")

            # Generate full-sky Gaussian realization from the measured fg spectrum
            gauss_fg_map = hp.synfast(cl_fg, nside=args.nside, lmax=args.lmax_len, new=True)

            """# Mask it
            gauss_fg_map_masked = gauss_fg_map * use_mask""" #no more masking

            # Get alms of the masked map
            gauss_fg_alm = hp.map2alm(gauss_fg_map, lmax=args.lmax_len)

            hp.write_alm(
                os.path.join(fg_dir,get_gauss_fg_alm_name(freq, fg_type, sim)),
                gauss_fg_alm,
                overwrite=True
            )

            del gauss_fg_map, gauss_fg_alm
            gc.collect()

print(f"Rank {rank} done.")

comm.Barrier()

if rank == 0:
    print(f"All done with{end}.")