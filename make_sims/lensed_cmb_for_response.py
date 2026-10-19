import gc
import numpy as np
import healpy as hp
import os
import lenspyx
from lenspyx.utils import camb_clfile
from lenspyx import lensing
from orphics import mpi, stats

#############################################################
# PARAMETERS
#############################################################
nside = 2048
nsims = 100
lmax_len = 3000
dlmax = 1000
lmax_unl = lmax_len + dlmax
mmax_unl = lmax_unl
amp_arcmin = 5.3

#############################################################
# PATHS
#############################################################
cls_path = "/home/spk24f/gitreps/lenspyx/lenspyx/data/cls/"
cl_unl = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))

outdir_base = "/data/data01/sophia/bias-sims/"
setdirs = [os.path.join(outdir_base, f"set{i}") for i in range(4)]
for sd in setdirs:
    os.makedirs(sd, exist_ok=True)

#############################################################
# PRECOMPUTE CONSTANTS
#############################################################
ell = np.arange(lmax_unl + 1)
sqrtfac = np.sqrt(ell * np.arange(1, lmax_unl + 2))
geom_info = ('healpix', {'nside': nside})
geom = lensing.get_geom(geom_info)

#############################################################
# MPI SETUP
#############################################################
comm, rank, my_tasks = mpi.distribute(nsims)
s = stats.Stats(comm)

for sim_idx in my_tasks:
    print(f"[Rank {rank}] Sim {sim_idx}", flush=True)

    ##########################################################
    # SET 0 (S): Unique CMB, Unique phi
    ##########################################################
    np.random.seed(abs(hash((0, sim_idx, 0, 'cmb'))) % (2**32))
    tlm_unl0 = hp.synalm(cl_unl['tt'], lmax=lmax_unl, mmax=mmax_unl)

    np.random.seed(abs(hash((0, sim_idx, 0, 'phi'))) % (2**32))
    plm0 = hp.synalm(cl_unl['pp'], lmax=lmax_unl, mmax=mmax_unl)

    dlm0 = hp.almxfl(plm0, sqrtfac, None, False)
    Tlen0 = lensing.alm2lenmap(tlm_unl0, dlm0, geometry=geom_info, verbose=0)
    del tlm_unl0, dlm0
    tlm_len0 = hp.map2alm(Tlen0, lmax=lmax_len)
    del Tlen0

    hp.write_alm(os.path.join(setdirs[0], f"tlm_len_{sim_idx:04d}.fits"), tlm_len0, overwrite=True)
    hp.write_alm(os.path.join(setdirs[0], f"plm_{sim_idx:04d}.fits"), plm0, overwrite=True)
    del tlm_len0, plm0
    gc.collect()

    ##########################################################
    # SET 1 (S'): Unique CMB, Unique phi
    ##########################################################
    np.random.seed(abs(hash((0, sim_idx, 1, 'cmb'))) % (2**32))
    tlm_unl1 = hp.synalm(cl_unl['tt'], lmax=lmax_unl, mmax=mmax_unl)

    np.random.seed(abs(hash((0, sim_idx, 1, 'phi'))) % (2**32))
    plm1 = hp.synalm(cl_unl['pp'], lmax=lmax_unl, mmax=mmax_unl)

    dlm1 = hp.almxfl(plm1, sqrtfac, None, False)
    Tlen1 = lensing.alm2lenmap(tlm_unl1, dlm1, geometry=geom_info, verbose=0)
    del tlm_unl1, dlm1
    tlm_len1 = hp.map2alm(Tlen1, lmax=lmax_len)
    del Tlen1

    hp.write_alm(os.path.join(setdirs[1], f"tlm_len_{sim_idx:04d}.fits"), tlm_len1, overwrite=True)
    hp.write_alm(os.path.join(setdirs[1], f"plm_{sim_idx:04d}.fits"), plm1, overwrite=True)
    del tlm_len1, plm1
    gc.collect()

    ##########################################################
    # SET 2 (Sphi): Unique CMB, SHARED phi (with set 3)
    ##########################################################
    np.random.seed(abs(hash((0, sim_idx, 2, 'cmb'))) % (2**32))
    tlm_unl2 = hp.synalm(cl_unl['tt'], lmax=lmax_unl, mmax=mmax_unl)

    np.random.seed(abs(hash((0, sim_idx, 2, 'phi'))) % (2**32))
    plm_shared = hp.synalm(cl_unl['pp'], lmax=lmax_unl, mmax=mmax_unl)

    dlm_shared = hp.almxfl(plm_shared, sqrtfac, None, False)
    Tlen2 = lensing.alm2lenmap(tlm_unl2, dlm_shared, geometry=geom_info, verbose=0)
    del tlm_unl2
    tlm_len2 = hp.map2alm(Tlen2, lmax=lmax_len)
    del Tlen2

    hp.write_alm(os.path.join(setdirs[2], f"tlm_len_{sim_idx:04d}.fits"), tlm_len2, overwrite=True)
    hp.write_alm(os.path.join(setdirs[2], f"plm_{sim_idx:04d}.fits"), plm_shared, overwrite=True)
    del tlm_len2
    gc.collect()

    ##########################################################
    # SET 3 (Sphi'): Unique CMB, SHARED phi (with set 2)
    ##########################################################
    np.random.seed(abs(hash((0, sim_idx, 3, 'cmb'))) % (2**32))
    tlm_unl3 = hp.synalm(cl_unl['tt'], lmax=lmax_unl, mmax=mmax_unl)

    # Reuse plm_shared and dlm_shared from set 2
    Tlen3 = lensing.alm2lenmap(tlm_unl3, dlm_shared, geometry=geom_info, verbose=0)
    del tlm_unl3, dlm_shared
    tlm_len3 = hp.map2alm(Tlen3, lmax=lmax_len)
    del Tlen3

    hp.write_alm(os.path.join(setdirs[3], f"tlm_len_{sim_idx:04d}.fits"), tlm_len3, overwrite=True)
    hp.write_alm(os.path.join(setdirs[3], f"plm_{sim_idx:04d}.fits"), plm_shared, overwrite=True)
    del tlm_len3, plm_shared
    gc.collect()

    print(f"[Rank {rank}] Done all sets for sim {sim_idx}", flush=True)

comm.Barrier()
print(f"[Rank {rank}] Finished all sims.", flush=True)