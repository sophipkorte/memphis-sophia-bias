import numpy as np
import healpy as hp
import os
from lenspyx import remapping

import os
import lenspyx
import pickle
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl
from lenspyx import lensing 
from orphics import mpi,stats

nside = 2048
nsims = 100

lmax_len = 3000 # desired lmax of the lensed field. #!! careful here, lmax of ilc is 799...
dlmax = 1000 # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
epsilon = 1e-6 # target accuracy of the output maps (execution time has a fairly weak dependence on this)
lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax


# Paths for Cls

cls_path = cls_path = "/home/spk24f/gitreps/lenspyx/lenspyx/data/cls/"
cl_gradlens = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_gradlensedCls.dat')) 
cl_unl = camb_clfile(os.path.join(cls_path,'FFP10_wdipole_lenspotentialCls.dat'))
cl_len = camb_clfile(os.path.join(cls_path,'FFP10_wdipole_lensedCls.dat'))

ells = np.arange(lmax_unl+1)
cltt = cl_unl["tt"]
clee = cl_unl["ee"]
clbb = cl_unl["bb"]
clte = cl_unl["te"]
clpp = cl_unl["pp"]

"""
cl_unl = np.zeros((4, lmax_len + 1))
cl_unl[0, 2:len(cltt)+2] = cltt[2:]
cl_unl[1, 2:len(clee)+2] = clee[2:]
cl_unl[2, 2:len(clbb)+2] = clbb[2:]
cl_unl[3, 2:len(clte)+2] = clte[2:]
clpp_full = np.zeros(lmax_len + 1)
clpp_full[2:len(clpp)+2] = clpp[2:]"""

splitdirs = [f"sims/set{i}" for i in range(4)]
for sd in splitdirs:
    os.makedirs(sd, exist_ok=True)

    # ---- MPI Setup ----
comm,rank,my_tasks = mpi.distribute(nsims)
s = stats.Stats(comm)

for sim_idx in my_tasks:


  # --- S (set0): Unique CMB, Unique phi
    seed_cmb0 = (0, sim_idx, 0)
    seed_phi0 = (1, sim_idx, 0)
    alm_unl0 = hp.synalm((cltt,clte,clee,clbb), lmax=lmax_unl, mmax=mmax_unl)
    plm0 = synalm(clpp, lmax=lmax_unl, mmax=mmax_unl)
    ell = np.arange(lmax_unl + 1)
    sqrtfac = np.sqrt(ell * np.arange(1, lmax_unl + 2))
    dlm0 = almxfl(plm0, sqrtfac, None, False)
    geom_info = ('healpix', {'nside': nside})
    Tlen0 = lensing.alm2lenmap(alm_unl0[0], dlm0, geometry=geom_info)
    Qlen0, Ulen0 = lenspyx.lensing.alm2lenmap_spin(alm_unl0[1], dlm0, 2, geometry=geom_info, verbose=1)
    tlm_len0 = hp.map2alm(Tlen0, lmax=lmax_len)
    geom = lenspyx.lensing.get_geom(geom_info)
    elm_len0, blm_len0 = geom.map2alm_spin([Qlen0, Ulen0], 2, lmax_len, lmax_len, nthreads=os.cpu_count())
    hp.write_alm(os.path.join(splitdirs[0], f"alm_len_{sim_idx:04d}.fits"), np.array([tlm_len0, elm_len0, blm_len0]), overwrite=True)


     # --- S' (set1): Unique CMB, Unique phi
    seed_cmb1 = (0, sim_idx, 1)
    seed_phi1 = (1, sim_idx, 1)
    alm_unl1 = hp.synalm((cltt,clte,clee,clbb), lmax=lmax_unl, mmax=mmax_unl)
    plm1 = synalm(clpp, lmax=lmax_unl, mmax=mmax_unl)
    dlm1 = almxfl(plm1, sqrtfac, None, False)
    Tlen1 = lensing.alm2lenmap(alm_unl1[0], dlm1, geometry=geom_info)
    Qlen1, Ulen1 = lenspyx.lensing.alm2lenmap_spin(alm_unl1[1], dlm1, 2, geometry=geom_info, verbose=1)
    tlm_len1 = hp.map2alm(Tlen1, lmax=lmax_len)
    elm_len1, blm_len1 = geom.map2alm_spin([Qlen1, Ulen1], 2, lmax_len, lmax_len, nthreads=os.cpu_count())
    hp.write_alm(os.path.join(splitdirs[1], f"alm_len_{sim_idx:04d}.fits"), np.array([tlm_len1, elm_len1, blm_len1]), overwrite=True)

 # --- S_phi (set2): Unique CMB, shared phi (with set3)
    seed_cmb2 = (0, sim_idx, 2)
    seed_phi_shared = (1, sim_idx, 2)
    alm_unl2 = hp.synalm((cltt,clte,clee,clbb), lmax=lmax_unl, mmax=mmax_unl)
    elm_unl2 = alm_unl2[1]  # Extract E-mode alm
    plm_shared = synalm(clpp, lmax=lmax_unl, mmax=mmax_unl)
    dlm_shared = almxfl(plm_shared, sqrtfac, None, False)
    Tlen2 = lensing.alm2lenmap(alm_unl2[0], dlm_shared, geometry=geom_info)
    Qlen2, Ulen2 = lenspyx.lensing.alm2lenmap_spin(alm_unl2[1], dlm_shared, 2, geometry=geom_info, verbose=1)
    tlm_len2 = hp.map2alm(Tlen2, lmax=lmax_len)
    elm_len2, blm_len2 = geom.map2alm_spin([Qlen2, Ulen2], 2, lmax_len, lmax_len, nthreads=os.cpu_count())
    hp.write_alm(os.path.join(splitdirs[2], f"alm_len_{sim_idx:04d}.fits"), np.array([tlm_len2, elm_len2, blm_len2]), overwrite=True)


  # --- S_phi' (set3): Unique CMB, shared phi (with set2)
    seed_cmb3 = (0, sim_idx, 3)
    alm_unl3 = hp.synalm((cltt,clte,clee,clbb), lmax=lmax_unl, mmax=mmax_unl)
    # Use plm_shared, dlm_shared here as well!
    Tlen3 = lensing.alm2lenmap(alm_unl3[0], dlm_shared, geometry=geom_info)
    Qlen3, Ulen3 = lenspyx.lensing.alm2lenmap_spin(alm_unl3[1], dlm_shared, 2, geometry=geom_info, verbose=1)
    tlm_len3 = hp.map2alm(Tlen3, lmax=lmax_len)
    elm_len3, blm_len3 = geom.map2alm_spin([Qlen3, Ulen3], 2, lmax_len, lmax_len, nthreads=os.cpu_count())
    hp.write_alm(os.path.join(splitdirs[3], f"alm_len_{sim_idx:04d}.fits"), np.array([tlm_len3, elm_len3, blm_len3]), overwrite=True)

    print(f"[Rank {rank}] Done all splits for sim {sim_idx}")

comm.Barrier()
print(f"[Rank {rank}] Finished all sims.")


