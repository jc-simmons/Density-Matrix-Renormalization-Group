#!/usr/bin/env python3

import numpy as np
from scipy.sparse.linalg import eigsh
from numpy import kron, identity
import timeit

from my_linalg import lanczos
from mpi_context import setup_mpi_context

# Spin-½ site operators
sz = np.array([[0.5, 0], [0, -0.5]], dtype='d')
sp = np.array([[0, 1], [0, 0]], dtype='d')

site = {"Sz": sz, "Sp": sp}

# Initial Hamiltonian for 1-site block
initial_H = np.zeros((2, 2), dtype='d')


def interaction(block1, block2):
    """Returns two-site Heisenberg interaction Hamiltonian."""
    return 0.5 * (kron(block1["Sp"], block2["Sp"].conj().T) + 
                  kron(block1["Sp"].conj().T, block2["Sp"])) + \
           kron(block1["Sz"], block2["Sz"])

def enlarge(block):
    """Adds a site to the block and updates operators."""
    n = block["H"].shape[0]
    new_block = {
        "H": kron(block["H"], identity(2)) + interaction(block, site),
        "Sp": kron(identity(n), sp),
        "Sz": kron(identity(n), sz)
    }
    return new_block

def dmrg_step(sys_block, env_block, sys_size, trunc_limit, iflag, mpi_ctx=None):
    """Performs a single DMRG step: grow system/env, build superblock, truncate."""
    sys_block = enlarge(sys_block)
    env_block = enlarge(env_block)

    sys_size += 1
    sys_dim = sys_block["H"].shape[0]
    env_dim = env_block["H"].shape[0]

    superH = kron(sys_block["H"], identity(env_dim)) + \
             kron(identity(sys_dim), env_block["H"]) + \
             interaction(sys_block, env_block)

    if iflag == 0 or iflag == 1:
        energy, super_groundstate = lanczos(superH, mpi_ctx=mpi_ctx)
    elif iflag == 2:
        energies, states = np.linalg.eigh(superH)
        energy = energies[0]
        super_groundstate = states[:, 0]
    elif iflag == 3:
        energy, super_groundstate = eigsh(superH, k=1, which="SA")
        energy = energy[0]
    else:
        raise ValueError(f"Unsupported iflag: {iflag}")

    psi = np.reshape(super_groundstate, (sys_dim, env_dim))
    rho = psi @ psi.conj().T
    evals, evecs = np.linalg.eigh(rho)
    trunc_states = np.flip(evecs[:, -trunc_limit:], axis=1)

    for key in sys_block:
        sys_block[key] = trunc_states.conj().T @ sys_block[key] @ trunc_states

    return sys_block, energy, sys_size, trunc_states.shape[1]

def exact_ground(L):
    """Diagonalizes full system for comparison."""
    block = {"H": initial_H, "Sz": sz, "Sp": sp}
    for _ in range(L - 1):
        block = enlarge(block)
    vals, _ = eigsh(block["H"], k=1, which="SA")
    print('Groundstate energy:', vals[0] / L)

def infinite(block, chain_len, trunc_limit, iflag, mpi_ctx=None):
    """Infinite-system DMRG algorithm."""
    sys_size = 1
    while 2 * sys_size < chain_len:
        block1 = dict(block)
        block2 = dict(block)
        block, energy, sys_size, _ = dmrg_step(block1, block2, sys_size, trunc_limit, iflag, mpi_ctx)

    if iflag == 1 and mpi_ctx and mpi_ctx.rank != 0:
        return
    print('Groundstate energy:', energy / chain_len)

def finite(block, chain_len, trunc_limit, iflag, mpi_ctx=None):
    """Finite-system DMRG sweeps."""
    sys_size = 1
    blocks = {(0, sys_size): dict(block), (1, sys_size): dict(block)}

    while 2 * sys_size < chain_len:
        block1 = dict(block)
        block2 = dict(block)
        block, energy, sys_size, _ = dmrg_step(block1, block2, sys_size, trunc_limit, iflag, mpi_ctx)
        blocks[(0, sys_size)] = dict(block)
        blocks[(1, sys_size)] = dict(block)

    tol = 1e-8
    epsilon_p = 0.0
    epsilon_q = energy / chain_len

    nsweep = 1
    i, j = 0, 1

    while abs(epsilon_q - epsilon_p) > tol:
        if chain_len - sys_size - 2 > 1:
            epsilon_p = epsilon_q
            env_block = dict(blocks[(j, chain_len - sys_size - 2)])
            block, energy, sys_size, _ = dmrg_step(block, env_block, sys_size, trunc_limit, iflag, mpi_ctx)
            blocks[(i, sys_size)] = dict(block)
            epsilon_q = energy / chain_len
        else:
            nsweep += 1
            print('---start sweep---', nsweep)
            sys_size = 1
            block = dict(blocks[(j, sys_size)])
            i, j = j, i  # swap roles

    if iflag == 1 and mpi_ctx and mpi_ctx.rank != 0:
        return
    print('Groundstate energy:', energy / chain_len)



def main():
    iflag = 3            # 0: Lanczos, 1: Lanczos MPI, 2: eigh, 3: eigsh
    chain_len = 22
    trunc_limit = 15
    run_finite = False

    mpi_ctx = setup_mpi_context(iflag == 1)
    if mpi_ctx:
        rank = mpi_ctx.rank
    else:
        rank = 0

    block = {
        "H": initial_H,
        "Sz": sz,
        "Sp": sp
    }

    start = timeit.default_timer()
    if run_finite:
        finite(block, chain_len, trunc_limit, iflag, mpi_ctx)
    else:
        infinite(block, chain_len, trunc_limit, iflag, mpi_ctx)
    stop = timeit.default_timer()

    if iflag != 1 or rank == 0:
        print("Time elapsed:", stop - start)

if __name__ == "__main__":
    main()
