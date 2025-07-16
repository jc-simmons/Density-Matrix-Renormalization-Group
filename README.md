## Density Matrix Renormalization Group Applied to a Heisenberg Spin Chain

This repository contains a refactored version of a High-Performance Computational Physics project that implements a basic **Density Matrix Renormalization Group (DMRG)** algorithm applied to a **ferromagnetic Heisenberg spin-½ chain**.

Further details can be found in the accompanying report (report.pdf).

---

### Optional: MPI Support

Parallel execution is optionally supported via `mpi4py`. However, because MPI requires system-level installation that varies by platform, detailed MPI installation instructions are not included here. This project focuses on demonstrating the core logic and structure of the algorithm.

The code gracefully falls back to non-MPI execution if MPI is unavailable, allowing it to run out-of-the-box without additional setup.

Because installation of `mpi4py` will fail without an MPI compiler, it is **omitted from `requirements.txt`**. If you wish to enable parallelism provided a working MPI implementation, then:

pip install mpi4py==3.1.4  # optional modern version
