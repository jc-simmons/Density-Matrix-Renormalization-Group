import numpy as np

def setup_mpi_context(if_available=True):
    if not if_available:
        return None
    try:
        from mpi4py import MPI
        return MPIContext(MPI)
    except ImportError:
        return None

class MPIContext:
    def __init__(self, MPI):
        self.MPI = MPI 
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.numpy_to_mpi = {
        np.dtype('float64'): self.MPI.DOUBLE,
        np.dtype('float32'): self.MPI.FLOAT,
        np.dtype('int32'): self.MPI.INT,
        np.dtype('int64'): self.MPI.LONG
    }
    
    def get_counts(self, n: int):
        counts = [(n // self.size) + (1 if r < n % self.size else 0) for r in range(self.size)]
        return counts
    
    def get_displacements(self, n: int):
        displs = [0] * self.size
        counts = self.get_counts(n)
        for i in range(1, self.size):
            displs[i] = displs[i-1] + counts[i-1]
        return displs
    
    def gather(self, local_chunk, global_array, n, cast = True):
        counts =  self.get_counts(n)
        displs = self.get_displacements(n)
        mpi_dtype = self.numpy_to_mpi.get(np.dtype(local_chunk.dtype), self.MPI.BYTE)

        self.comm.Gatherv(local_chunk, [global_array, counts, displs, mpi_dtype], root=0)
        if cast:
            self.comm.Bcast(global_array, root=0)