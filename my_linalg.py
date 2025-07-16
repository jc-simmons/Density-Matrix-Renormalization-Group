import numpy as np

def custom_dot(A, v, n, mpi_ctx = None):
    """
    Compute dot product A @ v in parallel if mpi_ctx is provided.
    Requires appropriate mpi_ctx methods.
    """
    if mpi_ctx is None:
        return np.dot(A, v)

    #  number of rows assigned to rank r
    counts = mpi_ctx.get_counts(n)
    # starting index for each rank
    displs = mpi_ctx.get_displacements(n)

    start_idx, end_idx = displs[mpi_ctx.rank], displs[mpi_ctx.rank]+counts[mpi_ctx.rank]

    z_rank = np.dot(A[start_idx:end_idx, :], v)
    z = np.empty(n, dtype=A.dtype)

    mpi_ctx.gather(z_rank, z, n, cast=True)

    return z


def lanczos(A, tol = 1e-8, max_iter = None, mpi_ctx = None, seed=0):
    en_prev = 0.0
    en_next = 1.0
    en_low = 0.0

    n = A.shape[0]
    V = np.zeros((n, n+1))
    v_start = np.random.rand(n)
    v_start /= np.linalg.norm(v_start)
    V[:, 0] = v_start.copy()

    V_i = V[:, 0].copy()
    z = np.zeros(n)
    T = np.zeros((n, n))
    alpha = np.zeros(n+1)
    beta = np.zeros(n+1)
    beta[0] = 0
    super_groundstate = np.zeros(n)

    i = 0

    while (abs(en_next - en_prev) > tol or abs(en_low - en_next) > tol):

        en_prev = en_next

        z = custom_dot(A, V_i, n, mpi_ctx=mpi_ctx)

        alpha[i] = np.dot(V[:, i], z)

        z = z - alpha[i] * V[:, i]
        if i > 0:
            z = z - beta[i] * V[:, i - 1]

        beta[i + 1] = np.linalg.norm(z)
        if beta[i + 1] < 1e-14:  
            break
        V[:, i + 1] = z / beta[i + 1]

        T[i, i] = alpha[i]
        if i > 0:
            T[i - 1, i] = beta[i]
            T[i, i - 1] = beta[i]

        V_i = V[:, i + 1].copy()

        vals, vecs = np.linalg.eig(T[:i+1, :i+1])

        en_next = vals[0]
        T_groundstate = vecs[:, 0]
        super_groundstate = np.dot(V[:, :i+1], T_groundstate)

        if en_low > en_next:
            en_low = en_next

        i += 1

        if max_iter is not None and i >= max_iter:
            break

    return en_next, super_groundstate
