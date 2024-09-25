import networkx as nx


class InverseLaplacian:

    def __init__(self, L, width=None, dtype=None):
        global np
        import numpy as np
        n, n = L.shape
        self.dtype = dtype
        self.n = n
        if width is None:
            self.w = self.width(L)
        else:
            self.w = width
        self.C = np.zeros((self.w, n), dtype=dtype)
        self.L1 = L[1:, 1:]
        self.init_solver(L)

    def width(self, L):
        """Compute the width of the Laplacian matrix."""
        return min(max(20, L.shape[0] // 10), 100)

    def init_solver(self, L):
        """Initialize the solver."""
        pass

    def solve(self, r):
        """Solve the linear system."""
        raise NotImplementedError("Subclasses must implement this method")


class FullInverseLaplacian(InverseLaplacian):
    def init_solver(self, L):
        """Initialize the solver by computing the full inverse."""
        self.IL1 = np.linalg.inv(self.L1)

    def solve(self, r):
        """Solve the linear system using the full inverse."""
        return self.IL1 @ r[1:]


class SuperLUInverseLaplacian(InverseLaplacian):
    def init_solver(self, L):
        """Initialize the SuperLU solver."""
        from scipy.sparse.linalg import splu
        self.LU = splu(self.L1.tocsc(), permc_spec='MMD_AT_PLUS_A')

    def solve(self, r):
        """Solve the linear system using SuperLU."""
        return self.LU.solve(r[1:])


class CGInverseLaplacian(InverseLaplacian):
    def init_solver(self, L):
        """Initialize the Conjugate Gradient solver."""
        from scipy.sparse.linalg import cg
        self.cg_solver = cg

    def solve(self, r):
        """Solve the linear system using Conjugate Gradient method."""
        x, info = self.cg_solver(self.L1, r[1:])
        if info != 0:
            raise nx.NetworkXError("Conjugate Gradient method failed to converge")
        return x
