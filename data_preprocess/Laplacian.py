import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import identity, spdiags, linalg

import torch
def caslaplacian_matrix(G, nodelist=None, weight='weight',alpha=0.95):
    import scipy as sp
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)
    n, m = M.shape
    if not (0 < alpha < 1):
            raise nx.NetworkXError('alpha must be between 0 and 1')
    # this is using a dense representation
    M = M.todense()
    # add constant to dangling nodes' row
    dangling = sp.where(M.sum(axis=1) == 0) #出度为0
    for d in dangling[0]:
        M[d] = 1.0 / n
    # normalize
    M = M / M.sum(axis=1)

    P = alpha * M + (1 - alpha) / n
    evals, evecs = linalg.eigs(P.T, k=1,tol=1E-2)
    v = evecs.flatten().real
    p = v / v.sum()
    sqrtp = sp.sqrt(p)
    I = sp.identity(len(G))
    Q = spdiags(sqrtp, [0], n, n) * (I-P) * spdiags(1.0 / sqrtp, [0], n, n)
    return Q


def laplacian_matrix(graph,lmax=2, normalize=True):
    if normalize:
        D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
        L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
    else:
        D = torch.diag(torch.sum(graph, dim=-1))
        L = D - graph

    if lmax is None:
        lmax, _ = linalg(L, 1, which='LM', tol=1E-2)
        lmax = lmax[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lmax * L) - I

    L = torch.sparse.FloatTensor(torch.LongTensor([L.row.tolist(), L.col.tolist()]),
                                 torch.FloatTensor(L.data.astype(L.dtype)))
    return L


def calculate_scaled_laplacian_dir(graph, kind_of_laplacin = None, lambda_max=2):
    if kind_of_laplacin == 'caslaplacian':
        L = caslaplacian_matrix(graph)
    else:
        L = laplacian_matrix(graph)

    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM', tol=1E-2)
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L