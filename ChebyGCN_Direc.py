import torch
import torch.nn as nn
import scipy as sp
from scipy.sparse import linalg,identity,spdiags

class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.
    Laplacian is motified for direct-graph

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.FloatTensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graphs):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        mul_L = self.cheb_polynomial(graphs).unsqueeze(1)

        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]
        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [B,N, N].
        :return: the multi order Chebyshev laplacian, [K,B, N, N].
        """
        N = laplacian.size(0)  # [N, N]

        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        # N = laplacian.size(1)  # [N, N]
        # b = laplacian.size(0)
        # multi_order_laplacian = torch.zeros([self.K, b, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        # eye = torch.eye(N, device=laplacian.device, dtype=torch.float).unsqueeze(0)
        # eye_mul = torch.repeat_interleave(eye,b,dim=0) #[B,N,N]
        # multi_order_laplacian[0] = eye_mul


        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.matmul(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]

        return multi_order_laplacian



    # @staticmethod
    # def get_laplacian(graph, lmax, normalize):
    #     """
    #     return the laplacian of the graph.
    #
    #     :param graph: the graph structure without self loop, [N, N].
    #     :param normalize: whether to used the normalized laplacian.
    #     :return: graph laplacian.
    #     """
    #     if normalize:
    #
    #         D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
    #         L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
    #     else:
    #         D = torch.diag(torch.sum(graph, dim=-1))
    #         L = D - graph
    #
    #     if lmax is None:
    #         lmax, _ = linalg(L, 1, which ='LM' ,tol= 1E-2)
    #         lmax = lmax[0]
    #     L = sp.csr_matrix(L)
    #     M, _ = L.shape
    #     I = sp.identity(M, format='csr', dtype= L.dtype)
    #     L = (2 / lmax * L) -I
    #
    #     L = torch.sparse.FloatTensor(torch.LongTensor([L.row.tolist(), L.col.tolist()]),
    #                                  torch.FloatTensor(L.data.astype(L.dtype)))
    #     return L
    #
    # @staticmethod
    # def get_directed_laplacian(graph, lmax, alpha=0.95):
    #
    #     n,m = graph.shape
    #     dangling = sp.where(graph.sun(axis=1)==0)
    #     for d in dangling[0]:
    #         graph[d] = 1.0 /n
    #     graph = graph / graph.sum(axis=1)
    #
    #     P = alpha * graph +(1- alpha) / n
    #     evals,evecs = linalg(P.T , k=1, tol = 1E-2)
    #     v = evecs.flatten().real
    #     p = v/v.sum()
    #     sqrtp = sp.sqrt(p)
    #     I = identity(len(graph))
    #     L = spdiags(sqrtp, [0], n, n) * (I-P) * spdiags(1.0 / sqrtp, [0], n, n)
    #
    #     if lmax is None:
    #         lmax, _ = linalg(L, 1, which ='LM' ,tol= 1E-2)
    #         lmax = lmax[0]
    #     L = sp.csr_matrix(L)
    #     M, _ = L.shape
    #     I = sp.identity(M, format='csr', dtype= L.dtype)
    #     L = (2 / lmax * L) -I
    #
    #     L = torch.sparse.FloatTensor(torch.LongTensor([L.row.tolist(), L.col.tolist()]),
    #                              torch.FloatTensor(L.data.astype(L.dtype)))
    #     return L
