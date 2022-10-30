import torch
import torch.nn as nn
from numpy import linalg, dot, diag, ones


def compute_lse(X,y):
    tmp = torch.matmul(X.T, X)
    tmp = torch.inverse(tmp)
    tmp = torch.matmul(tmp,X.T)
    tmp = torch.matmul(tmp,y)

    return tmp

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize,bias = False)

    def forward(self, x):
        out = self.linear(x)
        return out

    @classmethod
    def least_square_estimate(cls,X,y):
        return compute_lse(X,y)


def ridg_regression_estimate(X, y, lam):
    d = X.shape[1]
    inv = linalg.inv( dot(X.T,X) + lam*diag(ones(d)) )
    return dot( inv ,  dot(X.T,y) )


class PolynomialRegression(nn.Module):
    def __init__(self, dim: int, output_dim: int) -> None:
        super().__init__()
        if dim < 0 or output_dim <0:
            raise RuntimeError(f"You entered invalid dimension:{dim}")
        self.dim = dim
        self.linear = nn.Linear(dim+1,output_dim, bias=False)
    @classmethod
    def pd(cls,x,dim):
        '''
        x: scalar
        return: [1, x, x**2, ...]
        '''
        result = [1]
        for i in range(dim):
            result.append(torch.pow(x,(i+1)))
        return torch.Tensor(result)
    @classmethod
    def build_X(cls,x,dim:int):
        shape = x.shape

        # N
        if len(shape) == 1:
            X = torch.stack([cls.pd(t,dim) for t in x], dim=0)

        # N,1
        elif len(shape) == 2 and shape[1] == 1:
            X = torch.stack([cls.pd(t[0],dim) for t in x], dim=0)
        else:
            raise NotImplementedError('This shape of X is not implemented yet.')
        return X
        

    
    def forward(self,x):
        '''
        x: N or (N,1)
        '''

        # X: N,D+1
        return self.linear(self.build_X(x,self.dim))

    @classmethod
    def least_square_estimate(cls, X, y,dim):
        return compute_lse(cls.build_X(X,dim), y)

    def fit_lse(self,X,y):
        lse = self.least_square_estimate(X,y,self.dim).view((1,-1))
        self.linear.weight = nn.Parameter(lse)





            
        