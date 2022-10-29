import torch
from numpy import linalg, dot, diag, ones

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize,bias = False)

    def forward(self, x):
        out = self.linear(x)
        return out

def ridg_regression_estimate(X, y, lam):
    d = X.shape[1]
    inv = linalg.inv( dot(X.T,X) + lam*diag(ones(d)) )
    return dot( inv ,  dot(X.T,y) )