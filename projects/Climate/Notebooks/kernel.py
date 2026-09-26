import numpy as np
# Sobolev kernel
def B1(x):
    return x-1/2

def B2(x):
    return x**2-x+1/6

def sobolev_kernel(x,y):
    return 1+B2(np.abs(x-y))+B2(x+y)

def dirac_kernel(x,y):
    return (x==y).astype('int')