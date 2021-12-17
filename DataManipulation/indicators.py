import pandas as pd
import numpy as np
# Numerical 1st Derivative
def D1(TS):
    if isinstance(TS,pd.Series):
        D1 = np.zeros(len(TS))
        D1[0] = np.nan
        D1[1:] = TS[1:].to_numpy()-TS[:-1].to_numpy()
        D1 = pd.Series(D1,index = TS.index)
    elif isinstance(TS,np.ndarray):
        D1 = np.zeros(len(TS))
        D1[0] = np.nan
        D1[1:] = TS[1:]-TS[:-1]
    return D1
# Numerical 2nd Derivative
def D2(TS):
    if isinstance(TS,pd.Series):
        D2 = np.zeros(len(TS))
        npTS = TS.to_numpy()
        D2[:2] = np.nan
        D2[2:] = npTS[2:]-2*npTS[1:-1]+npTS[:-2]
        D2 = pd.Series(D2,index = TS.index)
    elif isinstance(TS,np.ndarray):
        D2 = np.zeros(len(TS))
        D2[:2] = np.nan
        D2[2:] = TS[2:]-2*TS[1:-1]+TS[:-2]
    return D2

from abc import ABC
class ForwardPredictIndicator(ABC):
    def __init__(self):
        pass

class NStepForwardPredictByD12(ForwardPredictIndicator):
    """[summary]
    Usage: NStepForwardPredictByD12(Nsteps:int).look(trendtopredict,trend's D1, trend's D2)
    """
    def __init__(self,Nsteps):
        self.Nsteps = Nsteps
    def look(self,TS,d1=None,d2=None):
        if d1 == None:
            d1 = D1(TS)
        if d2 == None:
            d2 = D2(TS)
        store = np.zeros(shape=(len(TS),self.Nsteps))
        for step in range(1,self.Nsteps+1):
            store[:,step-1] = step**2*d2/2 + step*d1 + TS
        return store

#import numba
class NStepForwardPredictByD1(ForwardPredictIndicator):
    """[summary]
    Usage: NStepForwardPredictByD12(Nsteps:int).look(trendtopredict,trend's D1, trend's D2)
    """
    def __init__(self,Nsteps):
        self.Nsteps = Nsteps
    #@numba.jit(nopython=True)
    def look(self,TS,d1=None):
        if d1 is None:
            d1 = D1(TS)
        store = np.zeros(shape=(len(TS),self.Nsteps))
        for step in range(1,self.Nsteps+1):
            store[:,step-1] = step*d1 + TS
        return store
def D1Crossing(TS1,TS2, TS1D1=None,TS2D1=None,n = 5):
    """
    checks whether the D1 projection (straight line) of two signals cross within n steps
    i.e. assume y = mx+b, b= y[x=0]=TS, m=x[-1]-x[-2], check that the x[cross]<n and y1(x)=y2(x)
    """
    if TS1D1 is None:
        TS1D1 = D1(TS1)
    if TS2D1 is None:
        TS2D1 = D1(TS2)
    result = np.zeros(len(TS1D1))
    #result.fill(np.nan)
    inter = np.true_divide(TS1-TS2,TS2D1-TS1D1) 
    
    result = (inter<=n) & (inter > 0)
    return result