import pandas as pd
import numpy as np
import numba
from talib import DEMA

def STDNorm(TS,period):
    pass
def RollTStat(TS,period):
    if isinstance(TS,pd.Series):
        normed = np.zeros(len(TS))
        normed[0] = np.nan
        #  X' = X/X.std(-n-1~-1)
        TSn = TS.to_numpy()
        std = TS.rolling(period).std().to_numpy()
        normed[1:] = (TSn[1:]/std[:-1])
        return pd.Series(normed,index =TS.index)
    else:
        raise Exception("not supporting other types yet, e.g. np.array")

def MaxMinRollScaled(TS,period):
    if isinstance(TS,pd.Series):
        scaled = np.zeros(len(TS))
        scaled[0] = np.nan
        #  X' = (X)/ (max_X(-p,0)-min_X(-p,0)) 
        TSn = TS.to_numpy()
        tsmin = TS.rolling(period).min().to_numpy()
        tsmax = TS.rolling(period).max().to_numpy()
        scaled = (TSn[:])/(tsmax[:]-tsmin[:])
        return pd.Series(scaled,index =TS.index)
def MaxMinRollNorm(TS,period):
    if isinstance(TS,pd.Series):
        normed = np.zeros(len(TS))
        normed[0] = np.nan
        #  X' = (X - min_X(-p-1,-1))/ (max_X(-p-1,-1)-min_X(-p-1,-1)) 
        TSn = TS.to_numpy()
        tsmin = TS.rolling(period).min().to_numpy()
        tsmax = TS.rolling(period).max().to_numpy()
        normed[1:] = (TSn[1:]-tsmin[:-1])/(tsmax[:-1]-tsmin[:-1])
        return pd.Series(normed,index =TS.index)
    else:
        raise Exception("not supporting other types yet, e.g. np.array")
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

@numba.jit(nopython = True)
def NUMBAEMA(series,period):
    myema = np.zeros(len(series))#np.empty(len(series),dtype="float64")
    #myema[:] = np.nan
    myema.fill(np.nan)
    smoothing = 2
    nan_idx = np.where(np.isnan(series))[0]
    lastnan_idx = -1
    if len(nan_idx)!=0:
        lastnan_idx = nan_idx[-1]
    start_idx = lastnan_idx + 1 + period
    
    myema[start_idx] = np.mean(series[lastnan_idx + 1:start_idx])
    for i in range(start_idx+1,len(series)):
    #start_idx = period
    #myema[start_idx] = np.mean(series[0:period])
    #for i in range(period+1,len(series)):
        #print(myema[i-1]*(1+period-smoothing))
        myema[i] = myema[i-1]*(1+period-smoothing)/(1+period)+series[i]*(smoothing)/(1+period)
    return myema
def MYEMA(series,period,debug = False):
    if isinstance(series,pd.Series):
        raw = series.to_numpy()
        if debug:
            print("MYEMA")
            print(np.where(np.isnan(raw)))
            print(type(np.where(np.isnan(raw))))
            print("FUCK U PIECE OF SHIT")
    else:
        raw = series
    return pd.Series(NUMBAEMA(raw,period),index = series.index)


# This version is base on long range
def LTSupport(ema,emastd = None,emastdd1 = None,emaperiod = None):
    emastd      =  pd.Series(ema).rolling(emaperiod).std().to_numpy() if emastd is None else emastd
    emastdd1    = D1(emastd) if emastdd1 is None else emastdd1
    return LTSupportCalc(ema,emastd,emastdd1)
@numba.jit(nopython = True)
def LTSupportCalc(ema,emastd,emastdd1):
    support = np.zeros(len(ema))
    support.fill(np.nan)
    
    enterid   = -1
    exitid    = -1
    direction = 0
    for i in range(1,len(ema)):
        if np.sign(emastdd1[i]) == 1 and np.sign(emastdd1[i-1]) == -1: #sign(nan) == sign(nan) gives false
            if ema[i]>ema[i-1]:
                direction =  1
            elif ema[i]<ema[i-1]:
                direction = -1
            enterid = i
        if np.sign(emastdd1[i]) == -1 and np.sign(emastdd1[i-1]) == 1:
            exitid = i
            if enterid != -1:
                support[enterid:exitid+1] = direction
            enterid = -1
            exitid = -1
    return support

