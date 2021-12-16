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