from DataHandler import OHLCTrainFactory

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch, numba

from talib import MACD, DEMA,EMA


from torch.utils.data import TensorDataset, Dataset
from typing import List,Callable

import sys
np.set_printoptions(threshold=sys.maxsize)
import time


@numba.jit(nopython = True)
def NUMBAEMA(series,period):
    myema = np.empty(len(series),dtype="float64")
    myema[:] = np.nan
    smoothing = 2
    start_idx = period

    myema[start_idx] = np.mean(series[0:period])
    for i in range(period+1,len(series)):
        #print(myema[i-1]*(1+period-smoothing))
        myema[i] = myema[i-1]*(1+period-smoothing)/(1+period)+series[i]*(smoothing)/(1+period)
    return myema
def MYEMA(series,period):
    if isinstance(series,pd.Series):
        raw = series.to_numpy()
    else:
        raw = series
    return pd.Series(NUMBAEMA(raw,period),index = series.index)




