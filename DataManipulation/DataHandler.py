from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch, numba

from time import time

from talib import MACD, DEMA,EMA
from .indicators import D1,D2
# will use my own implementation of EMA
# it differs from talib slightly at the beginning then is the same,
# using self write, because talib cant handle extremely large period like 20 days on minute interval,
# significantly slower than talib, but think its due to numba initial call overhead.
# i.e. i think it scales well

from .indicators import MYEMA

from torch.utils.data import TensorDataset, Dataset
from typing import List,Callable

import sys
np.set_printoptions(threshold=sys.maxsize)


import pickle


def getOHLC_pickle(pklpath):
        import pickle
        with open(pklpath,'rb') as f:
            data = pickle.load(f)
            return data
def getOHLC_raw(to_pickle=False):
    from DataManipulation.DataHandler import  mergeMonths, DemaMinDayMultinom
    import pandas as pd
    from torch.utils.data import ConcatDataset
    #data = mergePeriod(1,4,beg="DAT_ASCII_EURUSD_M1_2021",dir="./eurusd2021/",dump=False)
    data2021 = mergeMonths(start=1, end=9,beg="DAT_ASCII_EURUSD_M1_2021",dir = './Data/eurusd2021/', dump=False)

    dir = "./Data/"
    beg = "DAT_ASCII_EURUSD_M1_"

    years = range(2010,2021)
    dataY = []
    for y in years:
        dataY.append(pd.read_csv(dir+beg+str(y)+'.csv',sep=';',names = ['Open','High','Low','Close','Volume']))
    dataY.append(data2021)
    data = pd.concat(dataY)
    data.index.name = "Date"
    data.index = pd.to_datetime(data.index)
    if to_pickle:
        import pickle
        with open('OHLC.pkl','wb') as f:
            pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    return data

def mergeMonths(start, end, beg = "DAT_ASCII_EURUSD_M1_2021", dir = "./",dump = False):
    s = pd.concat(
        [pd.read_csv(dir+beg+'{:02d}.csv'.format(x),sep=';',names = ['Open','High','Low','Close','Volume']) 
            for x in range(start,end+1) 
        ])
    if dump == True:
        with open(dir+beg+"_"+str(start)+"_"+str(end)+".pkl","wb") as f:
            pickle.dump(s,f,protocol=pickle.HIGHEST_PROTOCOL)

    return s
class TrainDataFactory(ABC):
    def __init__(self,raw:pd.DataFrame):
        self.raw = raw
    @abstractmethod
    def generateTrainingPair(self):
        pass
class OHLCTrainFactory(TrainDataFactory):
    def __init__(self,OHLC:pd.DataFrame):
        super().__init__(OHLC)
    @abstractmethod
    def Xfeatures(self) -> np.ndarray:
        pass
    @abstractmethod
    def Ylabels(self) -> np.ndarray:
        pass

    def generateTrainingPair(self) -> Dataset:
        self.x = self.Xfeatures()
        self.y = self.Ylabels()
        return TensorDataset(self.x,self.y)
from enum import Enum
class Direction(Enum):
    Up = 0
    Down = 1
    Neither = 2
class DEMAsDifference(OHLCTrainFactory):
    def __init__(self, OHLC, emaperiods : List[int] = [20,50,100,200,300]):
        super().__init__(OHLC)
        self.data = "Close"
        demaperiod = 9
        func = lambda data:DEMA(data,demaperiod)
        func.__name__ = 'dema'+str(demaperiod)
        self.indicators = [func]
        for i in emaperiods:
            func = self.createEMAFunc(i)
            func.__name__ = 'ema'+str(i)
            self.indicators.append(func)
        self.orderColumns = [self.data]+[i.__name__ for i in self.indicators]
        # window size for future label lookahead
        self.short_p = 120 
        self.long_p = 12*60 #360 
        # n.o. data points use for prediction
        self.lookback = 4*60 
    def createEMAFunc(self,period) -> Callable:
        return lambda data:MYEMA(data,period)
    def Xfeatures(self):
        for I in self.indicators:
            self.raw[I.__name__] = I(self.raw[self.data])
        self._features = pd.DataFrame()
        for i, column in enumerate(self.orderColumns):
            if i != 0:
                self._features[str(i)] = self.raw[self.orderColumns[i-1]] - self.raw[column]
        return torch.Tensor(self._features.values)
    def Ylabels(self):
        df = self.raw
        #  column name def
        dema = self.orderColumns[1]
        emas = self.orderColumns[2:]
        
        labelPreprocess = pd.DataFrame()
                
        #symmetric between up and down
        labelPreprocess['fsu_v2'] = df.loc[::-1,dema].rolling(self.short_p).apply(lambda x: np.sum(x>x[-1]) ,engine ='numba',raw=True)/self.short_p
        labelPreprocess['flu_v2'] = df.loc[::-1,dema].rolling(self.long_p).apply(lambda x: np.sum(x>x[-1]) ,engine ='numba',raw=True)/self.long_p

        tmpU = (labelPreprocess['flu_v2']>0.6) & (labelPreprocess['flu_v2']>0.8)
        tmpD =(labelPreprocess['flu_v2']<0.4) & (labelPreprocess['flu_v2']<0.2)
        
        self._labelPreprocess = labelPreprocess
        self.label = pd.DataFrame()
        self.label['both_up'] = tmpU
        self.label['both_down'] = tmpD
        #self.label['neither'] = tmpN
        #self.label['class'] =  self.label.apply(lambda row:,axis=1)
        
        #t = time.time()
        label = DEMAsDifference2Momentum._compute_label(labelPreprocess['fsu_v2'].to_numpy(),labelPreprocess['flu_v2'].to_numpy())
        #print(time.time()-t)
        self.label = pd.Series(label,index = labelPreprocess.index,name='label')
        return torch.Tensor(self.label.values) #self.label[['both_up','both_down','neither']].values.tolist()
    @staticmethod
    @numba.jit(nopython=True)
    def _compute_label(future_short_up,future_long_up):
        n = len(future_short_up)
        result = np.zeros(n, dtype="float64")
        assert len(future_short_up) == len(future_long_up)
        for i in range(n):
            if i == 301:
                p = 1
            if np.isnan(future_long_up[i])==1 or np.isnan(future_short_up[i])==1:
                result[i] = np.NAN
            elif (future_long_up[i]>0.6) & (future_short_up[i]>0.8):
                result[i] = Direction.Up.value
            elif (future_long_up[i]<0.6) & (future_short_up[i]<0.8):
                result[i] = Direction.Down.value
            else:
                result[i] = Direction.Neither.value #np.double(Direction.Neither.value) #2 #float(Direction.Neither.value)
        return result
    def generateTrainingPair(self,transform:List[str]) -> Dataset:
        # overrides the above because pandas rolling windows does not have an iterator
        x = self.Xfeatures()
        y = self.Ylabels()
        y = torch.zeros((len(self.raw),1))
        
        self._xyjoin = self._features
        self._xyjoin['label'] = self.label
        
        self._sliding_win = np.lib.stride_tricks.sliding_window_view(self._xyjoin.values,(self.lookback,len(self._xyjoin.columns))).squeeze()
        self.samples = self._sliding_win[
            (
                ~np.isnan(self._sliding_win[:,self.lookback-1,len(self._xyjoin.columns)-1]) & # filter out where label is nan
                ~np.isnan(self._sliding_win[:,0,len(self._xyjoin.columns)-2])                 # filter out where largest indicator window is nan
            ),:,:
        ]

        prop_x = self.samples[:,:,:len(self._xyjoin.columns)-1]
        if 'normalize' in transform:                                                          # normalize all sequence by first abs difference
            #print(prop_x.shape)
            first = np.abs(prop_x[:,0,0])
            prop_x = np.swapaxes(prop_x,0,2)
            prop_x = prop_x/first
            prop_x = np.swapaxes(prop_x,0,2)
        prop_y = self.samples[:,self.lookback-1,len(self._xyjoin.columns)-1]
        #print(prop_y)
        #print("-----------------------------------")
        #print(prop_x[:,:,5])
        #print("-----------------------------------")
        #print(prop_x[:,:,0])
        
        self.prop_y = torch.LongTensor(prop_y)#.astype(np.int64))
        self.prop_x = torch.Tensor(prop_x)
        self.num_features = prop_x.shape[2]
        self.num_classes = 3
        return TensorDataset(self.prop_x,self.prop_y)

class DEMAsDiffnDerivatives(OHLCTrainFactory):
    def __init__(self, OHLC, emaperiods : List[int] = [20,50,100,200,300]):
        super().__init__(OHLC)
        self.datacol = "Close"
        demaperiod = 9
        func = lambda data:DEMA(data,demaperiod)
        func.__name__ = 'dema'+str(demaperiod)
        self.indicators = [func]
        
        for i in emaperiods:
            func = self.createEMAFunc(i)
            func.__name__ = 'ema'+str(i)
            self.indicators.append(func)

        # calculate ema and respective derivatives
        for i in self.indicators:
            self.raw[i.__name__] = i(self.raw[self.datacol])
            self.raw[i.__name__+'D1'] = self.raw[i.__name__].rolling(2).apply(lambda x:x[1]-x[0],raw=True,engine='numba')
            self.raw[i.__name__+'D2'] = self.raw[i.__name__].rolling(3).apply(lambda x:x[2]-2*x[1]+x[0],raw=True,engine='numba')
        self.colnameindex = {y:x for x,y in enumerate(self.raw.columns)}
        self.D1index = [x for x in self.colnameindex if x[-2:]=='D1']
        self.D2index = [x for x in self.colnameindex if x[-2:]=='D2']

        # derivative summary:
        D1Sign = self.raw[self.D1index]>0
        self.raw['D1Pos']= np.all(D1Sign.to_numpy(),axis=1)
        self.raw['D1Neg']= np.all(~D1Sign.to_numpy(),axis=1)
        
        DnSign = self.raw[self.D1index+self.D2index]>0
        self.raw['DnPos']= np.all(DnSign.to_numpy(),axis=1)
        self.raw['DnNeg']= np.all(~DnSign.to_numpy(),axis=1)
        
        # calculate period differences
        self.diffcolnames = []
        self.periodcols = [self.datacol] + [x.__name__ for x in self.indicators]
        for i, column in enumerate(self.periodcols):
            if i != 0:
                newcolname = self.periodcols[i-1]+'_'+column
                self.raw[newcolname] = self.raw[self.periodcols[i-1]] - self.raw[column]
                self.diffcolnames.append(newcolname)
        #self.diffcolidx = [self.raw.columns.get_loc(x) for x in diffcolnames]
        
        # period difference uni-direction check
        DiffSign  = self.raw[self.diffcolnames]>0
        
        self.raw['DiffPos']= np.all(DiffSign.to_numpy(),axis=1)
        self.raw['DiffNeg']= np.all(~DiffSign.to_numpy(),axis=1)
        #
        # window size for future label lookahead
        self.short_p = 120 
        self.long_p = 12*60 #360 
        # n.o. data points use for prediction
        self.lookback = 4*60 

    def createEMAFunc(self,period) -> Callable:
        return lambda data:EMA(data,period)
    def Xfeatures(self) -> np.ndarray:
        pass
    def Ylabels(self) -> np.ndarray:
        pass

class DemaMinDayMultinom(OHLCTrainFactory):
    def __init__(self, OHLC, emaperiods : List[int] = [100,200,300,60*24*100,60*24*200]):
        super().__init__(OHLC)
        self.datacol = "Close"
        demaperiod = 9
        func = lambda data:DEMA(data,demaperiod)
        func.__name__ = 'dema'+str(demaperiod)

        #ema
        #emadiff
        #----------------------------
        #emad1
        #emad1 sign
        #emad1diff
        #emad1diff sign
        #----------------------------
        #emad2
        #emad1 sign
        #emad2diff
        #emad2diff sign
        self.indicators = [func]
        
        for i in emaperiods:
            func = self.createEMAFunc(i)
            func.__name__ = 'ema'+str(i)
            self.indicators.append(func)

        # calculate ema and respective derivatives
        t = time()
        for i in self.indicators:
            self.raw[i.__name__] = i(self.raw[self.datacol])
            self.raw[i.__name__+'D1'] = D1(self.raw[i.__name__])
            self.raw[i.__name__+'D2'] = D2(self.raw[i.__name__])
            #self.raw[i.__name__+'D1'] = self.raw[i.__name__].rolling(2).apply(lambda x:x[1]-x[0],raw=True,engine='numba')
            #self.raw[i.__name__+'D2'] = self.raw[i.__name__].rolling(3).apply(lambda x:x[2]-2*x[1]+x[0],raw=True,engine='numba')
        self.colnameindex = {y:x for x,y in enumerate(self.raw.columns)}
        self.D1index = [x for x in self.colnameindex if x[-2:]=='D1']
        self.D2index = [x for x in self.colnameindex if x[-2:]=='D2']
        tx = time()
        print(tx-t)
        t=tx
        # derivative summary:
        

        # calculate indicator and gradient differences between idicators 
        self.periodDiffNames = []
        self.gradientDiffNames = []
        self.gradient2DiffNames  = []

        self.periodcols = [self.datacol] + [x.__name__ for x in self.indicators]
        for i, column in enumerate(self.periodcols):
            if i > 0:
                indicatordiffname = self.periodcols[i-1]+'_'+column
                self.raw[indicatordiffname] = self.raw[self.periodcols[i-1]] - self.raw[column]
                self.periodDiffNames.append(indicatordiffname)
            if i > 1:
                gradientdiffname = self.periodcols[i-1]+'D1'+'_'+column+'D1'
                self.raw[gradientdiffname] = self.raw[self.periodcols[i-1]+'D1'] - self.raw[column+'D1']
                self.gradientDiffNames.append(gradientdiffname)

                gradient2diffname = self.periodcols[i-1]+'D2'+'_'+column+'D2'
                self.raw[gradient2diffname] = self.raw[self.periodcols[i-1]+'D2'] - self.raw[column+'D2']
                self.gradient2DiffNames.append(gradient2diffname)
        tx = time()
        print(tx-t)
        t=tx
        self.raw['_y_enter_exit'],self.zonecount,self.bigzonecount = DemaMinDayMultinom._y_close_window_trim(
            self.raw['Close'].to_numpy(), 
            np.sign( (self.raw[self.indicators[1].__name__]-self.raw[self.indicators[-3].__name__]).to_numpy() ),
        )
        tx = time()
        print(tx-t)
        t=tx
    @staticmethod
    @numba.jit(nopython=True)
    def mergesupport(positive,negative):
        result = np.empty(shape = positive.shape)
        result.fill(np.nan)
        
        return result
    @staticmethod
    @numba.jit(nopython=True)
    def _y_close_window_trim(price,proposed_support):
        """creates signals to predict by the lstm with entry calculatable, 
            but exit time based on period max

        Args:
            price ([type]): [description]
            proposed_support ([type]): [description]

        Returns:
            [type]: [description]
        """
        result = np.empty(shape = (len(proposed_support),) )
        #result.fill(np.nan)
        result.fill(0)
        threshold = 0.003

        sign = 0
        startidx = -1
        peak = -1
        peakidx = -1

        zonecount = 0
        supportzonecount = 0
        for i in range(len(proposed_support)) :
            #sign change
            if np.sign(proposed_support[i])!=sign:
                # was it previously inside a zone last i
                if startidx>=0:
                    #print((price[peakidx]-price[startidx]),sign)
                    if (price[peakidx]-price[startidx])*sign>threshold and peakidx-startidx>60:
                        result[startidx:peakidx+1] = sign
                        result[peakidx:i] = 0
                        zonecount += 1
                    else:
                        result[startidx:i] = 0
                    supportzonecount += 1
                startidx = i
                peak = price[i]
                peakidx = i
            if startidx != i and startidx!=-1:
                if (price[i]-peak)*sign>0:
                    peak = price[i]
                    peakidx = i
            sign = np.sign(proposed_support[i])
        return result,zonecount,supportzonecount
    def createEMAFunc(self,period) -> Callable:
        return lambda data:MYEMA(data,period)
    def Xfeatures(self) -> np.ndarray:
        pass
    def Ylabels(self) -> np.ndarray:
        pass
