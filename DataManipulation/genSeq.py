import pandas as pd
import talib
import numpy as np
#print (talib.get_functions())
from talib import MACD, DEMA,EMA

# idea: use data from JAN to APR to train LSTM, then predict on MAY & JUN

import pickle
def mergePeriod(start, end, beg = "DAT_ASCII_EURUSD_M1_2021", dir = "./",dump = False):
    s = pd.concat(
        [pd.read_csv(dir+beg+'{:02d}.csv'.format(x),sep=';',names = ['Open','High','Low','Close','Volume']) 
            for x in range(start,end+1) 
        ])
    if dump == True:
        with open(dir+beg+"_"+str(start)+"_"+str(end)+".pkl","wb") as f:
            pickle.dump(s,f,protocol=pickle.HIGHEST_PROTOCOL)

    return s
from typing import Callable, List
from abc import ABC, abstractmethod

class TrainData(ABC):
    features_to_x: List[str]
    features_to_y: List[str]
    label:Callable
    indicators:List[Callable]
    
    def __init__(self,data:pd.DataFrame):
        self.raw = data
    
    @property
    @abstractmethod
    def features_to_x(self) -> List[str]:
        pass
    @property
    @abstractmethod
    def features_to_y(self) -> List[str]:
        pass
    
    @abstractmethod
    def label():
        pass
    def genIndicatorData(raw):
        """create data to be read by the network and its output
        """
        NetData = raw
        for I in self.indicators:
            NetData[I.__name__] = I(NetData["Close"])
        NetData["label"] = label(NetData)
        return NetData
    
    def genTrainLabelPair(self,indicatorsData):
        trainlabel = pd.DataFrame()
        ycols = df.columns.get_indexer_for(features_to_x)
        ycols = df.columns.get_indexer_for(features_to_y)   
        trainlabel = indicatorsData.apply(lambda r: applyXYTransform.__func__(r,xcols,ycols),axis = 1,raw=True)
        return trainlabel
    
    def applyXYTransform(self,row,xcols,ycols):
        return xFeaturesTransform.__func__(row[xcols]), yFeaturesTransform.__func__(row[ycols])
    @abstractmethod
    def xFeaturesTransform():
        pass
    @abstractmethod
    def yFeaturesTransform():
        pass
class EMADiff_FixPeriodOsc_Data(TrainData):
    #label = EMADiff_FixPeriodOsc_Data.label

    @staticmethod
    def dema(data):
        return DEMA(data,9)
    @staticmethod
    def ema1(data):
        return EMA(data,20)
    @staticmethod
    def ema2(data):
        return EMA(data,50)
    @staticmethod
    def ema3(data):
        return EMA(data,100)
    @staticmethod
    def ema4(data):
        return EMA(data,200)
    @staticmethod
    def ema5(data):
        return EMA(data,300)
    @staticmethod
    def label(df):
        #short period def: minutes
        short_p = 120
        #long period def
        long_p = 24*60 #360 

        df['up'] = (df.dema>df.ema1)& (df.ema1>df.ema2) & (df.ema2>df.ema3) & (df.ema3>df.ema4) & (df.ema4>df.ema5)
        df['down'] = (df.dema<df.ema1) & (df.ema1<df.ema2) & (df.ema2<df.ema3) & (df.ema3<df.ema4) & (df.ema4<df.ema5)

        df['future_short_up'] = df.loc[::-1,'up'].rolling(short_p).sum(engine = 'numba').loc[::-1]/short_p
        df['future_short_down'] = df.loc[::-1,'down'].rolling(short_p).sum(engine = 'numba').loc[::-1]/short_p
        df['future_long_up'] = df.loc[::-1,'up'].rolling(long_p).sum(engine = 'numba').loc[::-1]/long_p
        df['future_long_down'] = df.loc[::-1,'down'].rolling(long_p).sum(engine = 'numba').loc[::-1]/long_p

        # symmetric between up and down
        df['fsu_v2'] = df.loc[::-1,'dema'].rolling(short_p).apply(lambda x: np.sum(x>x[-1]) ,engine ='numba',raw=True)/short_p
        df['flu_v2'] = df.loc[::-1,'dema'].rolling(long_p).apply(lambda x: np.sum(x>x[-1]) ,engine ='numba',raw=True)/long_p


        #return df[['future_short_up','future_short_down','future_long_up','future_long_down']].values.tolist()

        #df['both_up'] = (df['future_short_up']>0.6) & (df['future_long_up']>0.5)
        #df['both_down'] = (df['future_short_down']>0.6) & (df['future_long_down']>0.5)
        #df['neither'] = ~(df['both_up'] | df['both_down'])

        #df['both_up'] = (df['future_short_up']>0.6) & (df['flu_v2']>0.6)
        #df['both_down'] = (df['future_short_down']>0.6) & (df['flu_v2']<0.4)
        #df['neither'] = ~(df['both_up'] | df['both_down'])

        tmpU = (df['flu_v2']>0.6) & (df['flu_v2']>0.8)
        tmpD =(df['flu_v2']<0.4) & (df['flu_v2']<0.2)
        tmpN = ~(tmpU | tmpD)
        #tmpU = tmpU.replace(False,float('nan'))
        #tmpD = tmpD.replace(False,float('nan'))
        #tmpN = tmpN.replace(False,float('nan'))
        df['buy'] = tmpU
        df['sell'] = tmpD
        df['neither'] = tmpN

        #cols = df.columns.get_indexer_for(['both_up','both_down','neither'])
        #return df[['both_up','both_down','neither']].apply(lambda x:int(np.where(x==True)[0]),axis=1)
        #return df[['both_up','both_down','neither']].rolling(1).apply(lambda x: int(np.where(x==True)[0]),raw=True)#,engine='cython')#,axis=1)

        return df[['buy','sell','neither']].values.tolist()
        #label = df.apply(lambda x:,raw=True)
        #df['futureup','futuredown'].values.tolist()
        # return [short_sig, long_sig] \in [0,1]
    indicators = [
        dema.__func__,
        ema1.__func__,
        ema2.__func__,
        ema3.__func__,
        ema4.__func__,
        ema5.__func__,
    ]
    features_to_x = [i.__name__ for i  in indicators]
    features_to_y = ['buy','sell','neither']

#@array_function_dispatch(_sliding_window_view_dispatcher)
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
def sliding_window_view(x, window_shape, axis=None, *,
                        subok=False, writeable=False):
    """
    Create a sliding window view into the array with the given window shape.
    Also known as rolling or moving window, the window slides across all
    dimensions of the array and extracts subsets of the array at all window
    positions.
    
    .. versionadded:: 1.20.0
    Parameters
    ----------
    x : array_like
        Array to create the sliding window view from.
    window_shape : int or tuple of int
        Size of window over each axis that takes part in the sliding window.
        If `axis` is not present, must have same length as the number of input
        array dimensions. Single integers `i` are treated as if they were the
        tuple `(i,)`.
    axis : int or tuple of int, optional
        Axis or axes along which the sliding window is applied.
        By default, the sliding window is applied to all axes and
        `window_shape[i]` will refer to axis `i` of `x`.
        If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to
        the axis `axis[i]` of `x`.
        Single integers `i` are treated as if they were the tuple `(i,)`.
    subok : bool, optional
        If True, sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional
        When true, allow writing to the returned view. The default is false,
        as this should be used with caution: the returned view contains the
        same memory location multiple times, so writing to one location will
        cause others to change.
    Returns
    -------
    view : ndarray
        Sliding window view of the array. The sliding window dimensions are
        inserted at the end, and the original dimensions are trimmed as
        required by the size of the sliding window.
        That is, ``view.shape = x_shape_trimmed + window_shape``, where
        ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less
        than the corresponding window size.
    See Also
    --------
    lib.stride_tricks.as_strided: A lower-level and less safe routine for
        creating arbitrary views from custom shape and strides.
    broadcast_to: broadcast an array to a given shape.
    Notes
    -----
    For many applications using a sliding window view can be convenient, but
    potentially very slow. Often specialized solutions exist, for example:
    - `scipy.signal.fftconvolve`
    - filtering functions in `scipy.ndimage`
    - moving window functions provided by
      `bottleneck <https://github.com/pydata/bottleneck>`_.
    As a rough estimate, a sliding window approach with an input size of `N`
    and a window size of `W` will scale as `O(N*W)` where frequently a special
    algorithm can achieve `O(N)`. That means that the sliding window variant
    for a window size of 100 can be a 100 times slower than a more specialized
    version.
    Nevertheless, for small window sizes, when no custom algorithm exists, or
    as a prototyping and developing tool, this function can be a good solution.
    Examples
    --------
    >>> x = np.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    This also works in more dimensions, e.g.
    >>> i, j = np.ogrid[:3, :4]
    >>> x = 10*i + j
    >>> x.shape
    (3, 4)
    >>> x
    array([[ 0,  1,  2,  3],
           [10, 11, 12, 13],
           [20, 21, 22, 23]])
    >>> shape = (2,2)
    >>> v = sliding_window_view(x, shape)
    >>> v.shape
    (2, 3, 2, 2)
    >>> v
    array([[[[ 0,  1],
             [10, 11]],
            [[ 1,  2],
             [11, 12]],
            [[ 2,  3],
             [12, 13]]],
           [[[10, 11],
             [20, 21]],
            [[11, 12],
             [21, 22]],
            [[12, 13],
             [22, 23]]]])
    The axis can be specified explicitly:
    >>> v = sliding_window_view(x, 3, 0)
    >>> v.shape
    (1, 4, 3)
    >>> v
    array([[[ 0, 10, 20],
            [ 1, 11, 21],
            [ 2, 12, 22],
            [ 3, 13, 23]]])
    The same axis can be used several times. In that case, every use reduces
    the corresponding original dimension:
    >>> v = sliding_window_view(x, (2, 3), (1, 1))
    >>> v.shape
    (3, 1, 2, 3)
    >>> v
    array([[[[ 0,  1,  2],
             [ 1,  2,  3]]],
           [[[10, 11, 12],
             [11, 12, 13]]],
           [[[20, 21, 22],
             [21, 22, 23]]]])
    Combining with stepped slicing (`::step`), this can be used to take sliding
    views which skip elements:
    >>> x = np.arange(7)
    >>> sliding_window_view(x, 5)[:, ::2]
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6]])
    or views which move by multiple elements
    >>> x = np.arange(7)
    >>> sliding_window_view(x, 3)[::2, :]
    array([[0, 1, 2],
           [2, 3, 4],
           [4, 5, 6]])
    A common application of `sliding_window_view` is the calculation of running
    statistics. The simplest example is the
    `moving average <https://en.wikipedia.org/wiki/Moving_average>`_:
    >>> x = np.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> moving_average = v.mean(axis=-1)
    >>> moving_average
    array([1., 2., 3., 4.])
    Note that a sliding window approach is often **not** optimal (see Notes).
    """
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)