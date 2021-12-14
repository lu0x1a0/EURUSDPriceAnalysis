import numba
import numpy as np
@numba.jit(nopython=True)
def split3way(price,short_support,long_support,threshold):
    result = np.zeros(shape=(len(price),))
    result.fill(np.nan)
    for i in range(len(price)):
        if np.isnan(price[i]) or np.isnan(short_support[i]) or np.isnan(long_support[i]):
            #print(price[i],short_support[i],long_support[i])
            pass
        elif short_support[i]-long_support[i]>threshold: #and long_support[i]-long_support[i-1]>=0:
            result[i] = 1
        elif long_support[i]-short_support[i]>threshold: #and long_support[i]-long_support[i-1]<=0:
            result[i] = -1
        else:
            result[i] = 0
    return result