import numba
import numpy as np

@numba.jit(nopython = True)
def frameTillMaxMean(pricearr,binaryInd):
    #store: class, startInd, endInd, maxInd, diff 
    store = np.zeros( (int(len(pricearr)/3),5) )
    store.fill(np.nan)

    entrycount = 0
    
    startidx = -1
    
    sign = np.nan
    #amendsign = np.nan
    maxidx = -1
    temp = np.nan
    for i in range(len(pricearr)):
        if not np.isnan(binaryInd[i]) and not np.isnan(pricearr[i]):            
            if binaryInd[i] != sign:
                if startidx!=-1:
                    #store[entrycount,:] = np.array([sign,startidx,i-1, maxidx, pricearr[maxidx]-pricearr[startidx]])
                    store[entrycount-1,0] = sign
                    store[entrycount-1,1] = startidx
                    store[entrycount-1,2] = i-1
                    store[entrycount-1,3] = maxidx
                    store[entrycount-1,4] = pricearr[maxidx]-pricearr[startidx]
                    #print(store[entrycount])
                startidx = i
                maxidx =  i
                entrycount += 1
                sign = binaryInd[i]
                #amendsign = (sign-0.5)*2
            else:
                if not np.isnan(sign):
                    if (pricearr[i]-pricearr[maxidx])*sign>0:
                    #temp = (pricearr[i]-pricearr[maxidx])*sign
                    #if temp>0:
                        maxidx = i
    return store, entrycount
@numba.jit(nopython = True)
def zoneQuantile(pricearr,binaryInd):
    #store: class, startInd, endInd, maxInd, quantiles[0.0,0.1,...0.9,1]~11 
    #  15 
    store = np.zeros( (int(len(pricearr)/10),15) )
    store.fill(np.nan)

    entrycount = 0
    
    startidx = -1
    
    sign = np.nan
    #amendsign = np.nan
    maxidx = -1
    temp = np.nan
    for i in range(len(pricearr)):
        if not np.isnan(binaryInd[i]) and not np.isnan(pricearr[i]):            
            if binaryInd[i] != sign:
                if startidx!=-1:
                    #store[entrycount,:] = np.array([sign,startidx,i-1, maxidx, pricearr[maxidx]-pricearr[startidx]])
                    store[entrycount-1,0] = sign
                    store[entrycount-1,1] = startidx
                    store[entrycount-1,2] = i-1
                    store[entrycount-1,3] = maxidx
                    store[entrycount-1,4:] = np.nanquantile(pricearr[startidx:i],np.linspace(0,1,11))
                    #print(store[entrycount])
                startidx = i
                maxidx =  i
                entrycount += 1
                sign = binaryInd[i]
                #amendsign = (sign-0.5)*2
            else:
                if not np.isnan(sign):
                    if (pricearr[i]-pricearr[maxidx])*sign>0:
                    #temp = (pricearr[i]-pricearr[maxidx])*sign
                    #if temp>0:
                        maxidx = i
    return store, entrycount
@numba.jit(nopython = True)
def zonePriceVarFromStart(pricearr,binaryInd):
    # pricearr should be something like close, 
    # binaryInd could be 3way including [-1,0,1], just that 0 wouldnt be useful for max
    # store: class, startInd, endInd, maxInd, quantiles[0.0,0.1,...0.9,1]~11 
    #  15 
    store = np.zeros( (int(len(pricearr)/10),15) )
    store.fill(np.nan)

    entrycount = 0
    
    startidx = -1
    
    sign = np.nan
    #amendsign = np.nan
    maxidx = -1
    temp = np.nan
    for i in range(len(pricearr)):
        if not np.isnan(binaryInd[i]) and not np.isnan(pricearr[i]):            
            if binaryInd[i] != sign:
                if startidx!=-1:
                    #store[entrycount,:] = np.array([sign,startidx,i-1, maxidx, pricearr[maxidx]-pricearr[startidx]])
                    store[entrycount-1,0] = sign
                    store[entrycount-1,1] = startidx
                    store[entrycount-1,2] = i-1
                    store[entrycount-1,3] = maxidx
                    store[entrycount-1,4:] = np.nanquantile(pricearr[startidx:i]-pricearr[startidx],np.linspace(0,1,11))
                    #print(store[entrycount])
                startidx = i
                maxidx =  i
                entrycount += 1
                sign = binaryInd[i]
                #amendsign = (sign-0.5)*2
            else:
                if not np.isnan(sign):
                    if (pricearr[i]-pricearr[maxidx])*sign>0:
                    #temp = (pricearr[i]-pricearr[maxidx])*sign
                    #if temp>0:
                        maxidx = i
    return store, entrycount