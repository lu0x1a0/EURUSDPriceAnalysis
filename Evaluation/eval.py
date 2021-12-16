import numpy as np
def evalIndicator(price, proposed_support):
    # proposed_support has to be +1,0,-1

    sign = 0
    startidx = -1
    peak = -1
    peakidx = -1
    
    diff = 0 
    
    supportzonecount = 0
    for i in range(len(proposed_support)) :
        #sign change
        if np.sign(proposed_support[i])!=sign:
            # was it previously inside a zone last i
            if sign != 0:
                if startidx>=0:
                    supportzonecount += 1
                    diff += (peak-price[startidx])*sign
                startidx = i
                peak = price[i]
                peakidx = i
        if startidx != i and startidx!=-1:
            if (price[i]-peak)*sign>0:
                peak = price[i]
                peakidx = i
        sign = np.sign(proposed_support[i])
    return supportzonecount