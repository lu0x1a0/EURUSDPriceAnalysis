# Experiment logs

* ideas:
    * use logistic regression based on minute interval data
    
        * use ema differences vs ema + ema d1 differences as predictors.
            * no difference in result.
        * increase filter threshold for idle vs buy/sell logistic probability 
            * did not change the ratio enter|enter vs enter|idle significantly.
    * previously in t2:

        * tried LTSM, might ve overfitted. also label engineering wasnt great.
            
        * $\therefore$ should look into reinforcement.
  
    * follow Dr chandra's code on time series prediction, change data source and see outcome.
    *  try random forest
        * before that try to manual tune an hourly strat
        * try to predict long term suport before happening using D1 and D2 2 step taylor series 
            D2 for 2400 h and 4800 h (100 day and 200 day using hour) is extremely unstable. 
                to utilize, will need to resample upwards again
            D1 seems to offer a degree of insight. need further testing.
    *  try poisson reggression
    * data normalization:
        * input:
            D1dema_p  : std(D1ema, year/2)
            D1ema_p   : std(D1ema, year/2)
            std(D1ema, p = 9,100,6000) : min max(std(D1ema), year)

    * prediction of indicator:
        std of dema9
            loss: use reverse kl,
            features:
                past std of dema9 normalized by 1 year min max
                past std of ema100 normalized by 1 year min max
                past D1dema9 normalized by 1 year min max
                past D1ema100 normalized by 1 year min max
            model: to start, use 1D convolution -> 2D convolution -> FC -> 1D probability output
        D1dema9
            output:  2x1 output, one for each direction and each out = magnitude.
