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
