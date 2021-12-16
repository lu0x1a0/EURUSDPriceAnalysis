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
    *  try poisson reggression
