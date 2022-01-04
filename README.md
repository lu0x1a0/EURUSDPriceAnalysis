# [Home](http://lu0x1a0.github.io)

## Current Task:
* Display Live Streaming Data from IG and aggregate to historical data
* make qtplot accept jupyter data

# Requirements
install conda environment from the environment.yml file via
```
conda env create -f environment.yml
```
or
```
conda env create -f environment.yml -p /home/user/anaconda3/envs/env_name
```
environment.yml is created via
```
conda env export | grep -v "^prefix: " > environment.yml
```

# How to visualize:
Data used in this repository are obtained [here](http://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes)

Run newqt.py, in MainWindow class change between Test1plot (minutes) and Test2plot (hourly)
![Screenshot](./Images/qtHourly.png)

# Predictions: 1 step Predition of test data for Dema_9-rolling_Standard Deviation_100 on Hourly data

details inside nnstruct.ipynb


![MSE Loss](./Images/MSELoss.png)
![BCE Loss](./Images/BCELoss.png)

# Preliminary 4 step prediction on scaled D1EMA100 on Hourly Data, 300 epochs
D1 means price[hour = i]-price[hour = (i-1)]

![4 step](./Images/nsteppredict.png)
## Discussion
The network can still improve further by adjusting learning rate, but the data source seems to be too noisy at small window scale.  Might consider smoothing it for prediction showcase. Or jump straight to reinforcement learning of entry leaving signals. 

Also suspecting that the loss of 1+n step is probably decreasing the learning speed of 1 step. s

#### Note This repository has a lot of redundant code reused from a past repo and codes inside ipynb are most likely obsolete.

