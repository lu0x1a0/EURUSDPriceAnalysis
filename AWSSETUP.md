# instructions to set up aws server
[source](https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04)

[SOURCE THAT WORKS FROM OFFICIAL FOR AARCH64](https://docs.anaconda.com/anaconda/install/graviton2/)
```
sudo apt-get update
sudo apt-get install curl
cd /tmp
#curl –O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
#curl -O https://repo.anaconda.com/archive/Anaconda3-2021.04-Linux-aarch64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh

sha256sum Anaconda3–2020.02–Linux–x86_64.sh
bash Miniconda3-latest-Linux-aarch64.sh

restart shell
conda update conda
conda config --set restore_free_channel true

conda create -n newenv
conda activate newenv
conda install python=3.9.7
pip install trading_ig
conda install pandas -y
...
cat > ./config.py
```
DONT USE CURL, it doesnt work

TRIED TO CREATE A YML for aws t4g instance, failed due to talib and pytorch intel mkl libraries

after environment set up, ran 
```
python Streaming/streamer.py |& tee EXECUTION_LOG.log
```
to see whether it is still executing:
```
ps -h
```
no idea why -h, cant find -h on linux man
i.e. 
```
man ps
```