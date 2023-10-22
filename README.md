# Training Farm for FinRL
## Multi-GPU enabled ElegantRL PPO training for live Stock Market data

### This is not financial advice, do not use this bot on live stock markets 

Training farm allows to initiate multiple agent trainers leveraging multiple GPUs which allows to train on different models while testing different hyper parameters. It uses all the possbile configuration options in configurations.py  to create an index of all possible combinations. The user can select how many trainers to run at the same time based on their CPU, RAM and GPU capabilities. The main.py script will initialize the trainer.py scripts and add the arguments needed to train the agent.  

## Features

- The trained agent can then use the existing FINRL papertrading code
- You can track the performance of your training either via processes.csv or going to http://localhost:5678/dashboard
- When the training rewards is not high enough the script will kill it and move to the next configuration combination ensuring time is not wasted on unseccessful runs.
- Load balances training workloads between multiple Nvidia GPUs
- Able to cache downloads and data after treatment to speed up training 
- Keeps track of all processes via unique uids and saves the actor.pth (saved model) in folders which show the configuration parameters 
- Uses Websockets for the trainer scripts to communicate with main.py 


## Installation

Training farm works better on Python 3.10 due to FinRL requirements. The installation instructions are based on running ubuntu 22.04 but they are also applicable to other linux distros or even MacOS (side note the Apple M1 silicone performance on training easily surpasses even an RTX 4090)

Install the OS dependencies
```sh
apt-get update && apt-get install -y wget git build-essential cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig libopenblas-dev libsuitesparse-dev libgsl0-dev libfftw3-dev libglpk-dev libdsdp-dev
```
Install Miniconda (advised to avoid conflicts with other Python environments)

```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
After, if you are using linux and bash
```sh
~/miniconda3/bin/conda init bash
```
After, if you are using MacOS zsh
```sh
~/miniconda3/bin/conda init zsh
```
Restart your terminal and you should see a (base) in your terminal prompt. When you have it do the following
```sh
conda create -n trainer python==3.10
conda activate trainer
```
Your terminal should change the prompt from (base) to (trainer). Make sure that if you type python --version, the result is 3.10 or 3.10.xx (xx can be any number)

After you need to install the following dependencies
```sh
pip install wrds swig git+https://github.com/mikazlopes/FinRLOptimized
```
if that completes successfully your should then run
```sh
git clone https://github.com/mikazlopes/training-farm.git && cd training-farm
pip install -r requirements.txt
```
Now you should be ready to run the training. You can also use the Dockerfile to build an image with everything pre-installed. The image is built for NVIDIA CUDA enabled Docker servers. 

## Using the script

There are a couple of configurable parameters in main.py and configurations.py that you will need to tweak before running.

In main.py line 336 change NUM_INSTANCES = 3, to any number of instances your hardware supports running at the same time.

In configurations.py feel free to change the parameters as needed, for more documentation about them please check out FINRL repo at github (https://github.com/AI4Finance-Foundation/FinRL)

Bear in mind you need an Alpaca account to get the data (https://app.alpaca.markets) and test the agent afterwards

To run the script you just need to type "python main.py", it will start initializing the instances, bear in mind that the instances don't initiate all at the same time, they start after the previous ones reaches the training cycle and goes through the initial 20K steps. The purpose is to avoid overloading the CPU and RAM with all the instances going through the same stage and also ensuring the following instances can use the cache, greatly speeding the training process.

Depending on how many different combinations of configurations you have, it can take several days to run through all of them, especially if the period of historical stock market is higher than 5 years.

You can check the live performance by opening http://localhost:5678/dashboard, it will show how far is the training process in the top progress bar, top results and the ongoing active processes and the last rewards they got. You can stop individual processes or start new ones.

## Docker

You can also use the Dockerfile to build an image with everything pre-installed. The image is built for NVIDIA CUDA enabled Docker servers. 
```sh
docker build --platform linux/amd64 -f docker/Dockerfile -t training-farm .
docker run -it --rm training-farm
```
You will need to open port 5678 if you want to access the dashboard webpage.

## License
MIT

