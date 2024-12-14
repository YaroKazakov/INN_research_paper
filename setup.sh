#!/bin/bash

python3.8 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install --upgrade setuptools wheel

pip install "gymnasium[classic-control]"

sudo apt-get install -y libxcb-cursor-dev

pip install -r requirements.txt

pip install git+https://github.com/TheStageAI/TorchIntegral.git

pip install "gymnasium[box2d]"

git clone https://github.com/AntonioTepsich/Convolutional-KANs.git

pip install "gymnasium[mujoco]"
