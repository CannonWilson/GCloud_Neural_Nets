#!/bin/bash
# script for setting up new vm for project

echo "Installing necessary packages from pip . . ."
pip install gdown tensorboard opencv-python

cd ~/.ssh
# Add ssh keys
echo "Adding ssh keys . . ."
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJ8X7q6i6F5BAqOS6SavxF3L6+lvGsC6NV8h66Ebc+ay CannonGWilson@gmail.com" > id_ed25519.pub
echo "Please enter the private key (end with blank line and hit enter): "
key_input = $(sed '/^$/q')
echo $key_input > id_ed25519 
chmod 600 id_ed25519
cd ~

# config git for my email and username
echo "Configuring git . . ."
git config --global user.name "CannonWilson"
git config --global user.email "CannonGWilson@gmail.com"

# git clone repo
echo "Cloning repo . . ."
git clone git@github.com:CannonWilson/GCloud_Neural_Nets.git

# Download dataset
cd GCloud_Neural_Nets
sh download_dataset.sh

echo "Happy hacking :)"