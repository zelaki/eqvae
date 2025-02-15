#!/bin/bash

mkdir -p pretrained_models
wget -O pretrained_models/kl-f8.zip https://ommer-lab.com/files/latent-diffusion/kl-f8.zip
unzip pretrained_models/kl-f8.zip -d pretrained_models
rm pretrained_models/kl-f8.zip