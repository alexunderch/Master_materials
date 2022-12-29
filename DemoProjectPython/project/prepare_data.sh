#!/bin/bash

wget -O models.zip 'https://github.com/google-research/self-organising-systems/blob/master/assets/mnist_ca/models.zip?raw=true'
unzip -oq "models.zip" -d "/model_data"
rm models.zip