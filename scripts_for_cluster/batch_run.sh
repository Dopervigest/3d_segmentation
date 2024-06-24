#!/bin/bash

bash bash_run.sh 1 16344 cnn acdc True
bash bash_run.sh 1 16344 cnn clinical False


bash bash_run.sh 1 128 resnet acdc True
bash bash_run.sh 1 128 resnet clinical False

bash bash_run.sh 1 128 unet acdc True
bash bash_run.sh 1 128 unet clinical False