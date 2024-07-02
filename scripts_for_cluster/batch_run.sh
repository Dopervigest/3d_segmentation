#!/bin/bash

bash bash_run.sh 1 16344 cnn acdc True
bash bash_run.sh 1 16344 cnn clinical False

bash bash_run.sh 1 128 unet acdc True
bash bash_run.sh 1 128 unet clinical False

bash bash_run.sh 1 128 resnet1 acdc True
bash bash_run.sh 1 128 resnet1 clinical False

bash bash_run.sh 1 128 resnet18 acdc True
bash bash_run.sh 1 128 resnet18 clinical False

bash bash_run.sh 1 128 resnet34 acdc True
bash bash_run.sh 1 128 resnet34 clinical False

bash bash_run.sh 1 128 resnet50 acdc True
bash bash_run.sh 1 128 resnet50 clinical False

bash bash_run.sh 1 128 resnet101 acdc True
bash bash_run.sh 1 128 resnet101 clinical False

bash bash_run.sh 1 128 resnet152 acdc True
bash bash_run.sh 1 128 resnet152 clinical False

