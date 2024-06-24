#!/bin/bash

# Function to print usage
usage() {
    echo "Usage: $0 <epochs> <batch> <model> <dataset> <transfer>"
    echo "  <epochs>  - integer value of how many epochs to pass to python script"
    echo "  <batch>   - integer value of how many data elements need to be passed to python script"
    echo "  <model>   - string value of name of the model, must be: cnn, resnet or unet"
    echo "  <dataset> - string value of which dataset to use, must be either acdc or clinical"
    echo "  <transfer>      - boolean value of wether to use transfer learning or not, must be either True or False"
    exit 1
}

# Check if the number of arguments is correct
if [ "$#" -ne 5 ]; then
    usage
fi

# Assign arguments to variables
epochs=$1
batch=$2
model=$3
dataset=$4
transfer=$5

# Validate the integer arguments
if ! [[ "$epochs" =~ ^-?[0-9]+$ ]]; then
    echo "Error: Epochs argument is not a valid integer"
    usage
fi

if ! [[ "$batch" =~ ^-?[0-9]+$ ]]; then
    echo "Error: Batch argument is not a valid integer"
    usage
fi

# Validate the boolean argument
if [ "$transfer" != "True" ] && [ "$transfer" != "False" ]; then
    echo "Error: transfer argument must be 'True' or 'False'"
    usage
fi


# Print the arguments
echo "bash script recieved the following arguments:"
echo "Epochs: $epochs"
echo "Batch size: $batch"
echo "Model: $model"
echo "Dataset: $dataset"
echo "Transfer Learning: $transfer"
echo ""


if [ "$transfer" = "True" ]; then
    python train.py --epochs=$epochs --batch=$batch --model=$model --dataset=$dataset
    python train.py --epochs=$epochs --batch=$batch --model=$model --dataset=clinical --tl=True
else
    
    python train.py --epochs=$epochs --batch=$batch --model=$model --dataset=$dataset 
fi