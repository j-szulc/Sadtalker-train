#!/bin/bash

set -eu

# Usage function to display help
usage() {
    echo "Usage: $0 -d DATASET_PATH -m MODEL"
    echo "  -d DATASET_PATH   Path to the dataset"
    echo "  -m MODEL          Model to use (automl or transformer)"
    exit 1
}

# Parse command-line arguments
while getopts "d:m:" opt; do
    case ${opt} in
        d )
            DATASET_PATH=$OPTARG
            ;;
        m )
            MODEL=$OPTARG
            ;;
        * )
            usage
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "${DATASET_PATH:-}" ] || [ -z "${MODEL:-}" ]; then
    usage
fi

# Define source and target betas
SOURCE_BETAS="${DATASET_PATH}/AUDIO-HI-BETAS/*.pt ${DATASET_PATH}/AUDIO-XX-BETAS/*.pt"
SOURCE_BETAS=$(eval echo ${SOURCE_BETAS})
TARGET_BETAS="${DATASET_PATH}/VIDEO-HI-BETAS/*.pt ${DATASET_PATH}/VIDEO-XX-BETAS/*.pt"
TARGET_BETAS=$(eval echo ${TARGET_BETAS})

# Define output file based on model type
if [ "${MODEL}" == "automl" ]; then
    ARGS="--model automl --cpu --output automl.pkl"
elif [ "${MODEL}" == "transformer" ]; then
    ARGS="--model transformer --output transformer.pkl"
else
    echo "Error: Invalid model type. Choose either 'automl' or 'transformer'."
    exit 1
fi
ARGS=$(eval echo ${ARGS})

# Run the training command
python train.py --betas-source ${SOURCE_BETAS} --betas-target ${TARGET_BETAS} ${ARGS}
