set -eu

SOURCE_BETAS="/nas/data/Angry/CREMA-D-NEW/AUDIO-HI-BETAS/*.pt /nas/data/Angry/CREMA-D-NEW/AUDIO-XX-BETAS/*.pt"
SOURCE_BETAS=$(eval echo ${SOURCE_BETAS})
TARGET_BETAS="/nas/data/Angry/CREMA-D-NEW/VIDEO-HI-BETAS/*.pt /nas/data/Angry/CREMA-D-NEW/VIDEO-XX-BETAS/*.pt"
TARGET_BETAS=$(eval echo ${TARGET_BETAS})

python train.py --betas-source ${SOURCE_BETAS} --betas-target ${TARGET_BETAS} --model transformer --output transformer.pkl