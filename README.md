# Talking face

Fork of [SadTalker](https://sadtalker.github.io) (CVPR 2023) - a more realistic text-to-video model with expression modeling.
This is a student project for the "Team programming project in machine learning" at the University of Warsaw (academic year 2023/2024).

This project is based on the following repositories:

1. https://github.com/OpenTalker/SadTalker - the code for the research paper our work is based upon,
1. https://github.com/wytcsuch/Sadtalker-train - the code for training the PoseVAE part of the model.

See the linked repositories for more information about the original project and the original "README" files.

## Installation

Installing dependencies from `requirements.txt` should be enough:
```bash
pip install -r requirements.txt
```
The code has been tested to work with Python 3.8.

## Our contribution

In general, we have focused on improving the expression modeling part of the SadTalker model, by adding emotions to otherwise quite bland facial expressions. 

### Generating expression vectors

The main object we were manipulating are expression vectors, which is a part of the latent space from [https://arxiv.org/abs/1903.08527](Deng et al.), also used by the authors of the original paper.

The script `get_betas.py` can be used to generate expression vectors for a given video. The script uses a pre-trained model to generate the vectors. The vectors are saved in a `.pt` file.
```bash
python get_betas.py --video_path path/to/video.mp4 --output-betas path/to/output.pt --device [cuda|cpu]
```

### Training the expression model

As mentioned earlier, we have focused on the expression vectors part of the model. The SadTalker consists of, among others, an "ExpNet" module used for generating expression vectors from audio. We manipulated the ExpNet module in several ways. All of them have been trained on CREMA-D and/or RAVDESS datasets, which consist of audio recordings with emotional content.

#### Retraining ExpNet from scratch

To retrain the ExpNet module from scratch, use the `train_expnet.py` script.
```bash
python train_expnet.py --driven_audios folder/with/audios --source_images folder/with/images --betas folder/with/target/betas --output path/to/output.pth
```
Each of the folders should contain files with the same names, but different extensions (appropriately ".wav", ".jpg", ".pt").
Folder `folder/with/target/betas` should be populated with expression vectors generated by `get_betas.py`, from the emotional videos as described above.

To test, change the `--expnet-checkpoint` argument in `inference.py` from the original repo:
```bash
python inference.py --expnet-checkpoint path/to/output.pth --driven_audio path/to/audio.wav --source_image path/to/image.jpg --result_dir path/to/output
```

### Adding emotions to pretrained ExpNet

To add emotions to the pretrained ExpNet, use the `emotion_adding` folder. You can choose from models: `automl, transformer, linear`. Each model will take `FRAME_LEN==5` context window of audio expressions from the original ExpNet to predict a (hopefully better) expression vector.

To train the model:
1. Get the existing predictions for expression vectors:
```bash
python inference.py --save-betas path/to/source/betas/beta.pt --driven_audio path/to/audio.wav --source_image path/to/image.jpg
```
It will interrupt the inference and only keep the predicted expression vector.
2. Train the model:
```bash
python emotion_adding/train.py --betas-target folder/with/target/betas --betas-source path/to/source/betas --model [automl|transformer|linear]
```
which will train a choosed model and save it in `./checkpoints/{model_name}.pkl`.

To test:
1. Create the new expression vectors:
```bash
python emotion_adding/inference.py --model path/to/model.pkl --betas-source path/to/source/betas --output path/to/generated/betas
```
2. Override the original expression vectors in the `inference.py` script:
```bash
python inference.py --override-betas path/to/generated/betas --driven_audio path/to/audio.wav --source_image path/to/image.jpg --result_dir path/to/output
```

#### Additional features

Note that there are two additional arguments in `emotion_adding/inference.py` script:
- `alpha` - the weight of the new expression vector in the final expression vector,
- `ortho` - whether to orthogonalize the new expression vector with the original one.
