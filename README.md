# TFG - Early detection of melanoma using artificial intelligence (AI) - unfinished! :)

This repository includes the code of my Biomedical Engineering Degree final project.

## Table of Contents
- [Introduction](#introduction)
- [Software](#software)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
In this repository I present a convolutional network architecture based on the EfficientNet-B3 backbone for accurate binary classification of skin lesions. This model is significantly smaller than most existing models, but has a comparable prformance to larger alternatives that use more resources. The model was trained with crops. Crops are small skin lesion images cropped from a total body map of the skin of a patient extracted by a 3D Total Body Scanner (TBP scanner). They are lower resolution images in comparison to dermatoscopic images. My best model achieved a **ROC-AUC of 0.91**, an **F1-score of 0.90**, a **sensitivity of 0.84 (with a predefined threshold of 0.9 = 90%)** and a **accuracy of 0.83** on skin-lesion detection task. The model can be trained as a image-only model or a combining the image and metadata of the patient. Model scheme can be seen in the following illustration:

![image](https://github.com/user-attachments/assets/0f340340-9097-49b3-8a6a-d86b2536ef91)

## Software

### Software used during the project:
-  Python: 3.10.12
-  Nvidia Drivers: 525.147.05
-  CUDA: 11.5.119
-  cuDNN: 8.9.2

The training of the models was done using a **Nvidia A100 80G**, GPU is highly recommended to execute this project in order to have acceptable training times.

## Installation
You can install all necessary Python packages requeriments and dependencies by using:

```
pip install -r requreriments.txt
```
After having installed the dependencies. Download the train and test datasets and place them in the data folder along with their corresponding `.csv` metadata files. iToBoS data is not yet public although will be publicly released in the future. Expermients can be ran with ISIC challenges datasets.

## Usage
Training commands to run the script. However the use of `nohup` and `.ssh` files is recommended. Training time for the models ranges multiple hours to tens of hours, depending on the GPU you use.

After training, the model will be saved in model path established in the configuration. If you do not want or can not use WandB, the training logs will appear in the terminal through the training session. You can use as well the nohup command with a `.out` file.
```
python3 train.py --wandb_entity (username) --wandb_project (your project) --wandb_api_key (your API key) --dataset iToBoS --data_dir (your data path) --model_path (path to save your model) --image_size 192 --arch efficientnet_b3 --batch_size 64 --epochs 35 --initial_lr 0.00003 --loss BCEWithLogitsLoss --pos_weight 10 --oversampling 150 --undersampling 1 --use_metadata no
```

You can make predictions on a test set. Same as before, the logs will appear in the terminal if you do not use the nohup command with a `.out` file.
```
python predict.py --dataset ISIC_crops_train --data_dir (your data path) --model_path (your model path) --image_size 192 --arch efficientnet_b3 --batch_size 256 --use_metadata no
```

In case you do not want to train a model the best model with and without metadata are on the `./models/` folder of this repo.

## Link to the thesis
Link to the Bachelor's thesis (TFG): soon...
