# TFG - Early detection of melanoma using artificial intelligence (AI) ||| PENDIENTE DE ACABAR - Faltan un montón de cosas

This repository includes the code of my Biomedical Engineering Degree final project.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Software](#software)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
....

## Data
You can download ISIC data executing **download_isic_data.sh** (found in utilities folder). iToBoS data is not yet public although will be publicly released in the future.

To execute the script just open a terminal and execute:

```
./download_isic_data.sh data_folder_path
```

You must have given execution permissions to the script via:

```
chmod +x download_isic_data.sh
```

The script will download and unzip resized 2020 and 2019 data (including 2017 and 2018) of all available sizes by Chris Deotte.

## Software

### Software used during the project:
-  Python: 3.10.12
-  Nvidia Drivers: 525.147.05
-  CUDA: 11.5.119
-  cuDNN: 8.9.2

The training of the models was done using a **Nvidia A100 80G**, GPU is highly recommended to execute this project in order to have acceptable training times.

## Installation
You can install all necessary requeriments and dependencies by using:

```
pip install -r requreriments.txt
```

## Usage
Training commands of the best model. Training time for the models ranges from 2 to 24 hours, depending on the GPU you use.

After training, the model will be saved in model path established in the configuration. If you do not want or can not use WandB, the training logs will appear in the terminal through the training session. You can use as well the nohup command with a `.out` file.
```
python train.py ....
```

You can make predictions on a test set. Same as before, the logs will appear in the terminal if you do not use the nohup command with a `.out` file.
```
python predict.py ....
```

In case you do not want to train a model the best model with and without metadata are on the `./models/` folder of this repo.

## Link to the thesis
Link to the Bachelor's thesis (TFG): soon...
