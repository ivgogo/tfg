# ================ Required imports & libraries ================

# Torch
from torch.utils.data import Subset, Dataset, DataLoader
import torch

# Basic libraries
import pandas as pd
import numpy as np
import argparse
import sys
import os

# Functions from other .py files
from models import SimpleEffnet
from datasets import Custom_Dataset, read_data, get_augmentations
from utils import metrics, predict

# ================ Prediction Script ================

# Test Configuration
def parse_args():
    parser = argparse.ArgumentParser(prog='Prediction Script')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--misc_path', type=str, required=True)
    parser.add_argument('--image_size', type=int, required=True)
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_metadata', type=str, default='no')

    args, _ = parser.parse_known_args()
    return args

def main():

    # ================ Device set-up ================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'======= Testing model on dataset: {args.dataset} | Device set on: {device} =======')   # we make sure we are running on gpu
    
    # ================ Dataset selection ================
    
    # Read df and meta in case we use it
    df, metadata_features, n_metadata_features = read_data(dataset=args.dataset, data_dir=args.data_dir, image_size=args.image_size, use_metadata=True if args.use_metadata == 'yes' else False)
    
    # Copy of original targets of the dataset
    targets = df['target']

    print(f'Testing images: {len(df)}')

    # ================ Create DataLoaders ================
    
    test_dataset = Custom_Dataset(df=df, mode="test", resize=args.image_size, metadata_features=metadata_features, augmentations=get_augmentations(mode="test", image_size=args.image_size), oversample_ratio=0, undersample_ratio=0)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # ================ Load Model ================

    # Load model from the path established in the config
    model = SimpleEffnet(arch=args.arch, n_metadata_features=n_metadata_features)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Predict
    predictions = predict(model, test_loader, device)
    
    # Extract model name in order to save ROC curve, PR curve and confusion matrix info in order to plot them in a jupyter notebook
    model_path = args.model_path
    model_name = os.path.basename(model_path).replace('.pth', '')
    saving_path_plots = f'predict_{model_name}_on_{args.dataset}'

    # metrics
    auc, f1, fpr, tpr, sensibility, precision, recall, balanced_accuracy_score = metrics(targets=targets, predictions=predictions, path=os.path.join(args.misc_path, saving_path_plots)).values()

    # Print results
    print(f'‣ AUC: {auc}')
    print(f'‣ Weighted F1 Score: {f1}')
    print(f'‣ Sensibility: {sensibility}')
    print(f'‣ Balanced Accuracy Score: {balanced_accuracy_score}')

if __name__ == '__main__':
    # Configuration parameters for the predicting
    args = parse_args()

    # Launch main()
    main()