# ================ Required imports & libraries ================

# Torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

# Basic libraries
import pandas as pd
import numpy as np
import argparse
import wandb 
import sys
import os

# Functions from other .py files
from utils import train_model, validate_model, build_model_name
from datasets import Custom_Dataset, read_data, get_augmentations
from models import SimpleEffnet, FocalLoss

# EarlyStopping function from "Well That's Fantastic Machine Learning" library
from wtfml.utils import EarlyStopping

# Notifications for discord
from knockknock import discord_sender

# ================ Training Script ================

# Get notifications when the training session ends or the code breaks/stops for some reason
WEBHOOK = 'https://discord.com/api/webhooks/1222850874863259721/lO07KQi2l9jgc_mOte36kT-_9i_Rg9hvCqcYV5Fb0jpTtee3kcIWYh8uwP-DRczgkV-R'

# Train Session Configuration
def parse_args():
    parser = argparse.ArgumentParser(prog='Model Training Script')
    parser.add_argument('--wandb_entity', type=str, required=False)
    parser.add_argument('--wandb_project', type=str, required=False)
    parser.add_argument('--wandb_api_key', type=str, required=False)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_size', type=int, required=True)
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--initial_lr', type=float, required=True)
    parser.add_argument('--loss', type=str, required=True)
    parser.add_argument('--pos_weight', type=int, default=10)
    parser.add_argument('--oversampling', type=int, required=False)
    parser.add_argument('--undersampling', type=int, required=False)
    parser.add_argument('--use_metadata', type=str, default='no')

    args, _ = parser.parse_known_args()
    return args
    
@discord_sender(webhook_url=WEBHOOK)
def main():
    
    # ================ Wandb ================
    
    if WANDB_IS_ENABLED:
        # Initialization of wandb (Weights&Biases - AI platform to track run results)
        train_config = {
            "dataset":args.dataset,
            "image_size":args.image_size,
            "arch":args.arch,
            "batch_size":args.batch_size,
            "initial_lr":args.initial_lr,
            "epochs":args.epochs,
            "loss": args.loss,
            "pos_weight": args.pos_weight,
            "oversampling": args.oversampling,
            "undersampling": args.undersampling,
            "use_metadata": args.use_metadata,
        }

        os.environ['WANDB_API_KEY'] = args.wandb_api_key
        wandb.init(
            entity=args.wandb_entity, project=args.wandb_project, config=train_config
        )

    # ================ Device set-up ================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'======= Training Session Starting | Device set on: {device} =======')   # we make sure we are running on gpu
    
    # ================ Dataset selection & creation ================
   
    df, metadata_features, n_metadata_features = read_data(dataset=args.dataset, data_dir=args.data_dir, image_size=args.image_size, use_metadata=True if args.use_metadata == 'yes' else False)

    df_train = df[df['mode'] != 'val'].reset_index(drop=True)
    df_valid = df[df['mode'] == 'val'].reset_index(drop=True)

    train_dataset = Custom_Dataset(df=df_train, mode="train", resize=args.image_size, metadata_features=metadata_features, augmentations=get_augmentations(mode="train", image_size=args.image_size), oversample_ratio=args.oversampling, undersample_ratio=args.undersampling)

    val_dataset = Custom_Dataset(df=df_valid, mode="val", resize=args.image_size, metadata_features=metadata_features, augmentations=get_augmentations(mode="val", image_size=args.image_size), oversample_ratio=0, undersample_ratio=0)

    print(f'Training images: {len(train_dataset)} | Validation images: {len(val_dataset)}')

    # ================ Create DataLoaders ================
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=val_dataset, batch_size=(args.batch_size)*2, shuffle=False, num_workers=4)

    # ================ Build Model ================

    model = SimpleEffnet(n_metadata_features=n_metadata_features)
    model.to(device)

    # ================ Optimizer, Loss, LR Scheduler & Early Stopping ================
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)

    # Loss function selection depending on the config established on the .ssh file
    match args.loss:
        case "BCEWithLogitsLoss":
            pos_weight = torch.tensor([float(args.pos_weight)], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        case "FocalLoss":
            criterion = FocalLoss(alpha=1.0, gamma=2.0, logits=True)
        case _:
            raise NotImplementedError()

    # Scheduler set-up
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,         # number of epochs with no improvement after which lr will be reduced
        factor=0.05,        # factor by which new lr will be reduced --> new_lr = lr*factor
        threshold=0.01,     # minimum difference of improvement
        mode="max"
    )

    # Early stop set-up
    es = EarlyStopping(patience=7, mode="max")  # patience --> number of epochs without improvement after which the training will early stop

    # ================ Train/Validation Loop ================

    print(f'======= Starting Training & Validation Loop =======')
    print()

    # Create the model name using the parameters established on the configuration of the training session
    model_name = build_model_name(args)
    # model_name = model_name.replace('.pth', '')
    # model_name += '_effnetb0.pth'

    for epoch in range(args.epochs):
        
        print(f'[Epoch {epoch+1}/{args.epochs}]')

        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, metrics = validate_model(model, valid_loader, criterion, device)

        # unpack metrics
        auc, f1, fpr, tpr, sensibility, precision, recall, balanced_accuracy = metrics.values()

        scheduler.step(auc)

        es(auc, model, os.path.join(args.model_path, model_name))

        # Extract current learning rate from optimizer
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Train Loss: {train_loss} | Current lr: {current_lr} | Val Loss: {val_loss} - AUC: {auc} - F1: {f1} - Sensibility: {sensibility} - Balanced Accuracy: {balanced_accuracy}')
        print()

        wandb.log({"train_loss": train_loss,
                   "val_loss": val_loss, 
                   "val_auc": auc, 
                   "val_f1": f1,
                   "val_sensibility": sensibility,
                   "val_balanced_accuracy": balanced_accuracy,
                   "lr": current_lr,
                   "roc_curve": wandb.plot.line_series(
                        xs=fpr, ys=[tpr], keys=["ROC Curve"],
                        title="ROC Curve", xname="False Positive Rate"
                    ),
                    "precision_recall_curve": wandb.plot.line_series(
                        xs=recall, ys=[precision], keys=["Precision-Recall Curve"],
                        title="Precision-Recall Curve", xname="Recall"
                    )
                })
        
        if es.early_stop:
            print("Early stopping counter reached limit. Early stopping training session...")
            break    
    
    return f"Training Session Finished"

if __name__ == '__main__':
    
    # Configuration parameters for the training session
    args = parse_args()

    # Check if all arguments required for WandB initialitzation have been given to track the training session
    WANDB_IS_ENABLED = (args.wandb_entity and args.wandb_project and args.wandb_api_key) is not None

    # Launch main()
    main()
    