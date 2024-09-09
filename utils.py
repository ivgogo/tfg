# ================ Required imports & libraries ================

# Torch
import torch

# Basic libraries
import argparse
import pandas as pd
import numpy as np
import cv2
import sys
import os

# Sklearn metrics
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, balanced_accuracy_score, recall_score, precision_recall_curve, confusion_matrix
  
# Returns the image with a preprocessing to delete hair done (NOT USED ON THE PROJECT)
def remove_hair(image):

    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(17,17))
    
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    
    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
    
    return final_image

# Calculate validation metrics
def metrics(targets, predictions, path):

    # Predefined threshold that will be used to binarize probs to calculate f1 score and sensibility
    threshold = 0.8

    # Binarize predictions
    predictions_binarized = (np.array(predictions) > threshold).astype(int)

    # ROC AUC SCORE
    auc = roc_auc_score(targets, predictions)

    # Weighted F1 Score
    f1 = f1_score(targets, predictions_binarized, average='weighted')

    # ROC Curve
    fpr, tpr, _ = roc_curve(targets, predictions)

    # Sensibility
    sensibility = recall_score(targets, predictions_binarized)

    # Precision recall curve
    precision, recall, _ = precision_recall_curve(targets, predictions)

    # Balanced accuracy
    balanced_accuracy = balanced_accuracy_score(targets, predictions_binarized)

    # Only save curves for predict, training & val curves --> WandB
    if path is not None:
        
        # Save curves
        np.savez(path+'_curves.npz', fpr=fpr, tpr=tpr, precision=precision, recall=recall)

        # Create and save confusion matrix
        cm = confusion_matrix(targets, predictions_binarized)
        np.savez(path+'_confusion_matrix.npz', confusion_matrix=cm)

    metrics = {
        "auc": auc,
        "f1": f1,
        "fpr": fpr,
        "tpr": tpr,
        "sensibility": sensibility,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy_score": balanced_accuracy,
    }

    return metrics
    
# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    
    # Set model to training mode and initialize loss to 0
    model.train()
    running_loss = 0.0
    
    # Training loop
    for data in train_loader:
        
        # Inputs and labels to the GPU and metadata in case we use it
        inputs = data['image'].to(device)
        labels = data['target'].to(device).unsqueeze(1)
        
        if model.use_meta:
            metadata = data['metadata'].to(device)
            outputs = model(inputs, metadata)
        else:
            outputs = model(inputs)

        # Optimizer and loss update
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    return epoch_loss

# Validation function
def validate_model(model, val_loader, criterion, device):
    
    # Set model to evaluation mode and initialize loss to 0
    model.eval()
    running_loss = 0.0
    
    # Initialize labels arrays
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        
        # Val loop
        for data in val_loader:
            
            # Inputs and labels to the GPU and metadata in case we use it
            inputs = data['image'].to(device)
            labels = data['target'].to(device).unsqueeze(1)  
            
            if model.use_meta:
                metadata = data['metadata'].to(device)
                outputs = model(inputs, metadata)
            else:
                outputs = model(inputs)

            # Loss update
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
    
    # Loss
    epoch_loss = running_loss / len(val_loader.dataset)

    # We use custom metrics function to calculate all the metrics of our interest during the validation process
    val_metrics = metrics(targets=all_labels, predictions=all_outputs, path=None)

    return epoch_loss, val_metrics

# Build model's name function
def build_model_name (args):

    '''

    This function is used to build the name of the model
    I trained multiple models during the project timeline. 
    In order to differentiate them I built this function to not be changing names all time.
    It uses the different configuration parameters used to build the model's name.    
    
    '''

    building_path = f'{args.dataset}_{args.epochs}e_{args.image_size}_{args.batch_size}bs_{args.initial_lr}lr_{args.use_metadata}_metadata'
    model_extension = '.pth'

    match args.loss:
        case "BCEWithLogitsLoss":
            building_path += "_BCEWithLogitsLoss"
        case "FocalLoss":
            building_path += "_FocalLoss"
        case _:
            raise NotImplementedError("Error on loss function name, revise train.ssh file or build_model_name function!")
        
    if args.oversampling > 1:
        building_path += f'_OSx{args.oversampling}'
    
    if args.undersampling > 0:
        building_path += f'_USRatio_{args.undersampling}:1'

    return building_path+model_extension

# Predict function    
def predict(model, test_loader, device):
    
    # Model to evaluation mode and initialize array for all the outputs
    model.eval()
    all_outputs = []

    # Prediction loop
    with torch.no_grad():
        for data in test_loader:

            # Inputs to the model and metadata if we are using it
            inputs = data['image'].to(device)

            if model.use_meta:
                metadata = data['metadata'].to(device)
                outputs = model(inputs, metadata)
            else:
                outputs = model(inputs)

            # Stack probabilities
            probabilities = torch.sigmoid(outputs)
            all_outputs.extend(probabilities.cpu().numpy())
    
    return all_outputs