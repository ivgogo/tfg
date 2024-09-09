# ================ Required imports & libraries ================

# Torch
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch

# Basic libraries
import pandas as pd
import numpy as np
import random
import cv2
import os

# Albumentations --> augmentations library for pytorch
from albumentations.core.transforms_interface import ImageOnlyTransform # create custom albumentations modules
import albumentations

# Sklearn
from sklearn.model_selection import train_test_split

# Custom dataset for the project       
class Custom_Dataset:
    def __init__(self, df, mode, resize, metadata_features, augmentations=None, oversample_ratio=0, undersample_ratio=0):
        
        self.df = df
        self.mode = mode
        self.resize = resize
        self.metadata_features = metadata_features
        self.augmentations = augmentations
        self.oversample_ratio = oversample_ratio
        self.undersample_ratio = undersample_ratio

        self.use_metadata = metadata_features is not None 

        # Positives and negatives cases for OS and US
        self.positive_cases = self.df[self.df['target'] == 1].reset_index(drop=True)
        self.negative_cases = self.df[self.df['target'] == 0].reset_index(drop=True)

        # Oversampling process if mode = train and OS_ratio > 1
        if self.mode == "train" and self.oversample_ratio > 1:
            self.positive_cases = pd.concat([self.positive_cases] * self.oversample_ratio, axis=0).reset_index(drop=True)
            self.df = pd.concat([self.negative_cases, self.positive_cases]).reset_index(drop=True)

        # Undersampling process
        if self.undersample_ratio > 0:
            self.negative_cases = self.negative_cases.sample(n=undersample_ratio*len(self.positive_cases), random_state=42)
            self.df = pd.concat([self.negative_cases, self.positive_cases]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['filepath']
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.resize is not None:
            image = cv2.resize(image, (self.resize, self.resize))
        
        image = np.array(image)
        
        # Apply augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        targets = self.df.iloc[item]['target']
        
        if self.use_metadata:
            return {
                "image": torch.tensor(image, dtype=torch.float),
                "metadata": torch.tensor(self.df.iloc[item][self.metadata_features], dtype=torch.float),
                "target": torch.tensor(targets, dtype=torch.float),
            }
        else:
            return {
                "image": torch.tensor(image, dtype=torch.float),
                "target": torch.tensor(targets, dtype=torch.float),
            }

# ========================== Read & Process ==========================

def read_data(dataset: str, data_dir: str, image_size: int, use_metadata: bool):
    
    metadata_features = None
    n_metadata_features = 0

    match dataset:
        case "ISIC_dermatoscopic_train":

            # Create data paths, read and assign filepaths
            isic_19_path = os.path.join(data_dir, f'isic_data/jpeg-isic2019-{image_size}x{image_size}')
            isic_20_path = os.path.join(data_dir, f'isic_data/jpeg-melanoma-{image_size}x{image_size}')

            df_train_19 = pd.read_csv(os.path.join(isic_19_path, 'train.csv'))
            df_train_19['filepath'] = df_train_19['image_name'].apply(lambda x: os.path.join(isic_19_path, 'train', f'{x}.jpg'))
                
            df_train_20 = pd.read_csv(os.path.join(isic_20_path, 'train.csv'))
            df_train_20['filepath'] = df_train_20['image_name'].apply(lambda x: os.path.join(isic_20_path, 'train', f'{x}.jpg'))

            df_train_19['filepath'] = df_train_19['filepath'].astype(str)
            df_train_20['filepath'] = df_train_20['filepath'].astype(str)

            if use_metadata:
                # Unify anatom_site_general_challenge in only a df
                concat = pd.concat([df_train_19['anatom_site_general_challenge'], df_train_20['anatom_site_general_challenge']], ignore_index=True)

                # One-hot enoce anatom_site_general_challenge
                dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')

                # Return it to its own original df
                df_train_19 = pd.concat([df_train_19.reset_index(drop=True), dummies.iloc[:df_train_19.shape[0]].reset_index(drop=True)], axis=1)
                df_train_20 = pd.concat([df_train_20.reset_index(drop=True), dummies.iloc[df_train_19.shape[0]:df_train_19.shape[0] + df_train_20.shape[0]].reset_index(drop=True)], axis=1)

                # Map sex to 0 and 1
                for df in [df_train_19, df_train_20]:
                    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
                    df['sex'] = df['sex'].fillna(-1)

                # Normalize age with max age on the whole df
                for df in [df_train_19, df_train_20]:
                    df['age_approx'] /= df['age_approx'].max()
                    df['age_approx'] = df['age_approx'].fillna(0)
                    df['patient_id'] = df['patient_id'].fillna(0)

                # Create n_images for each patient --> + images = + moles and it can be a signal to + probabilities of melanoma
                for df in [df_train_19, df_train_20]:
                    df['n_images'] = df.patient_id.map(df.groupby(['patient_id']).image_name.count())
                    df.loc[df['patient_id'] == -1, 'n_images'] = 1
                    df['n_images'] = np.log1p(df['n_images'].values)

                # Image size can be a signal of bigger lesions --> more probs of malignant
                for df in [df_train_19, df_train_20]:
                    train_images = df['filepath'].values
                    train_sizes = np.zeros(train_images.shape[0])
                    for i, img_path in enumerate(train_images):
                        train_sizes[i] = os.path.getsize(img_path)
                    df['image_size'] = np.log(train_sizes)

                df = pd.concat([df_train_19, df_train_20]).reset_index(drop=True)

                # Train val split of 80/20
                malignants = df[df['target'] == 1]
                benign = df[df['target'] == 0]

                malignant_train, malignant_val = train_test_split(malignants, test_size=0.2, random_state=42)
                benign_train, bening_val = train_test_split(benign, test_size=0.2, random_state=42)

                train_df = pd.concat([benign_train, malignant_train]).reset_index(drop=True)
                val_df = pd.concat([bening_val, malignant_val]).reset_index(drop=True)

                train_df['mode'] = 'train'
                val_df['mode'] = 'val'

                df = pd.concat([train_df, val_df]).reset_index(drop=True)

                # Group up metadata_features
                metadata_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df.columns if col.startswith('site_')]
                n_metadata_features = len(metadata_features)

            else:
                df = pd.concat([df_train_19, df_train_20]).reset_index(drop=True)

        case "ISIC_dermatoscopic_test":
            
            # Create data paths, read and assign filepaths
            isic_20_path = os.path.join(data_dir, f'isic_data/jpeg-melanoma-{image_size}x{image_size}')

            df = pd.read_csv(os.path.join(isic_20_path, 'test.csv'))
            df['filepath'] = df['image_name'].apply(lambda x: os.path.join(isic_20_path, 'test', f'{x}.jpg'))

            df['filepath'] = df['filepath'].astype(str)

            if use_metadata:
                # One-hot enoce anatom_site_general_challenge
                dummies = pd.get_dummies(df['anatom_site_general_challenge'], dummy_na=True, dtype=np.uint8, prefix='site')

                # Concat original df with dummies generated
                df = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

                # Map sex to 0 and 1
                df['sex'] = df['sex'].map({'male': 1, 'female': 0})
                df['sex'] = df['sex'].fillna(-1)

                # Normalize age with max age on the whole df
                df['age_approx'] /= df['age_approx'].max()
                df['age_approx'] = df['age_approx'].fillna(0)
                df['patient_id'] = df['patient_id'].fillna(0)

                # Create n_images for each patient --> + images = + moles and it can be a signal to + probabilities of melanoma
                df['n_images'] = df.patient_id.map(df.groupby(['patient_id']).image_name.count())
                df.loc[df['patient_id'] == -1, 'n_images'] = 1
                df['n_images'] = np.log1p(df['n_images'].values)

                # Image size can be a signal of bigger lesions --> more probs of malignant
                test_images = df['filepath'].values
                test_sizes = np.zeros(test_images.shape[0])
                for i, img_path in enumerate(test_images):
                    test_sizes[i] = os.path.getsize(img_path)
                df['image_size'] = np.log(test_sizes)

                # Group up metadata_features
                metadata_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df.columns if col.startswith('site_')]
                n_metadata_features = len(metadata_features)

        case "ISIC_crops_train":
            
            # Create data paths, read and assign filepaths
            isic_24_path = os.path.join(data_dir, 'isic-2024-challenge')

            df = pd.read_csv(os.path.join(isic_24_path, 'train-metadata.csv'))
            df['filepath'] = df['isic_id'].apply(lambda x: os.path.join(isic_24_path, 'train-image/image', f'{x}.jpg'))

            if use_metadata:
                # One hot encode anatom_site_general
                dummies = pd.get_dummies(df['anatom_site_general'], dummy_na=True, dtype=np.uint8, prefix='site')

                # Concat it with original df
                df = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

                # Map sex to 0 and 1
                df['sex'] = df['sex'].map({'male': 1, 'female': 0})
                df['sex'] = df['sex'].fillna(-1)

                # Normalize age with max age on the whole df
                df['age_approx'] /= df['age_approx'].max()
                df['age_approx'] = df['age_approx'].fillna(0)
                df['patient_id'] = df['patient_id'].fillna(0)

                # Create n_images for each patient --> + images = + moles and it can be a signal to + probabilities of melanoma
                df['n_images'] = df.patient_id.map(df.groupby(['patient_id']).isic_id.count())
                df.loc[df['patient_id'] == -1, 'n_images'] = 1
                df['n_images'] = np.log1p(df['n_images'].values)

                # Image size can be a signal of bigger lesions --> more probs of malignant
                train_images = df['filepath'].values
                train_sizes = np.zeros(train_images.shape[0])
                for i, img_path in enumerate(train_images):
                    train_sizes[i] = os.path.getsize(img_path)
                df['image_size'] = np.log(train_sizes)

                # Train val split of 80/20 in case we want to train with this dataset
                malignants = df[df['target'] == 1]
                benign = df[df['target'] == 0]

                malignant_train, malignant_val = train_test_split(malignants, test_size=0.2, random_state=42)
                benign_train, bening_val = train_test_split(benign, test_size=0.2, random_state=42)

                train_df = pd.concat([benign_train, malignant_train]).reset_index(drop=True)
                val_df = pd.concat([bening_val, malignant_val]).reset_index(drop=True)

                train_df['mode'] = 'train'
                val_df['mode'] = 'val'

                df = pd.concat([train_df, val_df]).reset_index(drop=True)

                # Group up metadata_features
                metadata_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df.columns if col.startswith('site_')]
                n_metadata_features = len(metadata_features)

        case "iToBoS":

            # Create paths and read csvs
            uq_path = os.path.join(data_dir, 'iToBoS_internal_ISIC_2024/UQ')
            fcrb_path = os.path.join(data_dir, 'iToBoS_internal_ISIC_2024/FCRB')

            columns_of_interest = ['filename', 'patient_id', 'sex', 'age', 'anatom_site_general', 'diagnosis']
            
            uq_df = pd.read_csv(os.path.join(uq_path, 'metadata_revised.csv'), low_memory=False)
            fcrb_df = pd.read_csv(os.path.join(fcrb_path, 'metadata_revised.csv'), low_memory=False)

            # Take columns of interest only
            uq_df = uq_df[columns_of_interest]
            fcrb_df = fcrb_df[columns_of_interest]

            # take filename and create filepath
            uq_df['filename'] = uq_df['filename'].astype('string')
            fcrb_df['filename'] = fcrb_df['filename'].astype('string')

            uq_df['filepath'] = uq_df.apply(lambda row: os.path.join(uq_path, 'crops', str(row['patient_id']), row['filename']), axis=1)
            fcrb_df['filepath'] = fcrb_df.apply(lambda row: os.path.join(fcrb_path, 'crops', str(row['patient_id']), row['filename']), axis=1)

            # unify dfs
            df = pd.concat([uq_df, fcrb_df]).reset_index(drop=True)

            # create target column and filter
            df['diagnosis'].fillna('None', inplace=True)
            df['target'] = 0

            # Create target column and make the train/val split 80/20
            malignant_strings = ['Melanoma', 'Basal cell carcinoma', 'Squamous cell carcinoma']

            for malignancy in malignant_strings:
                df.loc[df['diagnosis'].str.contains(malignancy, case=False), 'target'] = 1

            malignants = df[df['target'] == 1]
            benign = df[df['target'] == 0]

            malignant_train, malignant_val = train_test_split(malignants, test_size=0.2, random_state=42)
            benign_train, bening_val = train_test_split(benign, test_size=0.2, random_state=42)

            train_df = pd.concat([benign_train, malignant_train]).reset_index(drop=True)
            val_df = pd.concat([bening_val, malignant_val]).reset_index(drop=True)

            train_df['mode'] = 'train'
            val_df['mode'] = 'val'

            df = pd.concat([train_df, val_df]).reset_index(drop=True)

            # List of filepaths that do exist
            filepaths_exist = df['filepath'].apply(os.path.exists)

            # List of filepaths that do not exist
            non_existing_paths_df = df[~filepaths_exist].copy()

            # Paths
            uq_path = '/home/falcon/student3/tfg_ivan/data/iToBoS_internal_ISIC_2024/UQ/crops'
            fcrb_path = '/home/falcon/student3/tfg_ivan/data/iToBoS_internal_ISIC_2024/FCRB/crops'

            # This process is due to the folder structure the data has, may seem complex but is very simple
            # It's a process just to create some filepath that are on some folders that do not follow the predefined structure like the rest in order to use all images and not leave any behind
            def fix_filepath(row, base_path):
                # try
                if 'UQ' in base_path:
                    corrected_paths = [
                        os.path.join(base_path, 'tags', row['filename']),
                        os.path.join(base_path, 'ambiguous', row['filename'])
                    ]
                # try 2
                elif 'FCRB' in base_path:
                    corrected_paths = [
                        os.path.join(base_path, 'tags', row['filename'])
                    ]
                else:
                    return None
                
                for path in corrected_paths:
                    if os.path.exists(path):
                        return path
                return None

            # Finally
            non_existing_paths_df['source'] = non_existing_paths_df['filepath'].apply(lambda x: 'UQ' if uq_path in x else 'FCRB')

            # create final filepaths
            non_existing_paths_df['corrected_filepath'] = non_existing_paths_df.apply(
                lambda row: fix_filepath(row, uq_path if row['source'] == 'UQ' else fcrb_path), axis=1
            )

            # Final fix
            for index, row in non_existing_paths_df.iterrows():
                if pd.notnull(row['corrected_filepath']):
                    df.loc[df['filepath'] == row['filepath'], 'filepath'] = row['corrected_filepath']
            
            if use_metadata:

                # One hot encode anatom_site_general
                dummies = pd.get_dummies(df['anatom_site_general'], dummy_na=True, dtype=np.uint8, prefix='site')

                # Concat dummies to original df
                df = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

                # Map sex to 0 and 1
                df['sex'] = df['sex'].map({'male': 1, 'female': 0})
                df['sex'] = df['sex'].fillna(-1)

                # Normalize age with the max age on the whole dataset
                df['age'] = pd.to_numeric(df['age'], errors='coerce')
                df['age'] /= df['age'].max()
                df['age'] = df['age'].fillna(0)
                df['patient_id'] = df['patient_id'].fillna(0)

                # Create n_images for each patient --> + images = + moles and it can be a signal to + probabilities of melanoma
                df['n_images'] = df.patient_id.map(df.groupby(['patient_id']).filename.count())
                df.loc[df['patient_id'] == -1, 'n_images'] = 1
                df['n_images'] = np.log1p(df['n_images'].values)

                # Image size can be a signal of bigger lesions --> more probs of malignant
                train_images = df['filepath'].values
                train_sizes = np.zeros(train_images.shape[0])
                for i, img_path in enumerate(train_images):
                    train_sizes[i] = os.path.getsize(img_path)
                df['image_size'] = np.log(train_sizes)

                # Group up metadata_features
                metadata_features = ['sex', 'age', 'n_images', 'image_size'] + [col for col in df.columns if col.startswith('site_')]
                n_metadata_features = len(metadata_features)

        case "sana_external":
            
            # Create path
            sana_external_path = os.path.join(data_dir, 'ProveAI-TTA')

            # Read and load csv
            df = pd.read_csv(os.path.join(sana_external_path, 'test.csv'))
            df['filepath'] = df['isic_id'].apply(lambda x: os.path.join(sana_external_path, f'{x}.jpg'))

            # Create target column
            df['target'] = df['benign_malignant'].apply(lambda x: 1 if x == 'malignant' else 0)

            if use_metadata:

                # One hot encode anatom_site_general_challenge
                dummies = pd.get_dummies(df['anatom_site_general_challenge'], dummy_na=True, dtype=np.uint8, prefix='site')

                # Concat it to original df
                df = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

                # Map sex to 0 and 1
                df['sex'] = df['sex'].map({'male': 1, 'female': 0})
                df['sex'] = df['sex'].fillna(-1)

                # Normalize age
                df['age_approx'] /= df['age_approx'].max()
                df['age_approx'] = df['age_approx'].fillna(0)
                df['patient_id'] = df['patient_id'].fillna(0)

                # Create n_images for each patient --> + images = + moles and it can be a signal to + probabilities of melanoma
                df['n_images'] = df.patient_id.map(df.groupby(['patient_id']).isic_id.count())
                df.loc[df['patient_id'] == -1, 'n_images'] = 1
                df['n_images'] = np.log1p(df['n_images'].values)

                # # Image size can be a signal of bigger lesions --> more probs of malignant
                train_images = df['filepath'].values
                train_sizes = np.zeros(train_images.shape[0])
                for i, img_path in enumerate(train_images):
                    train_sizes[i] = os.path.getsize(img_path)
                df['image_size'] = np.log(train_sizes)

                 # Group up metadata_features
                metadata_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df.columns if col.startswith('site_')]
                n_metadata_features = len(metadata_features)

        case _:
            raise NotImplementedError("Not a dataset! - Error on dataset name in ssh file or read_data() function")

    if use_metadata:
        return df, metadata_features, n_metadata_features
    else: 
        metadata_features = None
        n_metadata_features = 0
        return df, metadata_features, n_metadata_features

# ========================== Custom Augmentations ==========================

class CustomHairAugmentation(ImageOnlyTransform):
    """
    Impose an image of a hair to an image

    Args: 
        hairs (int): number of hairs to add --> (el dataset de kaggle es de 5 solo por lo tanto si > 5 se repetiran)
        hairs_folder (str): path to the folder of the hair images

    Hair dataset (5 hair images):
        https://www.kaggle.com/datasets/nroman/melanoma-hairs

    References:
    |   https://www.kaggle.com/code/nroman/melanoma-pytorch-starter-efficientnet
    |   https://github.com/Masdevallia/3rd-place-kaggle-siim-isic-melanoma-classification/blob/master/kaggle_notebooks/melanoma-classification-model-training.ipynb
    |   https://github.com/albumentations-team/albumentations/issues/938
    |   https://github.com/albumentations-team/albumentations/pull/490/files

    """

    def __init__(self, hairs: int=5, hairs_folder: str="", always_apply=False, p=0.5):
        super(CustomHairAugmentation, self).__init__(always_apply=always_apply, p=p)
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def apply(self, img, **params):
        
        n_hairs = random.randint(0, self.hairs)

        if not n_hairs:
            return img

        height, width, _ = img.shape
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))

            hair = cv2.resize(hair, (height, width))

            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            # Creating a mask and inverse mask
            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of hair in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of hair from hair image.
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)
            hair_fg = hair_fg.astype(np.float32) # hot fix?

            # Put hair in ROI and modify the target image
            dst = cv2.add(img_bg, hair_fg)

            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst
                
        return img
    
class CustomHairDrawingAugmentation(ImageOnlyTransform):
    """
    Draw a random number of pseudo hairs

    Args:
        hairs (int): maximum number of hairs to draw
        width (tuple): possible width of the hair in pixels
    """

    def __init__(self, hairs:int = 4, width:tuple = (1, 2), always_apply=False, p=0.5):
        super(CustomHairDrawingAugmentation, self).__init__(always_apply=always_apply, p=p)
        self.hairs = hairs
        self.width = width

    def apply(self, img, **params):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        if not self.hairs:
            return img
        
        width, height, _ = img.shape
        
        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0, width), random.randint(0, height // 2))
            # The end of the line 
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)  # color of the hair. Black.
            cv2.line(img, origin, end, color, random.randint(self.width[0], self.width[1]))
        
        return img

# ========================== Data Augmentation ==========================

def get_augmentations(mode: str, image_size: int):
    
    # Data augmentations provided by Albumentations library

    # mean & std values:
    # https://www.kaggle.com/code/abhishek/melanoma-detection-with-pytorch
    # https://www.youtube.com/watch?v=WaCFd-vL4HA
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    match mode:
        case "train":
            transforms_train = albumentations.Compose([
                albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
                CustomHairAugmentation(hairs=5, hairs_folder="/home/falcon/student3/tfg_ivan/data/false_hairs", always_apply=False, p=0.5),
                CustomHairDrawingAugmentation(hairs=4, width=(1,2), always_apply=False, p=0.5),
                albumentations.Transpose(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
                albumentations.RandomBrightness(limit=0.2, p=0.75),
                albumentations.RandomContrast(limit=0.2, p=0.75),
                albumentations.OneOf([
                    albumentations.MotionBlur(blur_limit=5),
                    albumentations.MedianBlur(blur_limit=5),
                    albumentations.GaussianBlur(blur_limit=5),
                    albumentations.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.7),

                albumentations.OneOf([
                    albumentations.OpticalDistortion(distort_limit=1.0),
                    albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                    albumentations.ElasticTransform(alpha=3),
                ], p=0.7),

                albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                albumentations.Resize(image_size, image_size),
                albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
            ])

            return transforms_train
        case "val":
            transforms_val = albumentations.Compose([
                albumentations.Resize(image_size, image_size),
                albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
            ])

            return transforms_val
        case "test":
            transforms_test = albumentations.Compose([
                albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
            ])

            return transforms_test
        case _:
            raise NotImplementedError("Not a valid mode - get_augmentations() function")