CUDA_VISIBLE_DEVICES=MIG-fc49a16c-f2d9-52a0-a89c-85d123b90f28 \
python3 /home/falcon/student3/tfg_ivan/tfg_new/nueva_pipeline/predict.py \
--dataset ISIC_crops_train --data_dir /scratch/itobos/ISIC \
--model_path /home/falcon/student3/tfg_ivan/models/iToBoS_35e_192_64bs_3e-05lr_yes_metadata_BCEWithLogitsLoss_OSx150_USRatio_1:1.pth \
--misc_path '/home/falcon/student3/tfg_ivan/data' \
--image_size 192 \
--arch efficientnet_b3 \
--batch_size 256 \
--use_metadata yes