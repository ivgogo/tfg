CUDA_VISIBLE_DEVICES=MIG-fc49a16c-f2d9-52a0-a89c-85d123b90f28 \
python3 /home/falcon/student3/tfg_ivan/tfg_new/nueva_pipeline/train.py \
--wandb_entity ivgogo --wandb_project tfg \
--wandb_api_key d8e75ee521acf8be7eb7fbbe666ad1e068a56636 \
--dataset iToBoS --data_dir /home/falcon/student3/tfg_ivan/data --model_path /home/falcon/student3/tfg_ivan/models \
--image_size 192 \
--arch efficientnet_b3 \
--batch_size 64 --epochs 35 --initial_lr 0.00003 \
--loss BCEWithLogitsLoss --pos_weight 10 \
--oversampling 150 --undersampling 1 \
--use_metadata no