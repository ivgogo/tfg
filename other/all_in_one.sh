#!/bin/bash

# Main path
main_path="/home/falcon/student3/tfg_ivan/19_3_24_copy/Cropped_Lesion"

# Move all images inside subdirectories to the main path
find "$main_path" -mindepth 2 -type f -exec mv -t "$main_path" {} +

# Delete subdirectories
find "$main_path" -mindepth 1 -type d -exec rm -r {} +
