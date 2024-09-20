#!/bin/bash

# Contar subdirectorios dentro ya de un directorio
# ls | wc -l

# Dar permisos de ejecución a un script shell para que le ejecución en terminal sea --> ./script.sh
# chmod +x script.sh

# IMPORTANT DISCLAIMER:
# This script was created to count number of patients and lesions using the original data structure (not all images in one single folder)

# Verify path is given as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path>"
    exit 1
fi

# Path provided
dir="$1"

# Verify folder path exists
if [ ! -d "$dir" ]; then
    echo "This directory '$dir' does not exist."
    exit 1
fi

# Count subdirectories and images
total_subdirs=0
total_images=0

echo "---------------------------------------------------------------------------"
echo "Folder '$dir'"
echo "---------------------------------------------------------------------------"

# Iterate over subdirectories
for subdir in "$dir"/*/; do
    # Exclude "tags" directory
    # if [ "$(basename "$subdir")" != "tags" ]; then
        num_images=$(find "$subdir" -maxdepth 1 -type f -name "*.jpg" | wc -l)
        subdir_name=$(basename "$subdir")
        # echo "Subdirectory '$subdir_name' contains $num_images images."
        ((total_subdirs++))
        ((total_images+=num_images))
    # fi
done

# echo "---------------------------------------------------------------------------"
echo "Total subdirectories (pacient explorations): $total_subdirs"
echo "Total images (lesions): $total_images"
echo "---------------------------------------------------------------------------"
