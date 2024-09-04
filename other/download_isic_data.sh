# 2019 data - https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/164910
# 2020 data - https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/164092

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

# Install Kaggle API to download datasets
pip install kaggle

# isic_data path will be created inside the path provided
isic_data_dir="${dir}/isic_data"
# mkdir -p "$isic_data_dir"
cd "$isic_data_dir"

# Input size available: 128, 192, 256, 384, 512, 768, 1024

for input_size in 192
do
  kaggle datasets download -d cdeotte/jpeg-isic2019-${input_size}x${input_size}
  kaggle datasets download -d cdeotte/jpeg-melanoma-${input_size}x${input_size}
  unzip -q jpeg-melanoma-${input_size}x${input_size}.zip -d jpeg-melanoma-${input_size}x${input_size}
  unzip -q jpeg-isic2019-${input_size}x${input_size}.zip -d jpeg-isic2019-${input_size}x${input_size}
  rm jpeg-melanoma-${input_size}x${input_size}.zip jpeg-isic2019-${input_size}x${input_size}.zip
done

# External data - 70k images dataset - Uncomment 3 lasts lines to download
# Discussion - https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155859
# Kernel used - https://www.kaggle.com/code/shonenkov/merge-external-data

# kaggle datasets download -d shonenkov/melanoma-merged-external-data-512x512-jpeg
# unzip melanoma-merged-external-data-512x512-jpeg.zip -d melanoma-merged-external-data-512x512-jpeg
# rm melanoma-merged-external-data-512x512-jpeg.zip