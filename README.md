# GAN-for-MRI-Brian-Images
Author: Lok Yee Joey Cheung 

## Project Overview
This project leverages Generative Adversarial Networks (GANs) to generate realistic MRI brain images. By training on a dataset of MRI slices, the model aims to produce high-quality, synthetic brain images that can potentially aid in medical imaging researches.

## Principle of GAN in MRI Brain Image Generation 
Generative Adversarial Networks (GANs) are a class of machine learning frameworks that have shown significant promise in generating realistic data, such as images. In the context of generating MRI brain images, GANs can be used to create synthetic, high-quality MRI images from given input data. The process of generating MRI brain images with GANs involves two main components: the Generator and the Discriminator, which work together in a competitive process to improve the quality of generated images over time.

## OASIS dataset
The OASIS (Open Access Series of Imaging Studies) dataset is a widely used collection of neuroimaging data designed to facilitate research in the field of Alzheimer's disease (AD), dementia, and aging-related brain changes. The dataset provides a valuable resource for the development and evaluation of machine learning models, particularly in the areas of brain image segmentation, disease classification, and longitudinal studies.

## Usage

### Directory Structure
```
|-- gan.py                                              # Source code for model components
|-- keras_png_slices_data/keras_png_slices_test/        # Folder containing test dataset
|-- keras_png_slices_data/keras_png_slices_train/       # Folder containing train dataset
|-- keras_png_slices_data/keras_png_slices_validate/    # Folder containing validation dataset
|-- README.md                                           # Project documentation
|-- images/                                             # Folder for images and loss plot
```
### Implementation
1. Clone the Repository
```
git clone <repository_url>
cd <repository_folder>
```

2. Install Dependencies Run the following command to install the required libraries
```
pip install pandas pillow torch torchvision tqdm matplotlib
```

3. Explore the GAN implementation
``` 
python gan.py
```

## Training
Below shows the generator and discriminator loss during training for 50 epochs: 
![Alt Text](images/plot_final_50_sch.png)

## Generated Images
Below are some generated brain samples: 
![Alt Text](images/final0_50_sch.png)
![Alt Text](images/final1_50_sch.png)
![Alt Text](images/final2_50_sch.png)
![Alt Text](images/final3_50_sch.png)