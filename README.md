# Kvasir-PolypNet - Polyp Classification Project
Please find our detailed report here: [Report](report/Kvasir-PolypNet-Report.pdf)
## Project Overview

This repository contains code and documentation for a classification project as part of the *Data Science in Health* course at Zurich University of Applied Sciences (ZHAW). The goal is to develop a deep learning classifier using PyTorch to distinguish between *normal-cecum* and *polyps* in medical images. The dataset used is the Clahe Preprocessed Medical Imaging Dataset. 

## [Dataset: Clahe Preprocessed Medical Imaging Dataset](https://www.kaggle.com/datasets/heartzhacker/n-clahe)

This dataset is based on the widely used Kvasir Dataset, specifically focusing on two classes: **normal-cecum** and **polyps**. Several preprocessing techniques and augmentations were applied to enhance its utility for medical imaging classification. For more information about the dataset, please visit its kaggle page: https://www.kaggle.com/datasets/heartzhacker/n-clahe

### Key Features:
- **Data Augmentation:** Flipping, rotating, and mirroring were applied to increase dataset diversity and improve model generalization.
- **Incremental Learning Support:** The dataset is divided into three training sets for incremental learning experiments.
- **CLAHE Preprocessing:** All images have been enhanced using *Contrast Limited Adaptive Histogram Equalization (CLAHE)* to improve contrast and aid in detecting abnormalities.

## Results
<img width="570" alt="cm" src="https://github.com/user-attachments/assets/533b6d8b-2d7e-4712-8e87-2742eae454e6" />
Accuracy: 92.3%

## Instructions
1. Install all dependencies in requirements.txt
2. Run the desired script


## Contributors

- **Nils** - Zurich University of Applied Sciences (ZHAW): gaempnil@students.zhaw.ch
- **Jan** - Zurich University of Applied Sciences (ZHAW): dalgajan@students.zhaw.ch

For any questions or collaborations, feel free to reach out!


