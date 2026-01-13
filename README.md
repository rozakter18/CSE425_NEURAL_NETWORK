Unsupervised Learning Project: VAE for Hybrid Language Music Clustering

## Overview
This project implements unsupervised learning for music clustering using Variational Autoencoders (VAE) with MFCC audio features. Three progressive tasks build from basic to advanced implementations:

- **Easy Task**: Basic VAE with K-Means clustering and PCA baseline comparison
- **Medium Task**: Enhanced VAE with hybrid audio+lyrics features and multiple clustering algorithms
- **Hard Task**: Beta-VAE/CVAE with multi-modal fusion, advanced metrics, and baseline comparisons

## Setup
1. Clone or download the project.
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure data is placed in the correct directories as per the structure below.

## Usage

### Easy Task
1. Run VAE training: `python src/easy_vae.py`
2. Perform clustering: `python src/easy_clustering.py`
3. Evaluate results: `python src/easy_evaluation.py`

### Medium Task
1. Run enhanced VAE training: `python src/medium_vae.py`
2. Perform clustering: `python src/medium_clustering.py`
3. Evaluate results: `python src/medium_evaluation.py`

### Hard Task
1. Run complete multimodal analysis: `python src/hard_task.py`

Results will be saved to the `results/` directory.

**Dataset Link:**  
https://drive.google.com/drive/folders/1368ujLo65p4Y1ZvBVZ45q550MeemtySp?usp=drive_link
