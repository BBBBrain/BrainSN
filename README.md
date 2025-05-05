# BrainSN Project
## Overview
This repository contains the implementation of BrainSN (Brain States Network), a foundational fMRI model for representing continuous brain state information. The project provides a robust and scalable solution for model pretraining and downstream tasks analysis.
## Architecture
<img src="https://github.com/BBBBrain/BrainSN/blob/main/image/framework.png" width="500">

## Data preparation

We recommend reading and using the official website introduction and preprocessing process of each dataset below, and then using the brain atlas to calculate the final usable data. Here we use the [1000Parcels_Yeo2011_17Networks](https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Yan2023_homotopic/parcellations/MNI/yeo17/1000Parcels_Yeo2011_17Networks_FSLMNI152_2mm.nii.gz) , but you can choose different sizes and types for pre-training. All fMRI datasets underwent the HCP’s generic fMRI volume minimal preprocessing pipeline and were registered to the MNI152 common space. This
was followed by linear detrending, filtering (0.01 ∼ 0.1HZ in resting-state data and 0.01 ∼ 0.15HZ in natural stimulus-state) and normalization procedures.
#### [HCP-YA Resting, Movie](https://www.humanconnectome.org/study/hcp-young-adult/document/extensively-processed-fmri-data-documentation)
#### [HCP-Aging Resting](https://www.humanconnectome.org/study/hcp-lifespan-aging)
#### [Narratives](https://fcon_1000.projects.nitrc.org/indi/retro/Narratives.html)
#### [ABCD Resting](https://abcdstudy.org/)

## Installation
#### Prerequisites
Python 3.8 or higher  

pip or conda for package management
#### Steps
##### 1. Clone the repository:
   ```bash
   git clone https://github.com/BBBBrain/BrainSN.git
   
   cd train
```
##### 2. Install dependencies:
   ```bash
   pip install -r requirements.txt
```
## Usage
### Training
##### To train the model:  
```bash
sh scripts/train_BrainSN.sh
```
###### Edit train_BrainSN.sh to adjust hyperparameters, dataset paths, or other settings.
### Inference
##### To run inference with a pre-trained model:
```bash
python inference.py 
```
###### Edit inference.py  to adjust hyperparameters, pretrained model and dataset paths.
### Downstream tasks tests
In the downstream folder, there are training and testing codes for different downstream tasks. Please select according to different needs and modify the model and data file paths in the code according to the specific task.
### Pretrained model 
The pre-trained model weight file is too large to upload and is being uploaded through other channels. Please refer to the Pretrained foder for more details.
### Analysis
For visulization of latent embedding, please use tsne.py under the analysis folder.
### Contact
For questions or support, open an issue.
## License
This project is licensed under the MIT License. See the LICENSE file for details.
