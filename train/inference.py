import os
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from datasets import load_from_disk, concatenate_datasets
from fancyimpute import IterativeImputer

from BrainSN_main import BrainSNForPretraining as pre_series


from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import seaborn as sns

from torch.utils.data import Dataset
from random import randint

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

import warnings
# warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TR = 0.72    
from scipy.stats import pearsonr    
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer


class MyDataset(Dataset):
    def __init__(self, data_folder,flag="train"):
        self.data_folder = data_folder
        self.filenames = []
        self.flag = flag
        
        
        file_list = os.listdir(self.data_folder)
        
        
        if self.flag == "train":
            datas = file_list[:]    
            for per_data in datas:
                self.filenames.append(os.path.join(self.data_folder, per_data))
        elif self.flag == "val":
            datas = file_list[:]  
            for per_data in datas:
                self.filenames.append(os.path.join(self.data_folder, per_data))
    
    def __len__(self):
        return len(self.filenames) 
    
    def __getitem__(self, index):
   
        data_label  = -1000  ### 静息态
     
        
        data_label  = None
     
        if "rfMRI" in self.filenames[index]:
            data_label = 0.72
        elif "tfMRI" in self.filenames[index]:
            data_label = 0.72
        elif "MOVIE" in self.filenames[index]:
            data_label = 1
        elif "alice" in self.filenames[index]:
            data_label = 2
        elif "HCA" in self.filenames[index]:   
            data_label = 0.8
        elif "lppHK" in self.filenames[index]:
            data_label = 2
        elif "NNDb" in self.filenames[index]:
            data_label = 1
        else:
            data_label = 1.5
      

        image = np.load(self.filenames[index])
      
       
        data1 = self.preprocess(image,data_label)
        return data1,data_label



    
    def preprocess(self, examples,tr):
        
        
        examples_o ={}
        label = 1 
        label = torch.tensor(label, dtype=torch.int64)
        
        lenth=90

        start_idx = randint(0, examples.shape[0] - lenth)  #########  100 -> timepoints  shape【0】  
        end_idx = start_idx + lenth
        signal_vector = examples[start_idx:end_idx,:].transpose() # (1559, 30, 200)
        
        T = TimeSeries(signal_vector[:,:], sampling_interval=2)
        F = FilterAnalyzer(T, ub=0.15, lb=0.01)
        # signal_vector4 =F.filtered_fourier.data
        # signal_vector4 = np.array([standardize(signal_vector4[i]) for i in range(signal_vector4.shape[0])])
        signal_vector4 = NormalizationAnalyzer(F.filtered_fourier).z_score.data  ##  filtered_fourier   filtered_boxcar
        
     
        signal_window = torch.tensor(signal_vector4, dtype=torch.float32)
        
        signal_window = signal_window.transpose(1, 0)

        # examples_o["signal_vectors"] = signal_window[1:-1,:]
        examples_o["signal_vectors"] = signal_window
        examples_o["signal_vectors1"] = signal_window
        examples_o["xyz_vectors"] = window_xyz_list
        # examples_o["xyz_vectors"] = None
        examples_o["label"] = torch.tensor(tr, dtype=torch.float32)

        

        return examples_o
    
    





model_series = pre_series.from_pretrained("##############").to(device) 


for param in model_series.parameters():
    param.requires_grad = False
    
model_series.vit.embeddings.mask_ratio = 0.3
model_series.vit.embeddings.config.mask_ratio = 0.3
torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)


def collate_fn(examples):
    signal_vectors = torch.stack(
        [example for example in examples["signal_vectors"]], dim=0
    )
    
    signal_vectors1 = torch.stack(
        [example for example in examples["signal_vectors1"]], dim=0
    )

    labels = torch.stack([example for example in examples["label"]])
    
    xyz_vectors = torch.stack([example for example in examples["xyz_vectors"] ])


    return {
        "signal_vectors": signal_vectors,
        "signal_vectors1": signal_vectors1,
        "xyz_vectors": xyz_vectors,
        "input_ids": signal_vectors,
        "labels": labels,
    }





data_folder = "#######"  
data_folder_test = "###################"
train_dataset = MyDataset(data_folder=data_folder,flag="train")
val_dataset = MyDataset(data_folder=data_folder_test,flag="val")   
batch_size = 1
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=12,pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=12,pin_memory=True)







def calculate_r_squared_masked(pred_values, signal_values, mask):
        """
        Helper function to calculate R-squared between predicted pixel values and actual
        masked pixel values over all masked gene expression values from all cells and genes.

        Args:
            pred_values:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            b,1559*30,2,100
            signal_values:  numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            mask:           binary mask of shape [batch_size, num_voxels, num_tokens]
            b,1559,2
        """
        gt_list = []
        pred_vals_list = []
      
        for sample_idx in range(signal_values.shape[0]):
            for voxel_idx in range(signal_values.shape[1]):
                gt_list += list(
                    signal_values[sample_idx, voxel_idx][
                        mask[sample_idx, voxel_idx]
                    ].flatten()
                )
                pred_vals_list += list(
                    pred_values[sample_idx, voxel_idx][
                        mask[sample_idx, voxel_idx]
                    ].flatten()
                )

        r_squared = r2_score(y_true=gt_list, y_pred=pred_vals_list)
        if r_squared < 0.0:
            r_squared = 0.0
        return r_squared

def plot_masked_pred_trends_one_sample(
    pred_logits: np.array,
    signal_vectors: np.array,
    mask: np.array,
    sample_idx: int,
    node_idxs: np.array,
    dataset_split: str,
    index:int,
):
    """
    Function to plot timeseries of model predictions as continuation of input data compared to
    ground truth. A grid of sub-lineplots will be logged to wandb, with len(sample_idxs) rows
    and len(node_idxs) columns
    Args:
        pred_logits:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        signal_vectors: numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        mask:           binary mask of shape [batch_size, num_voxels, num_tokens]
        sample_idx:     index of sample to plot; one per figure
        node_idxs:      indices of voxels to plot; affects how many columns in plot grid there will be
        dataset_split:  train, val, or test
        epoch:          training epoch
    Returns:
    """
  
    signal_vectors = signal_vectors.reshape(pred_logits.shape)

    plt.rc('font', family='Times New Roman')
    fig, axes = plt.subplots(nrows=len(node_idxs), ncols=1, sharex=True, sharey=True)
    fig.set_figwidth(23)
    fig.set_figheight(4 * len(node_idxs))

    batch_size, num_voxels, num_tokens, time_patch_preds = pred_logits.shape

    # --- Plot Figure ---#
    for row_idx, node_idx in enumerate(node_idxs):
        ax = axes[row_idx]

        input_data_vals = []
        input_data_timepoints = []
        model_pred_vals = []
        model_pred_timepoints =[]
        
        for token_idx in range(signal_vectors.shape[2]):
            input_data_vals += signal_vectors[sample_idx, node_idx, token_idx].tolist()
            start_timepoint = time_patch_preds * token_idx
            end_timepoint = start_timepoint + time_patch_preds
            input_data_timepoints += list(range(start_timepoint, end_timepoint))

            for i in range(len(input_data_timepoints)):
                ax.plot(
                    input_data_timepoints[i],
                    input_data_vals[i],
                    marker="o",
                    markersize=7,
                    label="",
                    color="deepskyblue",
                    markeredgewidth=0.1, markeredgecolor='black'
                )
            if mask[sample_idx, node_idx, token_idx] == 1:
                model_pred_vals += pred_logits[sample_idx, node_idx, token_idx].tolist()
                model_pred_timepoints += list(range(start_timepoint, end_timepoint))
                
        ax.plot(
                    model_pred_timepoints,
                    model_pred_vals,
                    marker=",",
                    markersize=12,
                    label="",
                    color="red",
                    linewidth=2
                )

       
        ax.axhline(y=0.0, color="gray", linestyle="--", markersize=12)
        
        ax.tick_params(axis='x', labelsize=28)
        ax.tick_params(axis='y', labelsize=24)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.9))
    plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
   
    
    
   
    plt.show()
   
    plt.savefig('###############'+str(index)+'.jpg',dpi=300)

    plt.close()    

def calculate_pearson_masked(pred_values, signal_values, mask):
    """
    Helper function to calculate Pearson correlation between predicted pixel values and actual
    masked pixel values over all masked fMRI values from all voxels.

    Args:
        pred_values:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        signal_values:  numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        mask:           binary mask of shape [batch_size, num_voxels, num_tokens]
    """
    gt_list = []
    pred_vals_list = []
  
    for sample_idx in range(signal_values.shape[0]):
        for voxel_idx in range(signal_values.shape[1]):
            gt_list += list(
                signal_values[sample_idx, voxel_idx][
                    mask[sample_idx, voxel_idx]
                ].flatten()
            )
            pred_vals_list += list(
                pred_values[sample_idx, voxel_idx][
                    mask[sample_idx, voxel_idx]
                ].flatten()
            )

    pearson = pearsonr(x=gt_list, y=pred_vals_list)
    p = pearson.statistic
    if p < 0.0:
        p = 0.0
    return p

def forward_loss(signal_values, pred_values,mask):
       
        assert signal_values.shape == pred_values.shape
        
        # mask = torch.ones(pred_values.shape,device=pred_values.device)

       
        loss = (
            ((pred_values - signal_values) ** 2) * mask
        ).sum() / mask.sum()  # MSE
        
        return loss

p_list = []
r_list = []
los =[]        
for epoch in range(1):
    # for batch in train_loader:
    all_p = 0
    all_r = 0
    all_l=0
    for i, (inputs, labels) in enumerate(train_loader):
       
        
   

        model_series_inputs = collate_fn(inputs)
        signal = model_series_inputs["signal_vectors"].to(device)
        signal2 = model_series_inputs["signal_vectors1"].to(device)
        model_series_inputs["signal_vectors"] = signal.to(device)
        model_series_inputs["signal_vectors1"] = signal2.to(device)
        
        outputs_series = model_series(signal_vectors=model_series_inputs["signal_vectors"],signal_vectors1=model_series_inputs["signal_vectors1"],labels=model_series_inputs["labels"],input_ids=model_series_inputs["input_ids"],xyz_vectors=model_series_inputs["xyz_vectors"])
        
        pred_logits = outputs_series["logits"][0]
        mask1 = outputs_series["mask"].transpose(1,2).cpu().numpy()
        mask = outputs_series["mask"].cpu().numpy()
     
        mask = mask.astype(bool)
        mask1 = mask1.astype(bool)
       
        
        all_l +=forward_loss(pred_logits.cpu().numpy(),signal.cpu().numpy(),mask1)
        
        pred_logits =  np.expand_dims(np.transpose(pred_logits.cpu(),axes=(0,2,1)), axis=-1)
        signal_vectors=np.expand_dims( np.transpose(signal.cpu(),axes=(0,2,1)), axis=-1)
        
        
        plot_masked_pred_trends_one_sample(
            pred_logits=pred_logits,
            signal_vectors=signal_vectors,
            mask=mask,
            sample_idx=0,
            node_idxs=[200,600,900],
            dataset_split="val",
            index=i,
        )
        
        
#         unadjusted_r2 = calculate_r_squared_masked(
#             pred_logits, signal_vectors, mask
#         )

#         p = calculate_pearson_masked(pred_logits, signal_vectors, mask)
#         all_p += p
#         all_r += unadjusted_r2
        
        
        
#         # print(unadjusted_r2,p)
#     rr = all_r/len(train_loader)
#     pp = all_p/len(train_loader)
#     ll= all_l/len(train_loader)
#     print(pp,rr,ll)
#     p_list.append(pp)
#     r_list.append(rr)
        
# print("pp:",np.mean(np.array(p_list)),np.std(np.array(p_list)))
# print("rr:",np.mean(np.array(r_list)),np.std(np.array(r_list)))









