from typing import Dict

import torch
import numpy as np
from sklearn.metrics import r2_score
from transformers.trainer_utils import EvalPrediction
from scipy.stats import pearsonr

from utils.plots import (
    plot_model_output_histogram,
    plot_scatterplot,
    plot_masked_pred_trends_one_sample,
    plot_cls_token_2d_umap,
)


class MetricsCalculator:
    """
    Class for metric calculation. An object of this class will be passed to the Huggingface Trainer
    class as a callable to calculate all metrics for training BrainLM models.

    Receives an EvalPrediction object from Trainer.
    Call:
        eval_pred_obj:              EvalPrediction object containing predictions, label_ids, and model inputs

    Returns:
        metrics_dict: dictionary containing metrics
            - mse:                  mean square error between masked expression values and model predictions
            - mae:                  mean absolute error between masked expression values and model predictions
            - cell_r2_avg:          average R2 across cells in minibatch, calculated on masked expression values
            - cell_r2_list:         list of R2 of individual cells in minibatch
            - gene_r2_avg:          average R2 across genes in minibatch, calculated on masked expression values
            - gene_r2_list:         list of R2 values of individual genes in minibatch
            - r2gene_idx_list:      indices of genes which were considered in gene_r2_avg calculation
            - cross_entropy_loss:   cross entropy loss between masked expression and model prediction
    """

    def __init__(self) -> None:
        self.cross_entropy_criterion = torch.nn.CrossEntropyLoss()
        self.current_epoch = 0  # Updated in log() function of CellLM Trainer

    def __call__(self, eval_pred_obj: EvalPrediction) -> Dict:

        (pred_logits, encoder_latents), mask = eval_pred_obj.predictions

        cls_tokens = encoder_latents[:, 0, :]  
        age_labels = eval_pred_obj.label_ids

        # Get input expression vectors and sampled gene indices
        signal_vectors_padded = eval_pred_obj.inputs
        signal_vectors = signal_vectors_padded[: pred_logits.shape[0], :]
        signal_vectors = np.reshape(signal_vectors, pred_logits.shape)
        

        
        
        
        pred_logits =  np.expand_dims(np.transpose(pred_logits,axes=(0,2,1)), axis=-1)
        signal_vectors=np.expand_dims( np.transpose(signal_vectors,axes=(0,2,1)), axis=-1)


        # Calculate MSE and MAE
        mse = self.calculate_mse(pred_logits, signal_vectors, mask)
        mae = self.calculate_mae(pred_logits, signal_vectors, mask)
        

        # Calculate R2
        mask = mask.astype(bool)
        unadjusted_r2 = self.calculate_r_squared_masked(
            pred_logits, signal_vectors, mask
        )

        p = self.calculate_pearson_masked(pred_logits, signal_vectors, mask)

        plot_scatterplot(
            pred_logits,
            signal_vectors,
            mask,
            epoch=self.current_epoch,
            dataset_split="val",
        )
        plot_masked_pred_trends_one_sample(
            pred_logits=pred_logits,
            signal_vectors=signal_vectors,
            mask=mask,
            sample_idx=0,
            node_idxs=[0, 50,100,150,180],
            # node_idxs=[0, 5,10,15,20,25,40,45,50,65,70,75,80,90,100,200,300,400],
            # node_idxs=[0, 1000,2000,50000,10000],
            dataset_split="val",
            epoch=self.current_epoch,
        )
        
        # --- Return metrics dictionary ---#
        metrics_dict = {
            "mse": mse,
            "mae": mae,
            "r2": unadjusted_r2,
            "pearson r": p,
        }
        return metrics_dict

    @staticmethod
    def calculate_mse(pred_values, signal_values, mask):
        """
        Helper function to calculate Mean Square Error (MSE) on predicted masked gene expression values.

        Args:
            pred_values:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            signal_values:  numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            b,1559*30,2,100
            mask:           binary mask of shape [batch_size, num_voxels, num_tokens]
            b,1559,2

        Returns:
            loss:           mean square error loss on only masked timepoints
        """
        # mask = np.expand_dims(mask, axis=-1).repeat(30, axis=-1)  ## b,1559,2,30
        # mask = np.reshape(mask,(mask.shape[0],-1,mask.shape[2])) ## b,1559*30,2
        mask = np.expand_dims(mask, axis=-1).repeat(pred_values.shape[-1], axis=-1)
        

        # print("mask",mask.shape)
        loss = (((pred_values - signal_values) ** 2) * mask).sum() / mask.sum()
        return loss.item()

    @staticmethod
    def calculate_mae(pred_values, signal_values, mask):
        """
        Helper function to calculate Mean Absolute Error (MAE) on predicted masked gene expression values.

        Args:
            pred_values:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            signal_values:  numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
            mask:           binary mask of shape [batch_size, num_voxels, num_tokens]

        Returns:
            loss:           mean square error loss on only masked timepoints
        """
        # mask = np.expand_dims(mask, axis=-1).repeat(30, axis=-1)  ## b,1559,2,30
        # mask = np.reshape(mask,(mask.shape[0],-1,mask.shape[2])) ## b,1559*30,2
        mask = np.expand_dims(mask, axis=-1).repeat(pred_values.shape[-1], axis=-1)
        loss = abs((pred_values - signal_values) * mask).sum() / mask.sum()
        return loss.item()

    @staticmethod
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
        # mask = np.expand_dims(mask, axis=-1).repeat(30, axis=-1)  ##b,1559,2,30
        # mask = np.reshape(mask,(mask.shape[0],-1,mask.shape[2])) ## b,1559*30,2
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

    
    @staticmethod
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
        # mask = np.expand_dims(mask, axis=-1).repeat(30, axis=-1)  ##b,1559,2,30
        # mask = np.reshape(mask,(mask.shape[0],-1,mask.shape[2]))
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
