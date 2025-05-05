"""
Based on Huggingface script for pretraining Vit_MAE:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-pretraining/run_mae.py
"""
import torch
import logging
import os
import sys
import math
from dataclasses import dataclass, field
from typing import Optional
from random import randint

import torch
import wandb
from datasets import load_from_disk, DatasetDict, concatenate_datasets

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from brainlm_mae.modeling_brainlm import BrainLMForPretraining
from brainlm_mae.configuration import BrainLMConfig
from utils.trainer import BrainLMTrainer
from utils.metrics import MetricsCalculator
import pandas as pd

logger = logging.getLogger(__name__)

check_min_version("4.28.0.dev0")
require_version("datasets>=1.8.0", "To fix: conda env create -f environment.yml")
os.environ["WANDB_API_KEY"] = '###########' 
os.environ["WANDB_MODE"] = "offline"
torch.cuda.empty_cache()

import warnings
# warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


from torch.nn.utils.rnn import pad_sequence

import numpy as np
import os
import argparse
from math import ceil
import nibabel as nib
# from datasets import Dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer

from nilearn import signal    
    
import random
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def read_xlsx(file_path):
    df = pd.read_excel(file_path)
    data = df.to_numpy()
    return data

coords_ds = read_xlsx("./Coordinates-398.xlsx")
window_xyz_list = []       
for brain_region_idx in range(398):
    # Append voxel coordinates
    xyz = torch.tensor(
        [
            coords_ds[brain_region_idx][0],
            coords_ds[brain_region_idx][1],
            coords_ds[brain_region_idx][2],
        ],
        dtype=torch.float32,
    )
    window_xyz_list.append(xyz)
window_xyz_list = torch.stack(window_xyz_list)

def standardize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std
    
TR = 0.72



def which_class(file_name):
    labels = {'MOTOR_lf':0, 'MOTOR_rf':1,'MOTOR_lh':2,'MOTOR_rh':3,'MOTOR_t':4,
                 'WM_0bk_body':5,'WM_0bk_faces':6,'WM_0bk_places':7,'WM_0bk_tools':8,'WM_2bk_body':9,'WM_2bk_faces':10,'WM_2bk_places':11,'WM_2bk_tools':12,
                  'EMOTION_fear':13,'EMOTION_neut':14,'GAMBLING_loss':15,'GAMBLING_win':16,'LANGUAGE_math':17,'LANGUAGE_story':18,
                  'RELATIONAL_match':19 ,'RELATIONAL_relation':20,'SOCIAL_mental':21,'SOCIAL_rnd':22
                 }
    
    for i in labels.keys():
            if i in file_name:
                return labels[i]
            
            
from scipy.signal import butter, filtfilt
def band_pass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data




def z_score_normalize(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / std




@dataclass
class MyDataset(Dataset):
    def __init__(self, data_folder,flag="train"):
        self.data_folder = data_folder
        self.filenames = []
        self.flag = flag
        
        
        # file_list = os.listdir(self.data_folder)
        # file_list.sort(key= lambda x:x[:6])#
        
        # file_list = []
        # file_list0 = os.listdir(self.data_folder)
        file_list = os.listdir(self.data_folder)
        # for i in file_list0:
        #     if "rfMRI" in i:
        #         file_list.append(i)
        # print(len(file_list))
        # file_list.sort(key= lambda x:int(x[:6]))#
        file_list.sort(key= lambda x:x[:6])#

        
        if self.flag == "train":
            datas = file_list[:]    ### 860  2400
            for per_data in datas:
                self.filenames.append(os.path.join(self.data_folder, per_data))
        elif self.flag == "val":
            datas = file_list[:] 
            for per_data in datas:
                self.filenames.append(os.path.join(self.data_folder, per_data))
    
    def __len__(self):
        return len(self.filenames) 
    
    def __getitem__(self, index):
               
        data_label  = None
        # labels = {'MOVIE':0, 'rfMRI':0.72}
        if "rfMRI" in self.filenames[index]:
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
        data = self.preprocess(image,data_label)
        # image = nib.load(self.filenames[index]).get_fdata()
        # data1 = self.preprocess(image)
        # data2 = self.preprocess(image2)
        # return data1,data2,torch.from_numpy(np.array([int(data_label!=label2)],dtype=np.float32))
        
        # data = self.preprocess(image,image2,np.array([int(data_label!=label2)]))
        # data = self.preprocess(image,image2,torch.tensor(data_label, dtype=torch.float32))
        return data
    

    def preprocess(self, examples,tr):
        
        examples_o ={}
        # print(examples.shape)
        #label = torch.tensor(labels, dtype=torch.float32)
        
        lenth=100

        start_idx = randint(0, examples.shape[0] - lenth)  #########  100 -> timepoints  shape【0】  
        end_idx = start_idx + lenth
       
        
        
        
        signal_vector = examples[start_idx:end_idx,:]
        
        examples_o["signal_vectors"] = signal_vector
        examples_o["signal_vectors1"] = signal_vector
        examples_o["xyz_vectors"] = window_xyz_list
        # examples_o["label"] = torch.tensor(1, dtype=torch.int64)
        examples_o["label"] = torch.tensor(tr, dtype=torch.float32)
        
        
        
        return examples_o





@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_dataset_path: str = field(
        metadata={
            "help": "Path to saved train arrow dataset of cell x gene expression matrix."
        }
    )
    val_dataset_path: str = field(
        metadata={
            "help": "Path to saved val arrow dataset of cell x gene expression matrix."
        }
    )
    #coords_dataset_path: str = field(
    #    metadata={"help": "Path to saved arrow dataset of brain region coordinates."}
    #)
    #regions_dataset_path: str = field(
    #    metadata={"help": "Path to saved arrow dataset of brain region coordinates."}
    #)
    recording_col_name: str = field(
        default="Voxelwise_RobustScaler_Normalized_Recording",
        metadata={"help": "Column in dataset which contains recording for each patient. Choose from:"
                          "All_Patient_All_Voxel_Normalized_Recording, "
                          "Per_Patient_All_Voxel_Normalized_Recording, "
                          "Per_Patient_Per_Voxel_Normalized_Recording, "
                          "Per_Voxel_All_Patient_Normalized_Recording, "
                          "Subtract_Mean_Normalized_Recording, "
                          "or Subtract_Mean_Divide_Global_STD_Normalized_Recording"
                  }
    )
    variable_of_interest_col_name: str = field(
        default="Age.At.MHQ",
        metadata={
            "help": "Column in dataset containing desired label for each patient. Choose from:"
            "Order, eid, Gender, Age.At.MHQ, PHQ9.Severity, Depressed.At.Baseline"
            "Neuroticism, Self.Harm.Ever, Not.Worth.Living, PCL.Score, GAD7.Severity"
        },
    )
    num_timepoints_per_voxel: int = field(
        default=100,
        metadata={
            "help": "Number of timepoints for each voxel given in 1 sample input to model. "
            "Must be divisible by timepoint_patching_size."
        },
    )
    timepoint_patching_size: int = field(
        default=10,
        metadata={
            "help": "Length of moving window of timepoints from each brain "
            "regions signal for 1 sample."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        self.data_files = None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name_or_path"
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    hidden_size: int = field(default=128, metadata={"help": "Encoder hidden size."})
    num_hidden_layers: int = field(default=2, metadata={"help": "Encoder num layers."})
    num_attention_heads: int = field(
        default=2, metadata={"help": "Number of attention heads in encoder."}
    )
    intermediate_size: int = field(
        default=512, metadata={"help": "Intermediate size in MLP in encoder layers."}
    )
    decoder_hidden_size: int = field(
        default=128, metadata={"help": "Decoder hidden size."}
    )
    decoder_num_hidden_layers: int = field(
        default=2, metadata={"help": "Decoder num layers."}
    )
    decoder_num_attention_heads: int = field(
        default=2, metadata={"help": "Number of attention heads in the decoder."}
    )
    decoder_intermediate_size: int = field(
        default=512, metadata={"help": "Intermediate size in MLP in decoder layers."}
    )
    hidden_dropout_prob: float = field(
        default=0,
        metadata={"help": "Dropout probability for layer activations in CellLM."},
    )
    attention_probs_dropout_prob: float = field(
        default=0,
        metadata={"help": "Dropout probability for attention coefficients in CellLM."},
    )
    mask_ratio: float = field(
        default=0.3,
        metadata={"help": "The ratio of the number of masked tokens per voxel."},
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Don't remove unused columns."}
    )
    do_train: int = field(default=True, metadata={"help": "Whether to do training."})
    do_eval: int = field(default=True, metadata={"help": "Whether to do eval."})
    base_learning_rate: float = field(
        default=0.0001,
        metadata={
            "help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."
        },
    )
    lr_scheduler_type: str = field(
        default="cosine_with_restarts",
        metadata={"help": "What learning rate scheduler to use."},
    )
    weight_decay: float = field(
        default=0.05,
        metadata={
            "help": "Weight decay (L2 regularization coefficient) for optimizer."
        },
    )
    num_train_epochs: int = field(
        default=500, metadata={"help": "Number of epochs to train for."}
    )
    warmup_ratio: float = field(
        default=0.001, metadata={"help": "Warmup ratio for learning rate scheduler."}
    )
    per_device_train_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size for each device used during training."},
    )
    per_device_eval_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size for each device used during evaluation."},
    )
    logging_strategy: str = field(
        default="steps",
        metadata={
            "help": "How often to log training metrics. If choose 'steps', specify logging_steps."
        },
    )
    logging_steps:int = field(#####150
        default=1000,
        metadata={
            "help": "If logging_strategy is 'steps', log training metrics every X iterations."
        },
    )
    evaluation_strategy: str = field(
        default="steps", metadata={"help": "How often to log eval results."}
    )
    eval_steps: int = field(
        default=1000,#####20
        metadata={
            "help": "If evaluation_strategy is 'steps', calculate validation metrics every X iterations."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "How often to save results and models."}
    )
    save_steps: int = field(
        default=1000,
        metadata={
            "help": "If save_strategy is 'steps', save model checkpoint every X iterations."
        },
    )
    load_best_model_at_end: bool = field(
        default=True, metadata={"help": "At the end, load the best model."}
    )
    save_total_limit: int = field(
        default=50000, metadata={"help": "Maximum number of models to save."}
    )
    seed: int = field(default=1234, metadata={"help": "Random seed."})
    wandb_logging: bool = field(
        default=False,
        metadata={
            "help": "Whether to log metrics to weights & biases during training."
        },
    )
    include_inputs_for_metrics: bool = field(
        default=True,
        metadata={
            "help": "Trainer will include model inputs in call to metrics calculation function. Depends on 'input_ids' being one of the input parameters to model, comes from tokenizer used? Currently incompatible with single-cell dataloader, leave as False."
        },
    )
    loss_fn: str = field(
        default="mse",
        metadata={"help": "Loss function for CellLM to use for pretraining."},
    )
    use_tanh_decoder: bool = field(
        default=False,
        metadata={
            "help": "If we want to use TanH as the nonlinearity for the output layer."
        },
    )


def collate_fn(examples):
    """
    This function tells the dataloader how to stack a batch of examples from the dataset.
    
    """
    
        
    all_len = randint(20,100)  ##30-100
    
    
    
    
    
    for example in examples:
        seq_i = example["signal_vectors"][:all_len,:].transpose() 
        T = TimeSeries(seq_i, sampling_interval=2)
        F = FilterAnalyzer(T, ub=0.15, lb=0.01)
        signal_vector_i = NormalizationAnalyzer(F.filtered_fourier).z_score.data.transpose() 
        example["signal_vectors"] = torch.tensor(signal_vector_i, dtype=torch.float32)
        
        # seq_i = example["signal_vectors"][:all_len,:].transpose()
        # # print(seq_i.shape)
        # Q1 = np.percentile(seq_i, 25,axis=1,keepdims=True)
        # Q2 = np.percentile(seq_i, 50,axis=1,keepdims=True)
        # Q3 = np.percentile(seq_i, 75,axis=1,keepdims=True)
        # IQR = Q3 - Q1
        # # 应用鲁棒缩放
        # global_norm_dat_arr = (seq_i - Q2) / IQR
        # fMRI_data_standardized = scaler.fit_transform(seq_i)
        # example["signal_vectors"] = torch.tensor(global_norm_dat_arr.transpose(), dtype=torch.float32)
        
    signal_vectors = torch.stack(
        [example["signal_vectors"] for example in examples], dim=0
    )
    
    # signal_vectors = torch.stack([
    #     pad_sequence([example["signal_vectors"] for example in examples],batch_first=True, padding_value=0)], dim=0
    # ).squeeze(0)
    # print(signal_vectors.shape)
    xyz_vectors = torch.stack([example["xyz_vectors"] for example in examples])
    labels = torch.stack([example["label"] for example in examples])
    #regions  = examples[0]["regions"] 

    # These inputs will go to model.forward(), names must match
    return {
        "signal_vectors": signal_vectors,
        "signal_vectors1": signal_vectors,
        "xyz_vectors": xyz_vectors,
        "input_ids": signal_vectors,
        "labels": labels,
        #"regions":regions,
    }
 


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Check that arguments for  make sense
    # assert (
    #     data_args.num_timepoints_per_voxel % data_args.timepoint_patching_size == 0
    # ), "Number of timepoints per voxel should be divisible by the timepoint patching size."

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mae", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # --- Initialize Weights & Biases Logging ---#
    date_time_stamp = training_args.output_dir.split("/")[-1]

    if training_args.wandb_logging:
        # NOTE: Change project name to your own project name in your weights & biases account
        dataset_name = data_args.train_dataset_path.split("/")[-1]
        dataset_name = dataset_name.split("_cell_dataset")[0]
        wandb.init(
            project="BrainLM", name="{}-{}".format(dataset_name, date_time_stamp)
        )

    # --- Initialize Dataset ---#
    # Load arrow datasets
    #train_ds = load_from_disk(data_args.train_dataset_path)
    #val_ds = load_from_disk(data_args.val_dataset_path)
    
#     train_ds_ = [load_from_disk(data_args.train_dataset_path+"_"+str(shard_idx)) for shard_idx in range(70)]
#     for ds in train_ds_:
#         ds.set_format("torch")
#     train_ds = concatenate_datasets(train_ds_) 
    
#     val_ds_ = [load_from_disk(data_args.val_dataset_path+"_"+str(shard_idx)) for shard_idx in range(1)]
#     for ds in val_ds_:
#         ds.set_format("torch")
#     val_ds = concatenate_datasets(val_ds_) 

    train_folder = "/root/autodl-tmp/BrainLM-main/DataSet/movie_story_r_1000"   ######### HCP_rfMRI_origion   HCP_all_424t
    test_folder = "/root/autodl-tmp/BrainLM-main/DataSet/movie_story_r_1000_test"
    train_dataset = MyDataset(data_folder=train_folder,flag="train")
    val_dataset = MyDataset(data_folder=test_folder,flag="val")
    

    # ds = DatasetDict({"train": train_ds, "validation": val_ds})

    # Load gene information dataset (containing gene names, expression mean and std dev)
    #coords_ds = load_from_disk(data_args.coords_dataset_path)      
    #regions_ds = load_from_disk(data_args.regions_dataset_path)

    # Load model
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = BrainLMConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = BrainLMConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = BrainLMConfig(
            hidden_size=model_args.hidden_size,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            intermediate_size=model_args.intermediate_size,
            hidden_dropout_prob=model_args.hidden_dropout_prob,
            attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
            decoder_num_attention_heads=model_args.decoder_num_attention_heads,
            decoder_hidden_size=model_args.decoder_hidden_size,
            decoder_num_hidden_layers=model_args.decoder_num_hidden_layers,
            decoder_intermediate_size=model_args.decoder_intermediate_size,
            num_timepoints_per_voxel=data_args.num_timepoints_per_voxel,
            mask_ratio=model_args.mask_ratio,
            timepoint_patching_size=data_args.timepoint_patching_size,
            loss_fn=training_args.loss_fn,
            use_tanh_decoder=training_args.use_tanh_decoder,
        )
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # adapt config
    config.update({
        "mask_ratio": model_args.mask_ratio,
        "attention_probs_dropout_prob": model_args.attention_probs_dropout_prob
    })

    # create model
    if model_args.model_name_or_path:
        model = BrainLMForPretraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = BrainLMForPretraining(config)

    if training_args.wandb_logging:
        wandb.watch(model, log="all", log_freq=1)

    if data_args.recording_col_name is not None:
        recording_col_name = data_args.recording_col_name
    else:
        raise AttributeError(
            "Please specify the dataset column containing the signal recording (recording_col_name) in the DataTrainingArguments class."
        )

    if data_args.variable_of_interest_col_name is not None:
        variable_of_interest_col_name = data_args.variable_of_interest_col_name
    else:
        raise AttributeError(
            "Please specify the dataset column containing the signal recording (recording_col_name) in the DataTrainingArguments class."
        )

    if data_args.num_timepoints_per_voxel is not None:
        num_timepoints_per_voxel = data_args.num_timepoints_per_voxel
    else:
        raise AttributeError(
            "Please specify the moving window length (moving_window_len) in the DataTrainingArguments class."
        )


    # Compute absolute learning rate
    total_train_batch_size = (
        training_args.train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = (
            training_args.base_learning_rate * total_train_batch_size / 256
        )

    metrics_calculator = MetricsCalculator()

    # Initialize our trainer
    trainer = BrainLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=metrics_calculator,
    )

    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)




def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
