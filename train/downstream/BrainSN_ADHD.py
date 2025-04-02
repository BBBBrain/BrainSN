import os
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from datasets import load_from_disk, concatenate_datasets
from BrainSN_main  import BrainSNForPretraining as pre_series
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import seaborn as sns

from torch.utils.data import Dataset
from random import randint

import nibabel as nib
import numpy as np


import warnings

import logging
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TR = 0.72    
from scipy.stats import pearsonr    
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer


import pandas as pd
df = pd.read_csv('./Phenotypic_V1_0b_preprocessed1.csv')
desired_column = df['SUB_ID'].tolist()
asd = df['DX_GROUP'].tolist()




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
        data_label  = torch.tensor(data_label, dtype=torch.int64)
      
        image = np.load(self.filenames[index])
        fname = self.filenames[index].split("/")[-1]
      
       
        data1 = self.preprocess(image,2)
        if "rfMRI" in fname:
            
            return data1,torch.tensor(1, dtype=torch.int64),self.filenames[index][:-4]
        else:
          
            return data1,torch.tensor(0, dtype=torch.int64),self.filenames[index][:-4]
      


    def preprocess(self, examples,tr):
        
        
        examples_o ={}
        label = 1 
        label = torch.tensor(label, dtype=torch.int64)
        
        
   
   
        len = examples.shape[0]
       
       
        
        
        signal_vector = examples[:70,:].transpose()  # 70
        T = TimeSeries(signal_vector, sampling_interval=2)
        F = FilterAnalyzer(T, ub=0.1, lb=0.01)  #0.1
        signal_vector4 = NormalizationAnalyzer(F.filtered_fourier).z_score.data 

     
        signal_window = torch.tensor(signal_vector4, dtype=torch.float32)
        
        signal_window = signal_window.transpose(1, 0)

      
        examples_o["signal_vectors"] = signal_window
        examples_o["signal_vectors1"] = signal_window
        examples_o["xyz_vectors"] = window_xyz_list
       
        examples_o["label"] = torch.tensor(2, dtype=torch.float32)

        

        return examples_o
    
    
    





model_series = pre_series.from_pretrained("###########").to(device) 



for param in model_series.parameters():
    param.requires_grad = False
    
    
model_series.vit.embeddings.mask_ratio = 0
model_series.vit.embeddings.config.mask_ratio = 0
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





class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, bidirectional=False):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True, 
                          bidirectional=bidirectional)
        
        # Linear layer to map hidden state to output size
        if bidirectional:
          
            self.fc = nn.Linear(hidden_size*2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
            
        self.r = nn.ReLU()
        self.bn = nn.Dropout(p=0.3)##0.3
        self.r1 = nn.Tanh()

    def forward(self, x):
        # x is of shape (b, t, h)
        gru_out, hn = self.gru(x)  # hn is of shape (num_layers * num_directions, b, hidden_size)
        
        # Select the last hidden state (for the last time step)
        
        if self.gru.bidirectional:
            hn = torch.cat((hn[-2], hn[-1]), dim=1)  # For bidirectional, concat the two hidden states
         
        else:
            hn = hn[-1]  # For the last layer, get (b, hidden_size)
        
        # Pass through a fully connected layer
        output = self.fc(hn)  # Output shape (b, output_size)
        return output
    
num_classes = 2
class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()

        self.bn = nn.Dropout(p=0.1)

        self.r1 = nn.Tanh()


        
        self.grum = GRUModel(1024, 1024, 2)  #1024, 512, 64

    def forward(self, x):
        out = self.grum(self.bn(self.r1(x)))
        return out    
    

data_folder = "###########"  
data_folder_test = "####################"
train_dataset = MyDataset(data_folder=data_folder,flag="train")
val_dataset = MyDataset(data_folder=data_folder_test,flag="val")   
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=12,pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=12,pin_memory=True)


m_model = MyModel(512,num_classes).to(device)

learning_rate = 0.00001  ## 0.0001
optimizer = torch.optim.Adam(m_model.parameters(), lr=learning_rate, weight_decay=1e-2)
loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 1000


best_acc = 0

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

for epoch in range(num_epochs):
    # for batch in train_loader:
    for i, (inputs, labels,name) in enumerate(train_loader):
       
        m_model.train()
        
    
        model_series_inputs = collate_fn(inputs)
      
        signal = model_series_inputs["signal_vectors"].to(device)
        signal2 = model_series_inputs["signal_vectors1"].to(device)
    
        model_series_inputs["signal_vectors"] = signal.to(device)
        model_series_inputs["signal_vectors1"] = signal2.to(device)
        
        outputs_series = model_series(signal_vectors=model_series_inputs["signal_vectors"],signal_vectors1=model_series_inputs["signal_vectors1"],labels=model_series_inputs["labels"],input_ids=model_series_inputs["input_ids"],xyz_vectors=None)
        
        
        
        
        
        
        
        outputs_series_latent = outputs_series["logits"][0][:,2:,:] 
       

        outputs = m_model(outputs_series_latent).to(device)
    
        loss = loss_fn(outputs, labels.to(device))
        loss.backward()
        optimizer.step() # 更新网络参数
        optimizer.zero_grad()
    

    with torch.no_grad():
        model_series.eval()
        m_model.eval()
        correct = 0
        total = 0
        pred =[]
        lab = []
        y_proba = []
        for batch in val_loader:
            inputs, labels,_ = batch
            

            
            model_series_inputs = collate_fn(inputs)
          
            signal = model_series_inputs["signal_vectors"].to(device)
            signal2 = model_series_inputs["signal_vectors1"].to(device)
         
            model_series_inputs["signal_vectors"] = signal.to(device)
            model_series_inputs["signal_vectors1"] = signal2.to(device)
            
            outputs_series = model_series(signal_vectors=model_series_inputs["signal_vectors"],signal_vectors1=model_series_inputs["signal_vectors1"],labels=model_series_inputs["labels"],input_ids=model_series_inputs["input_ids"],xyz_vectors=model_series_inputs["xyz_vectors"])
           
            
            outputs_series_latent = outputs_series["logits"][0][:,2:,:] 
          
            outputs = m_model(outputs_series_latent).to(device)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
         
            pred.append(predicted.cpu()[0])
            lab.append(labels[0])
            y_proba.append(outputs.cpu()[0][0])

        accuracy = 100 * correct / total
       
        f1_score_value = f1_score(np.array(pred), np.array(lab), average='macro')
        
        tn, fp, fn, tp = confusion_matrix(np.array(lab), np.array(pred)).ravel()
        spe = tn / (tn + fp)
        sen = tp / (tp + fn)
        
        if accuracy>=best_acc:
            best_acc = accuracy
            torch.save(m_model, '##############'+'ACC{:.4f}%'.format(accuracy)+'SEN{:.4f}%'.format(sen)+'SPE{:.4f}%'.format(spe)+'_asd.pt')
        print('ACC{:.2f}%'.format(accuracy),'F1{:.2f}%'.format(f1_score_value),'SEN{:.2f}%'.format(sen),'SPE{:.2f}%'.format(spe))
