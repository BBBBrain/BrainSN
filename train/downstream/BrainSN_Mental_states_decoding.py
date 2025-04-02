
import os
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

from datasets import load_from_disk, concatenate_datasets
import torch.nn.functional as F


from brainlm_mae.modeling_brainlm_task  import BrainLMForPretraining as pre_series



from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import seaborn as sns

from torch.utils.data import Dataset
from random import randint

import nibabel as nib
import numpy as np

from torch.optim.lr_scheduler import StepLR
import warnings

import logging
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import precision_score, recall_score
   
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def standardize(data):
    median_value = np.median(data)
    Q1 = np.percentile(data,25)
    Q3 = np.percentile(data,75)
    IQR = Q3 - Q1
    
    return (data - median_value) / IQR




TR = 0.72   
    
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer


from scipy.signal import butter, filtfilt


def z_score_normalize(signal_vector):
    
    median_value = np.median(signal_vector[g])
    Q1 = np.percentile(signal_vector[g],25)
    Q3 = np.percentile(signal_vector[g],75)
    IQR = Q3 - Q1
    signal_vector[g] = (signal_vector[g] - median_value) / IQR




class MyDataset(Dataset):
    def __init__(self, data_folder,flag="train"):
        self.data_folder = data_folder
        self.filenames = []
        self.flag = flag
        
        
        file_list = os.listdir(self.data_folder)
        file_list.sort(key= lambda x:x[:6])#.png所以是[:-4]
        #print(len(file_list))
        
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
       
        labels = {'MOTOR_lf':0, 'MOTOR_rf':1,'MOTOR_lh':2,'MOTOR_rh':3,'MOTOR_t':4,
                 'WM_0bk_body':5,'WM_0bk_faces':6,'WM_0bk_places':7,'WM_0bk_tools':8,'WM_2bk_body':5,'WM_2bk_faces':6,'WM_2bk_places':7,'WM_2bk_tools':8,
                  'EMOTION_fear':9,'EMOTION_neut':10,'GAMBLING_loss':11,'GAMBLING_win':12,'LANGUAGE_math':13,'LANGUAGE_story':14,
                  'RELATIONAL_match':15 ,'RELATIONAL_relation':16,'SOCIAL_mental':17,'SOCIAL_rnd':18,'rfMRI':19
                 }
        
        data_label  = -1 
        data_label  = torch.tensor(data_label, dtype=torch.int64)
        for i in labels.keys():
            if i in self.filenames[index]:
                
                data_label = labels[i]
                break
        tr = -100
        if "rfMRI" in self.filenames[index]:
            tr = 2
        else:
            tr = 0.6
        # print(self.filenames[index])
        image = np.load(self.filenames[index])
        # print(image.shape)
       
        data1 = self.preprocess(image,0.72)
       
       
        return data1,data_label,self.filenames[index][:-4]
    

    
    def preprocess(self, examples,tr):
        
        examples_o ={}
        label = 1 
        label = torch.tensor(label, dtype=torch.int64)
        
        
   
   
        len = examples.shape[0]
       
        
        
        signal_vector = examples[:50,:].transpose()  
        T = TimeSeries(signal_vector, sampling_interval=2)
        F = FilterAnalyzer(T, ub=0.15, lb=0.01)
        signal_vector4 = NormalizationAnalyzer(F.filtered_fourier).z_score.data  

     
        signal_window = torch.tensor(signal_vector4, dtype=torch.float32)
        
        signal_window = signal_window.transpose(1, 0)

    
        examples_o["signal_vectors"] = signal_window
        examples_o["signal_vectors1"] = signal_window
        examples_o["xyz_vectors"] = None
       
        examples_o["label"] =  torch.tensor(0.72, dtype=torch.float32)

        

        return examples_o
    
    




model_series = pre_series.from_pretrained("##############").to(device)   




for param in model_series.parameters():
    param.requires_grad = False
    


model_series.vit.embeddings.mask_ratio = 0.0
model_series.vit.embeddings.config.mask_ratio = 0.0
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
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=True):  ## 1,true
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True, 
                          bidirectional=bidirectional)
        
        # Linear layer to map hidden state to output size
        if bidirectional:
            # self.fc = nn.Linear(hidden_size * 2, output_size)
            self.fc = nn.Linear(hidden_size*2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
            
        self.r = nn.ReLU()
        self.bn = nn.Dropout(p=0.05)
        self.r1 = nn.Tanh()

    def forward(self, x):
        # x is of shape (b, t, h)
        gru_out, hn = self.gru(x)  # hn is of shape (num_layers * num_directions, b, hidden_size)
        
        # Select the last hidden state (for the last time step)
        
        if self.gru.bidirectional:
            hn = torch.cat((hn[-2], hn[-1]), dim=1)  # For bidirectional, concat the two hidden states
            # hn =  hn[-2] + hn[-1] 
        else:
            hn = hn[-1]  # For the last layer, get (b, hidden_size)
        
        # Pass through a fully connected layer
        output = self.fc(hn)  # Output shape (b, output_size)
        return output,hn
        # return hn
    



num_classes = 20
class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()

        self.r1 = nn.Tanh()

        

   

    def forward(self, x):
        
        out,hn = self.grum(self.r1(x))


        return out,hn
    
    

    
    

data_folder = "#############"  
data_folder_test = "#######################"
train_dataset = MyDataset(data_folder=data_folder,flag="train")
val_dataset = MyDataset(data_folder=data_folder_test,flag="val")   
batch_size = 1
                                                                                                                                            
                                                                                                   
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=12,pin_memory=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=12,pin_memory=True)    


m_model = MyModel(512,num_classes).to(device)

learning_rate = 0.001

optimizer = torch.optim.Adam(m_model.parameters(), lr=learning_rate)

loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 100


best_acc = 0

from scipy.io import savemat




for epoch in range(num_epochs):
    # for batch in train_loader:
    for i, (inputs, labels,name) in enumerate(train_loader):
       
        m_model.train()
        
    
        model_series_inputs = collate_fn(inputs)

        signal = model_series_inputs["signal_vectors"].to(device)
        signal2 = model_series_inputs["signal_vectors1"].to(device)
     
        model_series_inputs["signal_vectors"] = signal.to(device)
        model_series_inputs["signal_vectors1"] = signal2.to(device)
        
        outputs_series = model_series(signal_vectors=model_series_inputs["signal_vectors"],signal_vectors1=model_series_inputs["signal_vectors1"],labels=model_series_inputs["labels"],input_ids=model_series_inputs["input_ids"],xyz_vectors=model_series_inputs["xyz_vectors"])
        
        
        
        
        
        
        
        outputs_series_latent = outputs_series["logits"][0][:,1:,:] 
      

        outputs,_ = m_model(outputs_series_latent)
        outputs = outputs.to(device)
    
        loss = loss_fn(outputs, labels.to(device))
        loss.backward()
        # optimizer.step() 
        
        if (i+1)%128==0: #
            
            optimizer.step()
            optimizer.zero_grad()
  
            
            




    with torch.no_grad():
        model_series.eval()
        m_model.eval()
        correct = 0
        total = 0
        pred =[]
        lab = []
        for batch in val_loader:
            inputs, labels,_ = batch
            

            
            model_series_inputs = collate_fn(inputs)
          
            signal = model_series_inputs["signal_vectors"].to(device)
            signal2 = model_series_inputs["signal_vectors1"].to(device)
          
            model_series_inputs["signal_vectors"] = signal.to(device)
            model_series_inputs["signal_vectors1"] = signal2.to(device)
            
            outputs_series = model_series(signal_vectors=model_series_inputs["signal_vectors"],signal_vectors1=model_series_inputs["signal_vectors1"],labels=model_series_inputs["labels"],input_ids=model_series_inputs["input_ids"],xyz_vectors=model_series_inputs["xyz_vectors"])
           
            
            outputs_series_latent = outputs_series["logits"][0][:,1:,:] 
        
            
            outputs,_ = m_model(outputs_series_latent)
            outputs = outputs.to(device)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            # print(predicted.shape,labels.shape)
            pred.append(predicted.cpu()[0])
            lab.append(labels[0])

        accuracy = 100 * correct / total
        print(np.array(pred).shape,np.array(lab).shape)
        f1_score_value = f1_score(np.array(pred), np.array(lab), average='macro')
        
        macro_prec = precision_score(np.array(lab), np.array(pred), average='macro')
        macro_rec = recall_score(np.array(lab), np.array(pred),  average='macro')
        
        if accuracy>best_acc:
            best_acc = accuracy
            torch.save(m_model, '##################'+str(accuracy)+'_HCP_Task.pt')
        
        print('Validation accuracy: {:.2f}%'.format(accuracy),"F1:",f1_score_value,macro_prec,macro_rec)