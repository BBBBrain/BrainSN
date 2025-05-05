import math
from copy import deepcopy
import random
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Union, Tuple
import torch.nn.functional as F
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEForPreTraining,
    ViTMAEEncoder,
    ViTMAEModel,
    ViTMAEEmbeddings,
    ViTMAEForPreTrainingOutput,
    ViTMAEModelOutput,
    ViTMAEDecoder,
    ViTMAEDecoderOutput,
)
from transformers.models.nystromformer.modeling_nystromformer import NystromformerLayer
from transformers.modeling_outputs import BaseModelOutput
from random import randint
from configuration import BrainlmConfig
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super(PositionalEncoding, self).__init__()
        self.position_embedding = nn.Parameter(torch.empty(max_len, d_model))
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()
        

    def init_weights(self):
    
        nn.init.xavier_uniform_(self.position_embedding)

    def forward(self, x):
        # Assuming x is of shape (batch_size, seq_len, embedding_dim)
        seq_len = x.size(1)
        pos_encoding = self.position_embedding[:seq_len, :].unsqueeze(0).repeat(x.shape[0], 1, 1)
        return x + pos_encoding
    
class BrainLMEmbeddings(ViTMAEEmbeddings):
    

    def __init__(self, config):
        super().__init__(config)
        self.patch_embeddings = None
        self.position_embeddings = None
        self.num_brain_voxels = 1000
        self.num_timepoints_per_voxel = 200
        self.mask_ratio = 0.7
        self.tokens = 1000
        self.pos_embedding = PositionalEncoding(d_model=config.hidden_size)
        
        
      
        
        self.signal_embedding_projection = nn.Linear(
            
            1000, config.hidden_size, bias=False     
        )
        
        self.signal_embedding_projection2 = nn.Linear(
            
            config.hidden_size, config.hidden_size, bias=False     
        )
        
        self.region_weight_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=True ),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True ),
            nn.Softmax(dim=-1) 
        )
        
        self.tr_embedding = tr_embedding(config.hidden_size)
        
        self.gate_weight = nn.Linear(config.hidden_size, config.hidden_size)
        
      

       

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)
       

    def forward(self, signal_vectors,  xyz_vectors, noise,labels):
       
        
        batch, num_timepoints_per_node,num_voxels = signal_vectors.shape  

       
        pred_len = int(num_timepoints_per_node*0.3) 
        first_end = num_timepoints_per_node-pred_len 
        
        
        time_indices = torch.arange(0, num_timepoints_per_node,device=device).float() 
       
        time_indices = time_indices.unsqueeze(0).repeat(batch, 1)  
        
        TR = labels.unsqueeze(-1).to(device)

      
        tr_sequence = time_indices * TR 
       
        tr_sequence = tr_sequence.unsqueeze(-1)
        tr_embedded = self.tr_embedding(tr_sequence) 
       
        x_emb = self.signal_embedding_projection(signal_vectors)
        
        x = self.pos_embedding(x_emb)  #### b,2604,1,256
        
        x = tr_embedded + x
        
        
        gate = torch.sigmoid(self.gate_weight(x))  # 形状仍是 (b, T, R)
        x_weighted = gate * x
        
        l1_regularization = 1e-5 * torch.norm(self.gate_weight.weight, 1)
       
        
        embeddings, mask, ids_restore = self.random_masking(x_weighted, first_end,pred_len,noise=noise)
        
        

        cls_tokens = self.cls_token.expand(embeddings.shape[0], -1, -1)
        
        embeddings = torch.cat((cls_tokens,embeddings), dim=1)  
        
        
      
        return embeddings, mask, ids_restore,first_end,pred_len,l1_regularization,x_emb
    

    def random_masking(self, sequence, first_end,pred_len,noise=None):
           
            
         
            
            
            all_len = first_end+pred_len
           
            
            sequence2 = sequence.clone()
            batch_size, seq_length, dim = sequence2.shape
            
            
            seq_len70 = first_end-0
            seq_len30 = pred_len-0
            
            seq_70 = sequence2[:,:seq_len70,:]
            seq_30 = sequence2[:,seq_len70:,:]

            if noise is None:
                noise = torch.rand(batch_size, seq_len70, device=sequence.device)  
                noise,_ = torch.sort(noise)


            ids_shuffle_70 = torch.argsort(noise, dim=1)  
            ids_restore_70 = torch.argsort(ids_shuffle_70, dim=1)
            ids_restore_30 = torch.arange(seq_len70,seq_length,device=sequence.device)
            ids_restore_30 = ids_restore_30.unsqueeze(0).repeat(batch_size, 1)
            
            ids_restore = torch.cat((ids_restore_70,ids_restore_30),dim=1)
            
         
            ids_keep_70 = ids_shuffle_70[:, :int(seq_len70*1)]
            sequence_unmasked = torch.gather(seq_70, dim=1, index=ids_keep_70.unsqueeze(-1).repeat(1, 1, dim))
            
            
            mask = torch.ones([batch_size, all_len], device=sequence.device)
            mask[:, :seq_len70] = 0
         
            mask = torch.gather(mask, dim=1, index=ids_restore)

            return sequence_unmasked, mask, ids_restore



    
    
class BrainLMEncoder(ViTMAEEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [NystromformerLayer(config) for _ in range(config.num_hidden_layers)] 
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    # layer_head_mask,  Nystromformer doesn't accept head_mask
                )
            else:
               
                # Nystromformer attention layer does not accept head_mask parameter
                layer_outputs = layer_module(
                    hidden_states, output_attentions=output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            
        
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BrainLMModel(ViTMAEModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BrainLMEmbeddings(config)  
        self.encoder = BrainLMEncoder(config)

        self.post_init()

    def forward(
        self,
        signal_vectors: torch.Tensor = None,
        xyz_vectors: torch.Tensor = None,
        labels: torch.Tensor = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        noise: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEModelOutput]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        

         
        
        embedding_output, mask, ids_restore,first_end,pred_len,L1,emb = self.embeddings(
            signal_vectors,  xyz_vectors,noise,labels
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print(encoder_outputs[0].shape)
        # print(encoder_outputs[1][0])
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask) + encoder_outputs[1:]

        return ViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ),first_end,pred_len,L1,emb

    
class BrainLMDecoder(ViTMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)
        self.decoder_pos_embed = None  
        self.num_brain_voxels = 1000
        self.mask_ratio = config.mask_ratio
        self.timepoint_patching_size = config.timepoint_patching_size
        self.use_tanh_decoder = config.use_tanh_decoder
        
        self.transformer_layer = CustomTransformerLayer(512, config.decoder_num_attention_heads, 2, 0,self.training,51)  


       
        self.pos_embedding = PositionalEncoding(d_model=config.hidden_size)
        self.tr_embedding = tr_embedding(config.hidden_size)

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [
                NystromformerLayer(decoder_config)
                for _ in range(config.decoder_num_hidden_layers)
            ]
        )
        
        if self.use_tanh_decoder:
            self.decoder_pred_nonlinearity2 = nn.Tanh()

        self.initialize_weights(num_patches)

    def initialize_weights(self, _):
       
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,

        ids_restore,
        pred_len,
        labels,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    ):
        
        x = self.decoder_embed(hidden_states)

        # Unflatten sequence
        batch_size, flatten_seq_len, hidden_dim = x[:, 1:, :].shape
        num_mask_tokens = ids_restore.shape[1] - flatten_seq_len

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(batch_size, num_mask_tokens, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token  
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, hidden_dim)
        )  # unshuffle
       
        x_ = self.pos_embedding(x_)
        
        time_indices = torch.arange(0, x_.shape[1],device=device).float()  # 
     
        time_indices = time_indices.unsqueeze(0).repeat(batch_size, 1)  
     
        TR = labels.unsqueeze(-1).to(device) 

       
        tr_sequence = time_indices * TR 
     
        tr_sequence = tr_sequence.unsqueeze(-1)
        tr_embedded = self.tr_embedding(tr_sequence)  
        
        x_ = x_ + tr_embedded


        hidden_states = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token  
        

      
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    # None,  Nystromformer layer does not accept argument head_mask
                )
            else:
                # layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)
                # Nystromformer layer does not accept argument head_mask
                layer_outputs = layer_module(
                    hidden_states, output_attentions=output_attentions
                )
              

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

     
        if self.use_tanh_decoder:
            logits = self.decoder_pred_nonlinearity2(logits)


        if not return_dict:
          
            return tuple(
                v
                for v in [logits, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return ViTMAEDecoderOutput(
            logits=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )    


class BrainLMForPretraining(ViTMAEForPreTraining):
  
    def __init__(self, config):
        super().__init__(config)
        self.vit = BrainLMModel(config)
        self.decoder = BrainLMDecoder(
            config, num_patches=self.vit.embeddings.num_patches
        )

        
        
        
        self.decoder_pred = nn.Linear(
            in_features=1024, 
            out_features=1000,     
            bias=False,
        )

        self.post_init()

    def init_weights(self):
        
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

       
        self.apply(self._initialize_weights)

        self.tie_weights()

    def _init_weights(self, module):  #
        if isinstance(module, nn.Linear):
           
            torch.nn.init.xavier_uniform_(module.weight)
           
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
            #     torch.nn.init.xavier_uniform_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.kaiming_uniform_(module.weight)

    def forward_loss(self, signal_values, pred_values,mask):
       
        assert signal_values.shape == pred_values.shape
        

        if self.config.loss_fn == "mse":
            loss = (
                ((pred_values - signal_values) ** 2) * mask
            ).sum() / mask.sum()  # MSE
            
            
        elif self.config.loss_fn == "mae":
            loss = abs((pred_values - signal_values) * mask).sum() / mask.sum()  # MAE
        else:
            raise NotImplementedError("Unknown loss function specified.")

        return loss


     
    

    def forward(
        self,
        signal_vectors: torch.Tensor = None,
        signal_vectors1: torch.Tensor = None,
        xyz_vectors: torch.Tensor = None,
        labels: torch.Tensor = None,  # not used
        input_ids: torch.Tensor = None,  # not used, 
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        noise: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEForPreTrainingOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
     
            outputs = self.vit(
            signal_vectors=signal_vectors,
            xyz_vectors=xyz_vectors,
            labels = labels,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            noise=noise,
        ) 
      
        outputs1 = outputs[0]
        first_end = outputs[1]
        pred_len = outputs[2]
        L1_loss = outputs[3]
        emb = outputs[4]
   
        ids_restore = outputs1.ids_restore
        mask = outputs1.mask
        latent_all = outputs1.hidden_states
        latent = outputs1.last_hidden_state  
        
        
        decoder_outputs = self.decoder(latent,  ids_restore,pred_len,labels)     
        logits = (
            decoder_outputs.logits 
        )  # 
        logits = self.decoder_pred(logits)[:,1:,:]
        
        mask2 = mask.unsqueeze(-1).repeat(1,1, 1000)
        
        
        
        mask4 = mask2.clone()
        mask4 = mask4-1
        mask4 = torch.where(mask4 == -1, torch.tensor(1), mask4)
        
        # print(signal_vectors.shape, logits.shape)
        
        loss =  0.75*self.forward_loss(signal_vectors, logits,mask2)+0.25*self.forward_loss(signal_vectors, logits,mask4) 
        
        

    
        mask3 = mask2.transpose(2,1)

        if not return_dict:
            output = (logits1, mask) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        
        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=(logits, latent),
            mask=mask3,
            hidden_states=outputs[0].hidden_states,
            attentions=outputs[0].attentions,
        )
        
       
    
    
    
    
    
    
