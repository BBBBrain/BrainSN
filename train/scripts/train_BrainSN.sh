#!/bin/bash
date;hostname;pwd


python train.py \
    --output_dir training-runs/BrainSN\
    --train_dataset_path /root/autodl-tmp/BrainLM-main/DataSet/train_dir \
    --val_dataset_path /root/autodl-tmp/BrainLM-main/DataSet/test_dir \
    --hidden_size 1024 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --intermediate_size 1024 \
    --decoder_hidden_size 1024 \
    --decoder_num_hidden_layers 8 \
    --decoder_num_attention_heads 8 \
    --decoder_intermediate_size 1024 \
    --attention_probs_dropout_prob 0.1 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 15000 \
    --save_total_limit 50000 \
    --dataloader_num_workers 15 \
    --save_steps 1000 \
    --dataloader_pin_memory True

