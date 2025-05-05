from typing import Dict, Union, Any

import torch
import torch.nn as nn
import wandb
from transformers import Trainer

from utils.plots import plot_masked_pred_trends_one_sample


class BrainLMTrainer(Trainer):
    # --- Overwrite Log Function to Log Metrics to Weights & Biases ---#
    def log(self, logs: Dict[str, float]) -> None:
        """
        Custom log function overriding default log function of Huggingface Trainer class.
        This function received various metrics objects during training, and logs metrics
        to wandb.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        logs["epoch"] = int(logs["epoch"])

        if logs["epoch"] > self.compute_metrics.current_epoch:
            # Update epoch in metrics calculator, plot gene embedding once per epoch.
            # Plotting gene embedding here because metrics calculator doesn't have access to model.
            self.compute_metrics.current_epoch = logs["epoch"]

        output = {**logs, **{"step": self.state.global_step}}
        if self.args.wandb_logging:
            wandb.log(output, step=logs["epoch"])
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)


        if self.state.global_step % self.args.eval_steps == 0 and self.state.global_step>0:
            signal_vectors = inputs["signal_vectors"]
            mask = outputs["mask"]
            pred_logits = outputs["logits"][0]
            signal_vectors = torch.reshape(signal_vectors, shape=pred_logits.shape)
            pred_logits = pred_logits

            
            
            pred_logits=pred_logits.transpose(2,1).unsqueeze(-1)
            signal_vectors=signal_vectors.transpose(2,1).unsqueeze(-1)  
            


            plot_masked_pred_trends_one_sample(
                pred_logits=pred_logits,
                signal_vectors=signal_vectors,
                mask=mask,
                sample_idx=0,
                node_idxs=[0, 50,100,150,180],
                dataset_split="train",
                epoch=self.state.epoch,
            )



        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
                elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()
