from __future__ import annotations

import datetime
import os
from typing import Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from einops import rearrange
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)
from torch.utils.tensorboard import SummaryWriter

from f5_tts.model.dur import DurationPredictor


# reference: https://github.com/lucasnewman/f5-tts-mlx/blob/4d24ebcc0c7c6215d64ed29eabdd32570084543f/f5_tts_mlx/duration_trainer.py

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


class DurationTrainer:
    def __init__(
            self,
            model: DurationPredictor,
            num_warmup_steps: int = 1000,
            max_grad_norm: float = 1.0,
            log_dir: str = "runs/f5tts_duration",
    ):
        self.accelerator = Accelerator()
        self.model = model
        self.num_warmup_steps = num_warmup_steps
        self.max_grad_norm = max_grad_norm
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model = self.accelerator.prepare(self.model)

    def save_checkpoint(self, step: int, val_loss: Optional[float] = None):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': step,
            'val_loss': val_loss
        }
        torch.save(
            checkpoint,
            os.path.join(self.checkpoint_dir, f"f5tts_duration_{step}.pt")
        )

    def load_checkpoint(self, step: int):
        checkpoint = torch.load(
            os.path.join(self.checkpoint_dir, f"f5tts_duration_{step}.pt"),
            map_location="cpu"
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('val_loss')

    @torch.no_grad()
    def validate(self, val_dataset) -> float:
        self.model.eval()
        total_loss = 0
        total_samples = 0

        for batch in val_dataset:
            text_inputs = batch["text"]
            mel_spec = batch["mel"].permute(0, 2, 1)
            mel_lens = batch["mel_lengths"]

            loss = self.model(
                mel_spec,
                text=text_inputs,
                lens=mel_lens,
                return_loss=True
            )

            total_loss += loss.item()
            total_samples += 1

        self.model.train()
        return total_loss / total_samples

    def train(
            self,
            train_dataloader,
            val_dataloader,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-2,
            total_steps: int = 100_000,
            log_every: int = 100,
            save_every: int = 1000,
            validate_every: int = 1000,
            checkpoint: Optional[int] = None,
    ):
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Setup learning rate scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-8 / learning_rate,
            end_factor=1.0,
            total_iters=self.num_warmup_steps
        )

        decay_steps = total_steps - self.num_warmup_steps
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=decay_steps,
            eta_min=1e-8
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.num_warmup_steps]
        )

        self.scheduler, self.optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(self.scheduler,
                                                                                                    self.optimizer,
                                                                                                    train_dataloader,
                                                                                                    val_dataloader)

        # Load checkpoint if specified
        start_step = 0
        best_val_loss = float('inf')
        if checkpoint is not None:
            val_loss = self.load_checkpoint(checkpoint)
            if val_loss is not None:
                best_val_loss = val_loss
            start_step = checkpoint

        global_step = start_step
        training_start_date = datetime.datetime.now()
        log_start_date = datetime.datetime.now()

        self.model.train()

        for batch in train_dataloader:
            text_inputs = batch["text"]
            mel_spec = batch["mel"].permute(0, 2, 1)
            mel_lens = batch["mel_lengths"]

            # Forward pass
            self.optimizer.zero_grad()
            loss = self.model(
                mel_spec,
                text=text_inputs,
                lens=mel_lens,
                return_loss=True
            )

            # Backward pass
            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            # Logging
            self.writer.add_scalar('Loss/train', loss.item(), global_step)
            self.writer.add_scalar(
                'Learning_rate',
                self.scheduler.get_last_lr()[0],
                global_step
            )
            self.writer.add_scalar(
                'Batch_length',
                mel_lens.sum().item(),
                global_step
            )

            if global_step > 0 and global_step % log_every == 0:
                elapsed_time = datetime.datetime.now() - log_start_date
                log_start_date = datetime.datetime.now()
                if self.accelerator.is_local_main_process:
                    self.accelerator.log({
                        "loss": loss.item(),
                        "sec_per_step": elapsed_time.seconds / log_every
                    }, step=global_step)

                    # Validation
                    if global_step % validate_every == 0:
                        val_loss = self.validate(val_dataloader)
                        self.writer.add_scalar('Loss/val', val_loss, global_step)
                        self.accelerator.log({
                            "val loss": val_loss,
                            "sec_per_step": elapsed_time.seconds / log_every
                        }, step=global_step)

                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_checkpoint(
                                global_step,
                                val_loss=val_loss
                            )
                            self.accelerator.log({
                                "best val loss": val_loss,
                                "sec_per_step": elapsed_time.seconds / log_every
                            }, step=global_step)

                    # Regular checkpoint saving
                    if global_step % save_every == 0 and global_step > 0:
                        self.save_checkpoint(global_step)

                    global_step += 1

                    if global_step >= total_steps:
                        break

        self.writer.close()
        self.accelerator.end_training()
