from __future__ import annotations
import datetime
from typing import Optional

import click
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
import os

from f5_tts.model.dataset import load_dataset
from f5_tts.model.duration import DurationPredictor, DurationTransformer
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import get_tokenizer

# reference: https://github.com/lucasnewman/f5-tts-mlx/blob/4d24ebcc0c7c6215d64ed29eabdd32570084543f/f5_tts_mlx/duration_trainer.py

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


class DurationTrainer:
    def __init__(
            self,
            model: nn.Module,
            num_warmup_steps: int = 1000,
            max_grad_norm: float = 1.0,
            log_dir: str = "runs/f5tts_duration",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.num_warmup_steps = num_warmup_steps
        self.max_grad_norm = max_grad_norm
        self.writer = SummaryWriter(log_dir)

        # Assuming MelSpec is implemented elsewhere
        self.mel_spectrogram = MelSpec()

        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
            map_location=self.device
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

            mel_spec = rearrange(
                torch.tensor(batch["mel"], device=self.device),
                "b 1 n c -> b n c"
            )
            mel_lens = torch.tensor(
                batch["mel_lengths"],
                dtype=torch.int32,
                device=self.device
            )

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
            train_dataset,
            val_dataset,
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

        for batch in train_dataset:
            text_inputs = batch["text"]

            mel_spec = rearrange(
                torch.tensor(batch["mel"], device=self.device),
                "b 1 n c -> b n c"
            )
            mel_lens = torch.tensor(
                batch["mel_lengths"],
                dtype=torch.int32,
                device=self.device
            )

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

                print(
                    f"step {global_step}: loss = {loss.item():.4f}, "
                    f"sec per step = {(elapsed_time.seconds / log_every):.2f}"
                )

            # Validation
            if global_step % validate_every == 0:
                val_loss = self.validate(val_dataset)
                self.writer.add_scalar('Loss/val', val_loss, global_step)
                print(f"Validation loss at step {global_step}: {val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        global_step,
                        val_loss=val_loss
                    )
                    print(f"New best validation loss: {val_loss:.4f}")

            # Regular checkpoint saving
            if global_step % save_every == 0:
                self.save_checkpoint(global_step)

            global_step += 1

            if global_step >= total_steps:
                break

        self.writer.close()
        print(f"Training complete in {datetime.datetime.now() - training_start_date}")


target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'


@click.command
@click.option("--dataset_name")
def main(dataset_name):
    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )
    tokenizer = "pinyin"

    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    train_dataset, test_dataset = load_dataset(dataset_name, mel_spec_kwargs=mel_spec_kwargs)

    trainer = DurationTrainer(DurationPredictor(
        transformer=DurationTransformer(
            dim=512,
            depth=8,
            heads=8,
            text_dim=512,
            ff_mult=2,
            conv_layers=2,
            text_num_embeds=len(vocab_char_map) - 1,
        ),
        vocab_char_map=vocab_char_map,
    ))
    trainer.train(train_dataset, test_dataset)


if __name__ == '__main__':
    main()
