# origin: https://github.com/lucasnewman/f5-tts-mlx/issues/17#issuecomment-2453045703

import os.path
from pathlib import Path

import click
from torch.optim import AdamW
from torch.utils.data import DataLoader

from f5_tts.model.dataset import load_dataset, collate_fn, TextAudioDataset
from f5_tts.model.dur import DurationPredictor, DurationTransformer
from f5_tts.model.dur_trainer import DurationTrainer
from f5_tts.model.utils import get_tokenizer

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'


@click.command
@click.option("--dataset_folder")
def main(dataset_folder):

    vocab = {v: i for i, v in enumerate(Path(os.path.join(dataset_folder, "vocab.txt")).read_text(encoding='utf-8').split("\n"))}

    train_dataset = TextAudioDataset(
        folder=dataset_folder,
        audio_extensions=["wav"],
        max_duration=44
    )

    duration_predictor = DurationPredictor(
        transformer=DurationTransformer(
            dim=512,
            depth=8,
            heads=8,
            text_dim=512,
            ff_mult=2,
            conv_layers=2,
            text_num_embeds=len(vocab) - 1,
        ),
        vocab_char_map=vocab,
    )
    print(f"Trainable parameters: {sum(p.numel() for p in duration_predictor.parameters() if p.requires_grad)}")

    optimizer = AdamW(duration_predictor.parameters(), lr=7.5e-5)

    trainer = DurationTrainer(
        duration_predictor,
        optimizer,
        num_warmup_steps=5000,
        accelerate_kwargs={"mixed_precision": "fp16", "log_with": "wandb"}
    )

    epochs = 300
    max_batch_tokens = 16_000

    print("Training...")

    trainer.train(train_dataset, epochs, max_batch_tokens, num_workers=0, save_step=5000)


if __name__ == '__main__':
    main()
