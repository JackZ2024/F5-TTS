import click
from torch.utils.data import DataLoader

from f5_tts.model.dataset import load_dataset, collate_fn
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

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        batch_size=8,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        num_workers=16,
        pin_memory=False,
        persistent_workers=True,
        batch_size=8,
        shuffle=True,
    )

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
    trainer.train(train_dataloader, test_dataloader)


if __name__ == '__main__':
    main()
