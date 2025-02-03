import os
import shutil

import click
import librosa
import soundfile as sf
from tqdm import tqdm


@click.command
@click.option("--ljspeech_root")
def convert_ljspeech_to_libritts(ljspeech_root):
    """
    Convert LJSpeech dataset to LibriTTS format.
    :param ljspeech_root: Path to the LJSpeech dataset root.
    :param libritts_root: Path to the output LibriTTS dataset root.
    """
    libritts_root = ljspeech_root+ "-libritts"
    if os.path.isdir(libritts_root):
        shutil.rmtree(libritts_root)
    os.makedirs(libritts_root, exist_ok=True)

    transcript_path = os.path.join(ljspeech_root, "metadata.csv")
    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    speaker_id = "0001"  # LJSpeech has only one speaker
    split = "train-clean-100"  # Default split name
    speaker_dir = os.path.join(libritts_root, split, speaker_id)
    os.makedirs(speaker_dir, exist_ok=True)

    wav_dir = os.path.join(speaker_dir, "")
    txt_dir = os.path.join(speaker_dir, "")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    for line in tqdm(lines, desc="Processing LJSpeech dataset"):
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue

        file_id, transcript = parts
        file_id = file_id.replace(".wav", "")
        src_wav_path = os.path.join(ljspeech_root, "wavs", f"{file_id}.wav")
        tgt_wav_path = os.path.join(wav_dir, f"{file_id}.wav")
        tgt_txt_path = os.path.join(txt_dir, f"{file_id}.normalized.txt")

        # Convert to 16kHz sampling rate (LibriTTS requirement)
        y, sr = librosa.load(src_wav_path, sr=16000)
        sf.write(tgt_wav_path, y, 16000)

        # Write transcript
        with open(tgt_txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(transcript)

    print("Conversion completed!")

if __name__ == '__main__':
    convert_ljspeech_to_libritts()
