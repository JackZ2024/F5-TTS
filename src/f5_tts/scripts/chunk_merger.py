import os

import click
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def remove_silence(audio, silence_threshold=-40, chunk_size=10):
    """
    移除音频首尾静音
    :param audio: 输入音频（AudioSegment 对象）
    :param silence_threshold: 静音阈值 (默认 -40 dB)
    :param chunk_size: 静音检测的块大小 (默认 10ms)
    :return: 去除静音后的音频
    """
    non_silent_ranges = detect_nonsilent(audio, min_silence_len=chunk_size, silence_thresh=silence_threshold)
    if not non_silent_ranges:
        return audio  # 如果没有检测到声音，则返回原始音频
    start, end = non_silent_ranges[0][0], non_silent_ranges[-1][1]
    return audio[start:end]


def create_silence(duration_ms):
    """
    创建指定时长的静音音频
    :param duration_ms: 静音时长（毫秒）
    :return: 静音音频（AudioSegment 对象）
    """
    return AudioSegment.silent(duration=duration_ms)


silence_1s = create_silence(1000)
silence_500ms = create_silence(500)
silence_200ms = create_silence(200)

punc_pause_map = {
    ".": 1000,
    "?": 1000,
    "!": 1000,
    ";": 1000,
    ",": 500,
    ":": 500,
}

custom_pause_map = {
    "0010": 300,
    "0020": 300,
    "0024": 300,
    "0049": 300,
    "0072": 300,
}


def concatenate_audio_with_labels(files, inputs):
    """
    根据标签拼接音频
    :param files: 音频文件路径列表
    :param inputs: 标签列表
    :param silence_500ms: 500ms 静音片段
    :param silence_1s: 1s 静音片段
    :return: 拼接后的音频
    """
    combined_audio = AudioSegment.empty()
    combined_audio += silence_1s  # 音频最开头留1秒
    for file, label in zip(files, inputs):
        # 加载音频
        audio = AudioSegment.from_file(file)
        # 去除首尾静音
        trimmed_audio = remove_silence(audio)
        # 添加到拼接音频
        combined_audio += trimmed_audio

        order_num = os.path.basename(file).split("_")[0]
        if order_num in custom_pause_map:
            print("自定义间距", order_num)
            combined_audio += create_silence(custom_pause_map[order_num])
        else:     # 根据句末标点
            if label[-1] in punc_pause_map:
                combined_audio += create_silence(punc_pause_map[label[-1]])
            else:
                combined_audio += create_silence(150)  # 不是标点

    return combined_audio


@click.command
@click.option("--gen_file")
@click.option("--chunk_folder")
def merge(gen_file, chunk_folder):
    texts = open(gen_file, 'r', encoding='utf-8').read().split("\n")
    files = os.listdir(chunk_folder)
    files = [os.path.join(chunk_folder, file) for file in files if file.endswith(".wav")]
    result_audio = concatenate_audio_with_labels(files, texts)
    result_audio.export("output.wav", format="wav")


if __name__ == '__main__':
    merge()
