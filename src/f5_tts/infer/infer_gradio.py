# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import json
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files
import os

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydub import AudioSegment
import gdown
import csv

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)


DEFAULT_TTS_MODEL = "F5-TTS"
tts_model_choice = DEFAULT_TTS_MODEL

# 在这里添加语言的中英对照，英文不区分大小写
languages = {"泰语":"thai", "默认":""}

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
    "默认",
]


# load models

vocoder = load_vocoder()


def load_f5tts():
    ckpt_path=str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
    F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


def load_e2tts():
    ckpt_path=str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path)

def get_url(ckpt_path):
    global models_dict
    model_url = ""
    vocab_url = ""
    for models_list in models_dict.values():
        for model_dict in models_list:
            if ckpt_path == model_dict["model"]:
                model_url = model_dict["model_url"]
                vocab_url = model_dict["vocab_url"]
                return model_url, vocab_url

    return model_url, vocab_url

def get_drive_id(url):
    """ 通过网盘文件url获取id """
    pattern = r"(?:https?://)?(?:www\.)?drive\.google\.com/(?:file/d/|folder/d/|open\?id=|uc\?id=|drive/folders/)([a-zA-Z0-9_-]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return url

def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    else:
        if not os.path.exists(ckpt_path):
            # 如果模型不存在，就根据链接下载
            model_url, vocab_url = get_url(ckpt_path)
            if model_url != "":
                file_id = get_drive_id(model_url)
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                gdown.download(id=file_id, output=ckpt_path, fuzzy=True)

    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    else:
        if not os.path.exists(vocab_path):
            # 如果vocab不存在，就根据链接下载
            model_url, vocab_url = get_url(ckpt_path)
            if vocab_url != "":
                file_id = get_drive_id(vocab_url)
                os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
                gdown.download(id=file_id, output=vocab_path, fuzzy=True)

    if model_cfg is None:
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)

def load_models_from_csv():
    csv_path = "./models.csv"
    models_dict = {}
    if not os.path.exists(csv_path):
        return models_dict
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # 名称	版本	语言	model链接	vocab链接
        for row in reader:
            model_name = row['名称'].strip()
            model_version = row['版本'].strip()
            model_lang = row['语言'].strip()
            model_url = row['model链接'].strip()
            vocab_rul = row['vocab链接'].strip()
            if model_name == "" or model_lang == "" or model_url == "" or vocab_rul == "":
                continue

            if model_version != "":
                model_version = "/" + model_version
            model_path = "./models/" + model_lang + model_version + "/" + model_name
            vocab_path = "./models/" + model_lang + model_version + "/vocab.txt"

            model_dict = {}
            model_dict["model"] = model_path.replace("\\", "/")
            model_dict["vocab"] = vocab_path.replace("\\", "/")
            model_dict["model_url"] = model_url
            model_dict["vocab_url"] = vocab_rul
            if model_lang in models_dict:
                models_dict[model_lang].append(model_dict)
            else:
                models_dict[model_lang] = [model_dict]

    return models_dict


def load_models_list():
    models_root_path = "./models"
    models_dict = {}
    model_dict = {}
    model_dict["model"] = "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"
    model_dict["vocab"] = "hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt"
    model_dict["model_url"] = ""
    model_dict["vocab_url"] = ""
    models_dict["默认"] = [model_dict]

    # 优先使用csv文件加载模型，放置第一次用了csv，第二次用的时候models已经存在了，就无法加载csv里的模型了
    csv_path = "./models.csv"
    if os.path.exists(csv_path):
        # 如果models文件夹不存在，说明不是在本地运行，那就到云端下载一份模型的列表，然后生成字典返回，等模型使用的时候再下载TODO
        models = load_models_from_csv()
        models_dict.update(models)
        return models_dict
    
    if not os.path.exists(models_root_path):
        return models_dict
    for folder in os.listdir(models_root_path):
        folder_path = models_root_path + "/" + folder
        if os.path.isdir(folder_path):
            language = folder
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for file in filenames:
                    if file.lower().endswith(".safetensors") or file.lower().endswith(".pt"):
                        model_path = dirpath + "/" + file
                        vocab_file = dirpath + "/vocab.txt"
                        if os.path.exists(vocab_file):
                            model_dict = {}
                            model_dict["model"] = model_path.replace("\\", "/")
                            model_dict["vocab"] = vocab_file.replace("\\", "/")
                            model_dict["model_url"] = ""
                            model_dict["vocab_url"] = ""
                            if language in models_dict:
                                models_dict[language].append(model_dict)
                            else:
                                models_dict[language] = [model_dict]

    return models_dict

def load_refs_list():
    refs_root_path = "./refs"
    refs_dict = {}
    if not os.path.exists(refs_root_path):
        # 如果refs文件夹不存在，那就到网盘下载一份，这里需要实现网盘下载refs文件夹的功能。TODO
        return refs_dict

    for folder in os.listdir(refs_root_path):
        folder_path = refs_root_path + "/" + folder
        if os.path.isdir(folder_path):
            language = folder

            ref_audio_dict = {}
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for file in filenames:
                    if file.lower().endswith(".wav") or file.lower().endswith(".mp3"):
                        ref_audio_path = dirpath + "/" + file
                        ref_txt_path = ref_audio_path[:-4] + ".txt"
                        if os.path.exists(ref_txt_path):
                            ref_audio_dict[ref_audio_path.replace("\\", "/")] = ref_txt_path.replace("\\", "/")

            refs_dict[language] = ref_audio_dict

    return refs_dict

# 组合生成的音频，并在中间根据参数添加静音
def get_final_wave(cross_fade_duration, generated_waves, final_sample_rate):
    final_wave = None
    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        # Simply concatenate
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            cross_fade_samples = int(cross_fade_duration * final_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )

            final_wave = new_wave

    return final_wave


F5TTS_ema_model = None
E2TTS_ema_model = None
custom_ema_model, pre_custom_path = None, ""

chat_model_state = None
chat_tokenizer_state = None

models_dict = load_models_list()
refs_dict = load_refs_list()


@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate response using Qwen"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_texts,
    model,
    remove_silence,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
    save_line_audio = False,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    lang = ""
    if "-" in tts_model_choice[4]:
        index = tts_model_choice[4].find("-")
        lang = tts_model_choice[4][index + 1:]

    if model == "F5-TTS":
        if F5TTS_ema_model is None:
            show_info("Loading F5-TTS model...")
            F5TTS_ema_model = load_f5tts()
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("Loading E2-TTS model...")
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model
    elif isinstance(model, list) and model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[1]:
            show_info("Loading Custom TTS model...")
            custom_ema_model = load_custom(model[1], vocab_path=model[2], model_cfg=model[3])
            pre_custom_path = model[1]
        ema_model = custom_ema_model

    # gen_wav_files = []
    gen_audio_path = "gen_audio"
    if not os.path.exists(gen_audio_path):
        os.mkdir(gen_audio_path)
    else:
        # 把里面的东西删除
        for file in os.listdir(gen_audio_path):
            try:
                os.remove(gen_audio_path + "/" + file)
            except:
                pass
    count = 0
    generated_waves = []
    spectrograms = []
    for i in range(len(gen_texts)):
        gen_text_box = gen_texts[i]
        if gen_text_box.strip() == "":
            continue
        gen_text_list = gen_text_box.split("\n")
        show_info(f"开始生成第 { i+ 1} 个文本框的音频")
        progress=gr.Progress()
        start_pos = count
        for j, gen_text in enumerate(progress.tqdm(gen_text_list, desc="Processing")):

            if gen_text.strip() == "":
                continue
            # gen_text = gen_text.replace(" ", ",")
            final_wave, final_sample_rate, combined_spectrogram = infer_process(
                ref_audio,
                ref_text,
                gen_text,
                ema_model,
                vocoder,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                speed=speed,
                show_info=show_info,
                # progress=gr.Progress(),
                lang=lang,
            )

            generated_waves.append(final_wave)
            spectrograms.append(combined_spectrogram)

            audio_filepath = gen_audio_path + f"/gen_audio_{count}.wav"
            count += 1
            if save_line_audio:
                sf.write(audio_filepath, final_wave, final_sample_rate, 'PCM_24')

        # 如果按行保存了，就不再按文本框保存，如果没有按行保存，那就按文本框保存
        if not save_line_audio:
            final_waves = get_final_wave(cross_fade_duration, generated_waves[start_pos:], final_sample_rate)
            audio_filepath = gen_audio_path + f"/gen_audio_{i}.wav"
            sf.write(audio_filepath, final_waves, final_sample_rate, 'PCM_24')

    # 导出合并后的24Khz音频
    last_gen_audio_path = "last_audio/gen_audio.wav"
    final_waves = None
    if len(generated_waves) > 0:
        if not os.path.exists("last_audio"):
            os.mkdir("last_audio")
        final_waves = get_final_wave(cross_fade_duration, generated_waves, final_sample_rate)
        sf.write(last_gen_audio_path, final_waves, final_sample_rate, 'PCM_24')

    # Remove silence
    if remove_silence and os.path.exists(last_gen_audio_path):
        remove_silence_for_generated_wav(last_gen_audio_path)
        final_waves, _ = torchaudio.load(last_gen_audio_path)
        final_waves = final_waves.squeeze().cpu().numpy()

    # Save the spectrogram
    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_waves), spectrogram_path, ref_text, last_gen_audio_path


def create_textboxes(num):
    try:
        num = int(num)
        if num <= 0:
            return [gr.update(visible=False) for _ in range(20)]
        
        # 控制输入框的可见性，最多支持 20 个
        updates = [gr.update(visible=True) if i < num else gr.update(visible=False) for i in range(20)]
        return updates
    except ValueError:
        return [gr.update(visible=False) for _ in range(20)]
    
def load_ref_txt(ref_txt_path):
    txt = ""
    if os.path.exists(ref_txt_path):
        with open(ref_txt_path, "r", encoding="utf8") as f:
            txt = f.read()
    return txt

with gr.Blocks() as app_credits:
    gr.Markdown("""
# Credits

* [mrfakename](https://github.com/fakerybakery) for the original [online demo](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) for initial chunk generation and podcast app exploration
* [jpgallegoar](https://github.com/jpgallegoar) for multiple speech-type generation & voice chat
""")
with gr.Blocks() as app_tts:
    gr.Markdown("# Batched TTS")
    ref_audio_input = gr.Textbox(label="Reference Audio")
    with gr.Row():
        num_input = gr.Textbox(label="请输入需要的输入框数量(1-20)", value="5")
        generate_textbox_btn = gr.Button("生成输入框")

    # 动态布局区域
    rows = []
    max_per_row = 5
    textboxes = []

    # 创建一个动态布局，最多 20 个输入框
    for i in range(4):  # 每行最多 5 个，4 行总共 20 个
        with gr.Row() as row:
            for j in range(max_per_row):
                index = i * max_per_row + j
                if index == 0:
                    textbox = gr.Textbox(label=f"生成文本:{index+1}", lines=10, visible=True)
                else:
                    textbox = gr.Textbox(label=f"生成文本:{index+1}", lines=10, visible=False)
                textboxes.append(textbox)
            rows.append(row)
    generate_textbox_btn.click(create_textboxes, inputs=[num_input], outputs=textboxes)

    generate_btn = gr.Button("合成", variant="primary")
    with gr.Accordion("高级设置", open=False):
        basic_ref_text_input = gr.Textbox(
            label="参考音频对应文本",
            info="如果留空则自动转录生成. 如果输入文本，则使用输入的文本，建议输入标准文本，转录出来的文本准确性可能不高。",
            lines=2,
            value="",
        )
        with gr.Row():
            remove_silence = gr.Checkbox(
                    label="删除静音",
                info="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.",
                value=False,
            )
            save_line_audio = gr.Checkbox(
                label="按行保存音频",
                info="勾选此项，中间结果会每行保存一个音频，不勾选，则每一个文本框保存一个音频。",
                value=False,
            )
        speed_slider = gr.Slider(
            label="语速设置",
            minimum=0.3,
            maximum=2.0,
            value=0.8,
            step=0.1,
            info="Adjust the speed of the audio.",
        )
        nfe_slider = gr.Slider(
            label="NFE Steps",
            minimum=4,
            maximum=64,
            value=64,
            step=2,
            info="Set the number of denoising steps.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="Set the duration of the cross-fade between audio clips.",
        )

    # audio_output = gr.Audio(label="合成音频")
    # spectrogram_output = gr.Image(label="Spectrogram")
    download_output = gr.Textbox(label="下载文件")

    @gpu_decorator
    def basic_tts(
        ref_audio_input,
        ref_text_input,
        remove_silence,
        save_line_audio,
        cross_fade_duration_slider,
        nfe_slider,
        speed_slider,
        *gen_texts_input,
    ):
        audio_out, spectrogram_path, ref_text_out, gen_audio_path = infer(
            ref_audio_input,
            ref_text_input,
            gen_texts_input,
            tts_model_choice,
            remove_silence,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
            save_line_audio=save_line_audio,
        )
        return gen_audio_path

    intputs = [ref_audio_input, basic_ref_text_input, remove_silence, save_line_audio, \
               cross_fade_duration_slider, nfe_slider, speed_slider] + textboxes

    generate_btn.click(
        basic_tts,
        inputs=intputs,
        outputs=[download_output],
    )

with gr.Blocks() as app:
    gr.Markdown(
        """
# 自定义 F5 TTS

这是我们修改过的F5-TTS，目前支持泰语音频的生成。

"""
    )

    
    last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info.txt")

    def load_last_used_custom():
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return DEFAULT_TTS_MODEL_CFG

    def switch_tts_model(new_choice):
        global tts_model_choice
        if new_choice == "Custom":  # override in case webpage is refreshed
            custom_ckpt_path, custom_vocab_path, custom_model_cfg, lang = load_last_used_custom()
            tts_model_choice = ["Custom", custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg), lang]
            return (
                gr.update(visible=True, value=custom_ckpt_path),
                gr.update(visible=True, value=custom_vocab_path),
                gr.update(visible=True, value=custom_model_cfg),
            )
        else:
            tts_model_choice = new_choice
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def set_custom_model(custom_ckpt_path, custom_vocab_path, custom_model_cfg, language):
        global tts_model_choice
        global models_dict
        model_dict_list = models_dict.get(language, [])
        for model_dict in model_dict_list:
            if model_dict["model"] == custom_ckpt_path:
                custom_vocab_path = model_dict["vocab"]
                break

        tts_model_choice = ["Custom", custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg), language]
        with open(last_used_custom, "w", encoding="utf-8") as f:
            f.write(custom_ckpt_path + "\n" + custom_vocab_path + "\n" + custom_model_cfg + "\n" + language + "\n")

        return gr.update(value=custom_vocab_path)

    def language_change(lang, custom_model_cfg):
        global models_dict
        global refs_dict
        model_dict_list = models_dict.get(lang, [])
        models = []
        vocabs = []
        for model_dict in model_dict_list:
            models.append(model_dict["model"])
            vocabs.append(model_dict["vocab"])
        if len(models) > 0:
            def_model = models[0]
        else:
            def_model = ""

        if len(models) > 0:
            def_vocab = vocabs[0]
        else:
            def_vocab = ""

        lang_alone = lang
        if "-" in lang_alone:
            index = lang.find("-")
            lang_alone = lang_alone[:index]

        ref_audio_dict = refs_dict.get(lang_alone, {})
        ref_audios = []
        ref_txts = []
        for key, value in ref_audio_dict.items():
            ref_audios.append(key)
            ref_txts.append(value)
        if len(ref_audios) > 0 and os.path.exists(ref_audios[0]):
            def_audio = ref_audios[0]
        else:
            def_audio = None

        if len(ref_txts) > 0:
            def_txt = load_ref_txt(ref_txts[0]).strip()
        else:
            def_txt = ""
        global tts_model_choice
        tts_model_choice = ["Custom", def_model, def_vocab, json.loads(custom_model_cfg), lang]
        return gr.update(choices=models, value=def_model), gr.update(choices=vocabs, value=def_vocab), \
                gr.update(value=def_audio),  gr.update(value=def_txt), gr.update(choices=ref_audios, value=def_audio)
    
    def ref_audio_change(lang, audio_path):
        global refs_dict
        lang_alone = lang
        if "-" in lang_alone:
            index = lang.find("-")
            lang_alone = lang_alone[:index]

        ref_audio_dict = refs_dict.get(lang_alone, {})
        def_audio = None
        def_txt = ""
        for key, value in ref_audio_dict.items():
            if key == audio_path:
                def_audio = key
                def_txt = load_ref_txt(value).strip()
                break

        return gr.update(value=def_audio),  gr.update(value=def_txt)

    with gr.Row():
        # if not USING_SPACES:
        #     choose_tts_model = gr.Radio(
        #         choices=[DEFAULT_TTS_MODEL, "E2-TTS", "Custom"], label="选择 TTS 模型", value="Custom"
        #     )
        # else:
        #     choose_tts_model = gr.Radio(
        #         choices=[DEFAULT_TTS_MODEL, "E2-TTS"], label="选择 TTS 模型", value=DEFAULT_TTS_MODEL
        #     )
        # 在这里添加新语言的支持，记得在languages里添加语言的英文对照
        language = gr.Dropdown(
            choices=list(models_dict.keys()), value="默认", label="语言", allow_custom_value=True
            )
        ref_audio = gr.Dropdown(
            choices=[], value="", label="参考音频", allow_custom_value=True
            )
    with gr.Row():
        custom_ckpt_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[0]],
            value=DEFAULT_TTS_MODEL_CFG[0],
            allow_custom_value=True,
            label="模型位置: model_1200k.safetensors",
            visible=True,
        )
        custom_vocab_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[1]],
            value=DEFAULT_TTS_MODEL_CFG[1],
            allow_custom_value=True,
            label="VOCAB 文件位置:  vocab.txt",
            visible=False,
        )
        custom_model_cfg = gr.Dropdown(
            choices=[
                DEFAULT_TTS_MODEL_CFG[2],
                json.dumps(dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)),
            ],
            value=DEFAULT_TTS_MODEL_CFG[2],
            allow_custom_value=True,
            label="Config: in a dictionary form",
            visible=False,
        )

    # choose_tts_model.change(
    #     switch_tts_model,
    #     inputs=[choose_tts_model],
    #     outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
    #     show_progress="hidden",
    # )
    custom_ckpt_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg, language],
        outputs=[custom_vocab_path],
        show_progress="hidden",
    )
    # custom_vocab_path.change(
    #     set_custom_model,
    #     inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg, language],
    #     show_progress="hidden",
    # )
    # custom_model_cfg.change(
    #     set_custom_model,
    #     inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg, language],
    #     show_progress="hidden",
    # )

    language.change(
        language_change,
        inputs=[language, custom_model_cfg],
        outputs=[custom_ckpt_path, custom_vocab_path, ref_audio_input, basic_ref_text_input, ref_audio],
        show_progress="hidden",
    )
    ref_audio.change(
        ref_audio_change,
        inputs=[language, ref_audio],
        outputs=[ref_audio_input, basic_ref_text_input],
        show_progress="hidden",
    )

    switch_tts_model("Custom")

    gr.TabbedInterface(
        [app_tts, app_credits],
        ["Basic-TTS", "Credits"],
    )


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=False, is_flag=True, help="Allow API access")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application, e.g. set "/myapp" or full URL for application served at "https://example.com/myapp".',
)
@click.option(
    "--inbrowser",
    "-i",
    is_flag=True,
    default=False,
    help="Automatically launch the interface in the default web browser",
)
def main(port, host, share, api, root_path, inbrowser):
    global app
    print("Starting app...")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=api,
        root_path=root_path,
        inbrowser=inbrowser,
    )


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()
