import sys
import os

# os.system("chcp 65001>nul")
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__) + "/src")

from src.f5_tts.train import finetune_gradio

finetune_gradio.main()