from functools import lru_cache
import transformers
import torch
import whisper
import os
import sys

LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

dupa_dir = os.getcwd() + "/src/MARS5-TTS"
sys.path.append(dupa_dir)
from inference import Mars5TTS, InferenceConfig as config_class


class load_models:
    def __init__(self):
        self.mars = self.load_mars()
        self.llama = self.load_llama()
        self.whisper = self.load_whisper()

    @staticmethod
    @lru_cache(maxsize=1)
    def load_mars(self):
        return Mars5TTS.from_pretrained("CAMB-AI/MARS5-TTS")

    @staticmethod
    @lru_cache(maxsize=1)
    def load_llama() -> transformers.Pipeline:
        return transformers.pipeline(
            "text-generation",
            model=LLAMA_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # tokenizer = AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    @lru_cache
    def load_whisper(model_type: str = "small") -> whisper.Whisper:
        return whisper.load_model(model_type)
