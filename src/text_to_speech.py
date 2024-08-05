import sys
import os
import librosa
import soundfile as sf
import torch
from functools import lru_cache
import init_models

dupa_dir = os.getcwd() + "/src/MARS5-TTS"
sys.path.append(dupa_dir)

from inference import Mars5TTS, InferenceConfig as config_class

BASE_VOICE_PATH = "maklowicz.wav"
OUTPUT_PATH = "output_audio.wav"


class TextToSpeech:
    def __init__(
        self,
        voice_to_replicate=BASE_VOICE_PATH,  # TODO: validate input
    ):
        self._wav_path = ""
        self._output_tensor = None
        self._voice_to_replicate = voice_to_replicate
        self.mars5 = init_models().mars

    @staticmethod
    @lru_cache(maxsize=1)
    def load_model(self):
        return Mars5TTS.from_pretrained("CAMB-AI/MARS5-TTS")

    def toSpeech(self, text):
        wav, sr = librosa.load(self._voice_to_replicate, sr=self.mars5.sr, mono=True)
        wav = torch.from_numpy(wav)

        deep_clone = True
        cfg = config_class(
            deep_clone=deep_clone,
            rep_penalty_window=100,
            top_k=100,
            temperature=0.7,
            freq_penalty=3,
        )

        ar_codes, self._output_tensor = self.mars5.tts(
            "The quick brown rat.",
            wav,
            text,
            cfg=cfg,  # TODO: what the fuck is that
        )
        # output_audio is (T,) shape float tensor corresponding to the 24kHz output audio.

    def get_wav(self):
        # Save as .wav file
        sf.write(OUTPUT_PATH, self._output_tensor.numpy(), samplerate=24000)
        return self._wav_path

    def get_tensor(self):
        return self._output_tensor


if __name__ == "__main__":
    speech = TextToSpeech()
    speech.toSpeech(text="Hejka, co tam")
    print(
        f"Base Inference for text_to_speech.py is in path: {os.getcwd() + speech.get_wav()}"
    )
