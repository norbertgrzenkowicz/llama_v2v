import whisper
from functools import lru_cache
from typing import Dict

SPEECH_PATH = "untitled.mp3"

@staticmethod
@lru_cache
def load_model(model_type: str = "small") -> whisper.Whisper:
    return whisper.load_model(model_type)

def get_text_from_speech(speech: str = SPEECH_PATH) -> str:
    whisper_model = load_model()
    result = whisper_model.transcribe(SPEECH_PATH)
    #TODO: log whole text
    return result["text"]

if __name__ == "__main__":
    print(get_text_from_speech())
