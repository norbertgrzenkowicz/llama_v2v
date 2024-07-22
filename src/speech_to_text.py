import init_models

SPEECH_PATH = "untitled.mp3"


def get_text_from_speech(speech: str = SPEECH_PATH) -> str:
    whisper_model = init_models().whisper
    result = whisper_model.transcribe(SPEECH_PATH)
    # TODO: log whole text
    return result["text"]


if __name__ == "__main__":
    print(get_text_from_speech())
