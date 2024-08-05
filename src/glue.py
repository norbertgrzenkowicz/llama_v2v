import speech_to_text
import text_to_llama
import text_to_speech
import record


def doStuff(SpeechPath: str):
    speechedText = speech_to_text.get_text_from_speech(speech=SpeechPath)
    llamaAns = text_to_llama.inference(
        messages=text_to_llama.define_message(message=speechedText)
    )
    llamaVoice = text_to_speech.TextToSpeech()
    llamaVoice.toSpeech(llamaAns)

    return llamaVoice.get_tensor()
