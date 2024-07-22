import speech_to_text
import text_to_llama
import text_to_speech
import record
import argparse
import subprocess


def doStuff() -> str:
    SpeechPath = record.record()
    speechedText = speech_to_text.get_text_from_speech(speech=SpeechPath)
    llamaAns = text_to_llama.inference(
        messages=text_to_llama.define_message(message=speechedText)
    )
    llamaVoice = text_to_speech.TextToSpeech()
    llamaVoice.toSpeech(llamaAns)

    return llamaVoice.get_wav()


def listenInVLC(path: str = ""):
    try:
        subprocess.run(["vlc", path])
    except ValueError:
        print(f"Error: The input file '{path}' isnt correct or isnt mp3/wav file.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama-v2v!")
    parser.add_argument("--output", type=str, help="The output file path")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )
    args = parser.parse_args()

    print(f"Output file path: {args.output}")
    if args.verbose:
        print("Verbose mode enabled")

    outputPath = doStuff()
    listenInVLC(outputPath)


if __name__ == "__main__":
    main()
