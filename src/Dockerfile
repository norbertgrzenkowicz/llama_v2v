FROM python:3.12

WORKDIR /

RUN pip install uv
#Too lazy to do requirements.txt
RUN uv pip install -U "huggingface_hub[cli]" runpod install transformers datasets evaluate accelerate pyaudio moviepy ffmpeg-python sentencepiece openai-whisper accelerate
RUN git clone https://github.com/Camb-ai/MARS5-TTS.git
ADD whatever.py .

CMD [ "python", "-u", "src/init_models.py" ]