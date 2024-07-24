FROM runpod/base:0.6.2-cuda11.1.1

WORKDIR /

RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    ffmpeg
RUN pip install uv
#Too lazy to do requirements.txt
RUN uv venv
# RUN /bin/bash -c "source /venv/bin/activate"
# RUN source venv/bin/activate
# "huggingface_hub[cli]"
RUN pip install -U "huggingface_hub[cli]" runpod transformers datasets evaluate accelerate pyaudio moviepy ffmpeg-python sentencepiece openai-whisper accelerate
RUN git clone https://github.com/Camb-ai/MARS5-TTS.git
# RUN python -u src/init_models.py
ADD src .

CMD [ "python", "-u", "/worker.py" ]