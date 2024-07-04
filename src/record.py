import pyaudio
import wave
import ffmpeg

# Parameters
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 2              # 2 channels for stereo
RATE = 44100              # 44.1kHz sampling rate
CHUNK = 1024              # 1024 samples per frame
RECORD_SECONDS = 10       # Duration to record
WAVE_OUTPUT_FILENAME = "output.wav"
MP4_OUTPUT_FILENAME = "output.mp4"

# Initialize pyaudio
audio = pyaudio.PyAudio()

# Create the audio stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

frames = []

# Record audio in chunks
for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording finished.")

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save the recorded data as a WAV file
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print("Audio data saved as WAV file.")

# Convert the WAV file to MP4 using ffmpeg
input_audio = ffmpeg.input(WAVE_OUTPUT_FILENAME)
ffmpeg.output(input_audio, MP4_OUTPUT_FILENAME).run(overwrite_output=True)

print(f"Audio data converted to MP4 file: {MP4_OUTPUT_FILENAME}")
