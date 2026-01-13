import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

def record_audio(
    filename="audio/user_input.wav",
    duration=5,
    sample_rate=16000
):
    print("Recording... Speak now")
    
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16
    )
    
    sd.wait()
    write(filename, sample_rate, audio)
    print("Recording finished")

if __name__ == "__main__":
    record_audio()
