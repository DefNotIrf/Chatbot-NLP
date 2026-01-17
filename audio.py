# ---------------------------------------------------------
# Voice IO (Microphone + Speaker)
# ---------------------------------------------------------
import subprocess
import time
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import edge_tts
import asyncio
import pygame
import os

class VoiceIO:
    def __init__(self):
        self.stt_model = WhisperModel("base", device="cpu", compute_type="int8")

    def get_input(self):
        print("\n Listening (5s)...")
        fs = 16000
        duration = 5
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        
        sd.stop() 
        time.sleep(0.5)
        
        filename = "temp_audio.wav"
        wav.write(filename, fs, recording)
        
        segments, _ = self.stt_model.transcribe(filename, beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        
        print(f"You said: {text}")
        return text

    def output(self, text):
        print(f"Bot: {text}")
        spoken_text = text.replace("MYR", "ringgit").replace("myr", "ringgit").replace("RM", "ringgit")
        OUTPUT_FILE = "response.mp3"
        communicate = edge_tts.Communicate(text, "en-SG-LunaNeural")
        asyncio.run(communicate.save(OUTPUT_FILE))
        
        pygame.mixer.init()
        pygame.mixer.music.load(OUTPUT_FILE)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        pygame.mixer.quit()
        os.remove(OUTPUT_FILE)

# ---------------------------------------------------------
# Text IO (Keyboard + Screen)
# ---------------------------------------------------------
class TextIO:
    def get_input(self):
        return input("\nYou (Type): ").strip()

    def output(self, text):
        print(f"Bot: {text}")