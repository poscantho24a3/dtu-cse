from google_speech import Speech
import sys
import subprocess
from config import *
import os
from pydub import AudioSegment
from pydub.playback import play

import os

class Speak_Helper():

    # text: text to speech
    # language: code language for speech (en-US, vi-VN...)
    # speed: speed of speech

    def __init__(self, language=DEFAULT_LANGUAGE):
        self.language = language

    def speak(self, text):
        speech = Speech(text, self.language)
        # speech.play(sox_effects=("speed", DEFAULT_SPEED))
        PATH = os.getcwd() + r"\Audio\test.mp3"
        speech.save(PATH)
        # self.play_audio_file(PATH)
        song = AudioSegment.from_mp3(PATH)
        play(song)

    def speak_with_speed(self, text, speed):
        speech = Speech(text, self.language)
        speech.play(sox_effects=("speed", speed))

    def speak_and_save(self, text, path, index):
        speech = Speech(text, self.language)
        speech.play(sox_effects=("speed", DEFAULT_SPEED))

        if not os.path.isdir(path):
            os.mkdir(path)
            pass

        full_path = "{}/{}.mp3".format(path, index)
        print(full_path)
        speech.save(full_path)

    def play_audio_file(self, path):
        audio_data = open(path, 'rb').read()
        cmd = ["sox", "-q", "-t", "mp3", "-"]
        if sys.platform.startswith("win32"):
            cmd.extend(("-t", "waveaudio"))

        cmd.extend(("-d", "trim", "0.1", "reverse", "trim", "0.1", "reverse"))  # "trim", "0.25", "-0.1"

        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        p.communicate(input=audio_data)