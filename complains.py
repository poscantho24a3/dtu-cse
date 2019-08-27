import listen_helper as lh
from text import NNClassification
from config import *
import json
import random
import speak_helper as sh


class Complains():
    def __init__(self):
        self.listener = lh.Listen_Helper()
        self.text_classification = NNClassification(DEFAULT_PATH_PROCESSING)
        self.path_response = DEFAULT_PATH_COMPLAINT_RESPONSE
        self.speaker = sh.Speak_Helper()
        pass
    
    def processing(self, text=""):
        # complain_text = self.listener.listen()
        # convert speech to text successfully
        complain_text = text
        print("????",complain_text)
        if complain_text != "":
            type_complain = self.text_classification.classify(complain_text)
            print("type",type_complain)
            return self.create_response(type_complain)
            pass
        # convert speech to text unsuccessfully
        else:
            pass

    def create_response(self, type_complain):
        with open(self.path_response, encoding='utf-8') as data_file:
            data = json.load(data_file)
            response = [text["sentence"] for text in data]
            indexes = range(0, len(response))
            random_index = random.choice(indexes)
            random_response = response[random_index]
            print(random_response)

            audio_path = DEFAULT_FOLDER_AUDIO_COMPLAINT + '/{}.mp3'.format(random_index)

            self.speaker.play_audio_file(audio_path)

            return random_response
        pass

if __name__ == "__main__":
    complain = Complains()
    complain.processing()