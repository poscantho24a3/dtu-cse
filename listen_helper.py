import speech_recognition as sr

class Listen_Helper():

    def __init__(self, language="vi-VN"):
        self.language = language

    def listen(self, language="vi-VN"):
        # Record Audio
        r = sr.Recognizer()
        with sr.Microphone() as source:
            # audio = r.listen(source, phrase_time_limit=3)
            r.adjust_for_ambient_noise(source) 
            print("Listening...")
            audio = r.listen(source)
        try:
            # defualt language is English, en
            result = r.recognize_google(audio, language=self.language)
            print(result)
            return result

        except sr.UnknownValueError:
            print("could not understand audio")

            return ""
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

            return ""