import pyttsx3                         
import speech_recognition as sr    
from MainLogic import intentAns
import googletrans
import gtts
import os

def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    Id = r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0'
    engine.setProperty('voice',Id)
    engine.say(text=text)
    # print(f"{text}")
    engine.runAndWait()

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening....')
        r.pause_threshold = 1
        r.energy_threshold = 50
        audio = r.listen(source,0,8)
    try:
        print("Recognizing....\n")
        query = r.recognize_google(audio,language='en')
        # print(f"You said : {query}")
        return query.lower()
    
    except:
        return"None"



def Language_code(input_language):
    languages = {
        "english": "en",
        "hindi": "hi",
        "marathi": "mr"
    }

    input_language_lower = input_language.lower()

    if input_language_lower in languages:
        return languages[input_language_lower]
    else:
        return "en"


def TranslatorAny(text , lang):
    language_code = Language_code(lang)
    output_language = str(language_code)
    # input_language = language_code
    text = str(text)


    # Translate the recognized text
    translator = googletrans.Translator()
    translation = translator.translate(text, dest=output_language).text
    return translation

# print(TranslatorAny("hello" , "marathi"))

def TranslatorEn(text):
    output_language = "en"

    text = str(text)


    # Translate the recognized text
    translator = googletrans.Translator()
    translation = translator.translate(text, dest=output_language).text
    return translation

# print(TranslatorEn("namaskar"))

    


if __name__ == "__main__":
    while True:
        pass
