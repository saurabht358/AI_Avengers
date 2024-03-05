import googletrans
import gtts
import os


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

print(TranslatorAny("hello" , "marathi"))

def TranslatorEn(text):
    output_language = "en"

    text = str(text)


    # Translate the recognized text
    translator = googletrans.Translator()
    translation = translator.translate(text, dest=output_language).text
    return translation

print(TranslatorEn("namaskar"))

