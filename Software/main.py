 

from MainLogic import intentAns
# import googletrans 
# import gtts 
import pyttsx3                          #pip install pyttsx3
import speech_recognition as sr         #pip install speech-recognition
import datetime                         
import pyautogui                        #pip install pyautogui
import os
import sys
import datetime                         
import pyautogui                        #pip install pyautogui
import os
from subprocess import call
             #pip install openai
# from SpeakRecog import *
from time import sleep
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QThread
  



# 



# def Language_code(input_language):
#     languages = {
        
#         "english": "en",
#         "hindi": "hi",
#         "marathi": "mr"
#     }

#     input_language_lower = input_language.lower()

#     if input_language_lower in languages:
#         return languages[input_language_lower]
#     else:
#         return "en"


# def TranslatorAny(text , lang):
#     language_code = Language_code(lang)
#     output_language = str(language_code)
#     # input_language = language_code
#     text = str(text)


#     # Translate the recognized text
#     translator = googletrans.Translator()
#     translation = translator.translate(text, dest=output_language).text
#     finalAns=f"\nMinesBot: {translation}"
#     ui.terminalPrint(finalAns)

# # print(TranslatorAny("hello" , "marathi"))

# def TranslatorEn(text):
#     output_language = "en"

#     text = str(text)


#     # Translate the recognized text
#     translator = googletrans.Translator()
#     translation = translator.translate(text, dest=output_language).text
#     from MainLogic import intentAns
#     ans= str(intentAns(translation))
#     TranslatorAny(ans,lang)

# # print(TranslatorEn("namaskar"))


from mainGUI import Ui_Widget

class chotuMain(QThread):
    def __init__(self):
        super(chotuMain, self).__init__()
    
    def takeCommand(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            ui.terminalPrint('Listening....')
            r.pause_threshold = 1
            r.energy_threshold = 50
            audio = r.listen(source, 0, 4)

        try:
            ui.terminalPrint("Recognizing....\n")
            query = r.recognize_google(audio, language='en-in')
            # ui.terminalPrint(f"You said : {query}")
            return query.lower()

        except:
            ui.terminalPrint("Say that again!")
            return ""

    def run(self):
        self.runChotu()

     

 
 
    def runChotu(self):
        # Query = self.takeCommand(self).lower()
        # if "wake up" in Query or "utho" in Query or "uth jao" in Query or "utre" in Query:
        # from GreetMe import greetMe
        # greetMe();
        text = self.takeCommand()
        ui.terminalPrint(f"\nUser: {text} ?\n")
        tex = text.lower()
        from MainLogic import intentAns
        ui.terminalPrint( intentAns(tex))
        
         
           

startExecution = chotuMain()
class guiOfChotu(QWidget):
    cpath = ""
    def __init__(self):
        super(guiOfChotu,self).__init__()
        self.chotuUI = Ui_Widget()
        self.chotuUI.setupUi(self)

        # self.chotuUI.pushButton.clicked.connect(self.close)
        self.greet()
        self.chotuUI.pushButton_2.clicked.connect(self.showcommand)
        self.chotuUI.pushButton.clicked.connect(self.run)
        
        

    def greet(self):
        self.terminalPrint("Hello.....\n I'm MineBot, an AI language model developed by AI Avengers. My purpose is to assist and provide information about mines rules and regulations. You can ask me questions, by entering text or giving the voice input.\n\n")

    def run(self):
        startExecution.start()
    def terminalPrint(self,text):
        self.chotuUI.plainTextEdit.appendPlainText(text)
    
    def showcommand(self):
        text = self.chotuUI.lineEdit.text()
        ui.terminalPrint(f"\nUser: {text} ?\n")
        tex = text.lower()
        from MainLogic import intentAns
        ui.terminalPrint( intentAns(tex))
        # TranslatorEn(tex)
         
    
     
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui =guiOfChotu()
    ui.show()
    sys.exit(app.exec_())
