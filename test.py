from gtts import gTTS
from playsound import playsound
import os

mp3_path = "temp/test.mp3"
test_command = "hey google"

def recogCommand(): 
    return


def playCommand(command):
    print(command, mp3_path)
    tts = gTTS(command, lang='en')
    
    tts.save(mp3_path)
    playsound(mp3_path)
    os.remove(mp3_path)


def main():
    playCommand(test_command)

main()