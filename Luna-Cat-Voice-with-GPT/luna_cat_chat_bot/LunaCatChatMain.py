# import module

import speech_recognition as sr
import openai
import time

# import python Lib

from Play_Wav import play_Wav
from speechToText_v2 import wavToText
from Text_To_Speech import text_to_wav
from Record_Speech import Record_Audio
from gpt_chat import chating_bot
from playYT import play_music

# load the api key of ChatGPT

openai.api_key = "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"

# create speech recognition client

listener = sr.Recognizer()

# file_path

listen_path = "meow2.wav"
answer_path = "answer.wav"
notice_path = "meow1.wav"

# Function trigger by keyword

def keyword_trigger():

    keyword = False

    try:

        with sr.Microphone() as source:

            print('Listening... ')
            voice = listener.listen(source)
            command = listener.recognize_google(voice)

            play_Wav(listen_path)

            command = command.lower()

            print(command)

            if 'hey luna' or 'hello luna' or 'hi luna' in command:

                keyword = True
                command = None

            else:
                keyword = False
                
    except:

        keyword = False
        pass

    return keyword

# Function of Chat Bot

def run_Luna():

    trigger = False
    result = False
    trans2text = False
    trans2wav = False
    ansGPT = False

    # call the keyword trigger function

    trigger = keyword_trigger()

    if trigger == True:

        """play_Wav(answer_path)

        time.sleep(1)

        play_Wav(notice_path)"""

        result = Record_Audio()

    
    if result == True:

        # Load the Audio File

        audio_file = 'temp.wav'

        text, lang_code ,trans2text = wavToText(audio_file)

        print(lang_code)

    if trans2text == True:

        if 'play' in text:

            song = text.replace('play', '')
             
            play_music(song)

        elif '播放' in text:
             
             song = text.replace('播放', '')

             play_music(song)

        else:
            answer, ansGPT = chating_bot(text , lang_code)

    if ansGPT == True:

            # Speak the text out with the AI output

            trans2wav = text_to_wav(answer, lang_code)

    if trans2wav == True:
            
            audio_path = "audio.wav"

            # Play the Auio File

            play_Wav(audio_path)
    
    # call the Chat Bot again, keep the bot listen to keyword

    time.sleep(1.5)

    run_Luna()

#while True:


if __name__ == "__main__":

    run_Luna()
