# import python module

import speech_recognition as sr
import pyaudio  
import wave 

# import python 

import speech_to_text 
from luna_record import Record_Audio

# file path

audio_wav = "temp.wav"
alert_wav = "alert.wav"

# define function to play the caution / alert sound
def play_wave():

    #define stream chunk   
    chunk = 1024 

    #open a wav format music 
    f = wave.open("alert.wav","rb") 

    #instantiate PyAudio  
    p = pyaudio.PyAudio() 

    #open stream  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate = f.getframerate(),  
                    output = True)  

    #read data  
    data = f.readframes(chunk)  

    #play stream  
    while data:
            
            stream.write(data)  

            data = f.readframes(chunk)  

    #stop stream  
    stream.stop_stream()

    stream.close()  

    f.close()

    #close PyAudio  
    p.terminate()

# Function of keyword alert

def keyword_trigger(text, lang_code):

    keyword = False

    print(text, lang_code)

    if lang_code == "yue-hant-hk":

        print("You are frome HK")

        if "救命" in text or "幫" in text:

            keyword = True
    
    elif lang_code == "en-us":

        print("You are not from HK")

        text = text.lower()

        if "help" in text:

            keyword = True

    elif lang_code == "cmn-hant-tw":

        print("You are a fucking Chinese?")

        if "救" in text or "幫" in text:

            keyword = True

    else:
        
        print("Your Code is suck")

        keyword = False

    print(keyword)

    # return the result of the trigger
    return keyword

# define function to hold the whole processing

def keyword_alert():

    counter = False
    flag = False

    Record_Audio()

    text, lang_code, counter = speech_to_text.wavToText(audio_wav)

    if counter == True:

        flag = keyword_trigger(text, lang_code)    

    if flag == True:

        play_wave()
    
    if flag == False:

        print("No Alert needed, Record Again...")

    keyword_alert()

# Main
    
if __name__ == "__main__":

    print("The Alert Start Now")

    keyword_alert()

    print("Error: Someone ended the program...")