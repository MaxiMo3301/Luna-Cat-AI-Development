# import module
from google.oauth2 import service_account
from google.cloud import speech
import io

# Set up API Envirnoment for Google Cloud

def wav_to_text(audio, lang_code):

    client_file = 'audio-bot-389910-9ef67fc68a47.json'
    credit = service_account.Credentials.from_service_account_file(client_file)
    client = speech.SpeechClient(credentials = credit)

    audio_file = audio

    with io.open(audio_file, 'rb') as f:

        content = f.read()

        audio = speech.RecognitionAudio(content = content)

    # Recognize the audio wave and trans to text
    config_en = speech.RecognitionConfig(
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code = 'en-US' )
    
    config_cn = speech.RecognitionConfig(
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code = 'yue-Hant-HK' )
    
    if lang_code == "en-US":

        config = config_en


    if lang_code == "yue-Hant-HK":

        config = config_cn
    
    response = client.recognize(config = config, audio = audio)

    for result in response.results:

        text = result.alternatives[0].transcript

    return text, True

if __name__ == "__main__":

    audio = '/home/asa/Desktop/Luna_Chat/temp.wav'

    text, counter =  wav_to_text(audio)

    print(text)
    print(counter)