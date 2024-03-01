# import python module / library

from google.oauth2 import service_account
from google.cloud import speech
import io

# define the text to speech function

def wavToText(audio_wav):

    text = None
    lang_code = None

    # get authority from googe cloud server

    client_file = 'audio-bot-389910-9ef67fc68a47.json'
    credit = service_account.Credentials.from_service_account_file(client_file)
    client = speech.SpeechClient(credentials = credit)

    # Language Code

    first_lang = "yue-Hant-HK"
    second_lang = "en-US"
    third_lang = "zh-TW"
    # open the audio file

    with open(audio_wav, "rb") as audio_file:
        content = audio_file.read()

    # config of speech to text functions

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code=first_lang,
        alternative_language_codes=[second_lang, third_lang],
    )

    print("Waiting for operation to complete...")

    response = client.recognize(config=config, audio=audio)

    if response.results != "":

        for result in response.results:

            text = result.alternatives[0].transcript

        for result in response.results:

            lang_code = result.language_code

        return text, lang_code, True
    
    else:

        return text, lang_code, False