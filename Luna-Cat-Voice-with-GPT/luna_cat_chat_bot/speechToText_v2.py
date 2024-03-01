# import module

import io
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech

def wavToText(path):

    client_file = 'audio-bot-389910-9ef67fc68a47.json'
    credit = service_account.Credentials.from_service_account_file(client_file)
    client = speech.SpeechClient(credentials = credit)

    first_lang = "en-US"
    second_lang = "yue-Hant-HK"

    with open(path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code=first_lang,
        alternative_language_codes=[second_lang],
    )

    print("Waiting for operation to complete...")
    response = client.recognize(config=config, audio=audio)

    for result in response.results:

        text = result.alternatives[0].transcript

    for result in response.results:

        lang_code = result.language_code

    return text, lang_code , True

if __name__ == "__main__":

    audio = 'answer.wav'

    text, lang_code, counter = wavToText(audio)

    print(text)
    
    print(lang_code)

    print(counter)