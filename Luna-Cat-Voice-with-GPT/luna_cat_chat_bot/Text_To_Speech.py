import google.cloud.texttospeech as tts
from google.oauth2 import service_account
import io

txt_test = "邁克爾·約瑟夫·傑克遜是美國流行音樂歌手、作曲家、唱片製作人、舞蹈家、演員、慈善家。被尊稱為「流行樂之王」。麥可傑克森在音樂、舞蹈、時尚方面的巨大貢獻，加上備受關注的個人生活，使他成為全球流行文化的代表人物。"
en_txt = "France won the 2018 Fifa World Cup"
sample = "Luna is listening... "
answer = "Luna is here, please talk your request to me after the cat meow sound."

def text_to_wav(txt, lang_code):

    client_file = 'audio-bot-389910-9ef67fc68a47.json'
    credit = service_account.Credentials.from_service_account_file(client_file)

    text_input = tts.SynthesisInput(text=txt)

    Voice_params_en = tts.VoiceSelectionParams(
        language_code = 'en-US',
        name = 'en-US-Neural2-H'
    )

    Voice_params_cn = tts.VoiceSelectionParams(
        language_code = 'yue-Hant-HK',
        name = 'yue-HK-Standard-A'
    )

    if lang_code == "en-us":

        Voice_params = Voice_params_en

    if lang_code == "yue-hant-hk":

        Voice_params = Voice_params_cn

    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient(credentials = credit)

    response = client.synthesize_speech(
        input=text_input,
        voice=Voice_params,
        audio_config=audio_config,
    )

    filename = "audio.wav"

    with open(filename, "wb") as out:
        out.write(response.audio_content)

    return True



if __name__ == "__main__":

    txt = answer

    text_to_wav(txt)

    print("done")