import pyaudio
import wave
import audioop

def Record_Audio():

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    SILENCE_THRESHOLD = 1500  # adjust this as needed
    SILENCE_DURATION = 3  # adjust this as needed

    p = pyaudio.PyAudio()

    stream = p.open(
        format = FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK

    )

    frames = []

    print("Recording started...")

    silence_counter = 0
    max_silence_duration = int(RATE / CHUNK * SILENCE_DURATION)

    #print(max_silence_duration)

    while True:

        data = stream.read(CHUNK)
        frames.append(data)
        rms = audioop.rms(data, 2)
        if rms < SILENCE_THRESHOLD:
            silence_counter += 1
            #print(silence_counter)
        else:
            silence_counter = 0
        
        if silence_counter >= max_silence_duration:
            break
    
    print("Recroding stopped...")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open("temp.wav", "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)   
    wf.writeframes(b"".join(frames))
    wf.close()

    print("Audio file saved as temp.wav")

    return True


if __name__ == "__main__":

    Record_Audio()