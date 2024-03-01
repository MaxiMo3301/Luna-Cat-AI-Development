# import python module

import pyaudio
import wave

# define recording function

def Record_Audio():

    # defince constant variable

    CHUNK = 1024
    RATE = 44100
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    SECONDS = 6
    
    filename = "temp.wav"

    p = pyaudio.PyAudio() # create an interface to PortAudio

    print("Recording")

    # create the stream for the audio input

    stream = p.open(
        format = FORMAT,
        channels = CHANNELS,
        rate = RATE,
        input = True,
        frames_per_buffer = CHUNK
    )

    frames = []

    # Store data in chunks for 5 seconds....

    for i in range(0, int(RATE / CHUNK * SECONDS)):

        data = stream.read(CHUNK)
        frames.append(data)

    # Stop and close the stream
    
    stream.stop_stream()
    stream.close()

    # Terminate the PortAudio insterface
    
    p.terminate()
    
    print("Finished Recording...")

    # Save the recorded data as a WAV file

    wf=wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    print("Audio file saved as temp.wav")

    return True

# Testing Main Function

if __name__ == "__main__":

    flag = Record_Audio()

    print(flag)