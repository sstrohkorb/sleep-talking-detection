import io
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

# https://googleapis.github.io/google-cloud-python/latest/speech/index.html#

client = speech.SpeechClient()
path = "/home/pi/Documents/sleep-talking-detection/speech_files/10000/utterance_0.wav"

with io.open(path, 'rb') as audio_file:
    content = audio_file.read()

audio = speech.types.RecognitionAudio(content=content)

config = speech.types.RecognitionConfig(
    encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
    language_code='en-US',
    sample_rate_hertz=48000)

response = client.recognize(config=config, audio=audio)

for result in response.results:
    for alternative in result.alternatives:
        print('=' * 20)
        print('transcript: ' + alternative.transcript)
        print('confidence: ' + str(alternative.confidence))
