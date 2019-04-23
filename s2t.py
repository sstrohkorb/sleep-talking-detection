import sys

from speech_recognition import SpeechRecognizer


if __name__ == '__main__':
    try:
        recog = SpeechRecognizer()
    except SpeechRecognizer.InvalidDevice as e:
        print(e.message)
        sys.exit(1)
