import numpy as np
import os
import unicodedata
import pickle

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types


class GCSWrapper(object):
    def __init__(self, encoding='LINEAR16', sample_rate=16000, 
                 service_account = 'lexicon-bot@exemplary-oath-179301.iam.gserviceaccount.com',
                key_file='Lexicon-e94eff39fad7.json',
                language_code='en-US',
                max_alternatives=10,
                include_time_offsets=True):
        
        self._encoding = encoding
        self._sample_rate = sample_rate
        self._service_account = service_account
        self._key_file = os.path.join(os.getcwd(), key_file)
        self._language_code = language_code
        self._max_alternatives = max_alternatives
        self._include_time_offsets = include_time_offsets
        
        # Install GCS
        os.system("CLOUDSDK_CORE_DISABLE_PROMPTS=1 ./google-cloud-sdk/install.sh")
        # Add Authentication to Environment Vars
        os.environ['GOOGLE_APPLICATION_CREDENTIALS']=self._key_file
        self._client = speech.SpeechClient()

        
    def get_audio_size(audio_filepath):
        statinfo = os.stat(audio_filepath)
        print(statinfo.st_size)
        return statinfo.st_size
    
    
    def transcribe_speech(self, audio_filepath):
        with open(audio_filepath, 'rb') as audio_file:
            content = audio_file.read()
            audio = types.RecognitionAudio(content=content)

            config = types.RecognitionConfig(
                encoding=self._encoding,
                sample_rate_hertz = self._sample_rate,
                language_code = self._language_code,
                max_alternatives=self._max_alternatives,
                profanity_filter=False,
                enable_word_time_offsets=self._include_time_offsets)

            # Detects speech and words in the audio file
            operation = self._client.long_running_recognize(config, audio)

            result = operation.result(timeout=90)
            return result


