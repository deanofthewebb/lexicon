import numpy as np
import os
import unicodedata
import pickle
from sphfile import SPHFile



class Speech(object):
    def __init__(self, speaker_id, speech_id, source_file, ground_truth, start = None, stop = None, audio_type='LINEAR16', sample_rate=16000, print_report=False):
        cache_file = os.path.join(os.getcwd(), 'datacache', 'speech_objects',
                                       '{}_preprocess.p'.format(speech_id.strip()))
        
        self._speaker_id = speaker_id
        self._speech_id = speech_id
        self._source_file = source_file
        self._ground_truth_transcript = ground_truth
        self._candidate_transcripts = []
        
        # Collect Time offsets of each candidate word of the best candidate transcript
        self._candidate_timestamps = []
        self._start_time  = float(start)
        self._stop_time   = float(stop)
        self._audio_file = self.cache_sph2wav()
        
        # LINEAR16 -> .sph and .wav | FLAC -> .flac
        self._audio_type = audio_type
        
        self._sample_rate = sample_rate
        
        
        
        
    @property
    def speaker_id(self):
        return self._speaker_id
    @property
    def speech_id(self):
        return self._speech_id
    @property
    def source_file(self):
        return self._source_file
    @property
    def audio_file(self):
        return self._audio_file
    @property
    def candidate_transcripts(self):
        return self._candidate_transcripts
    @property
    def candidate_timestamps(self):
        return self._candidate_timestamps
    @property
    def audio_type(self):
        return self._audio_type
    @property
    def sample_rate(self):
        return self._sample_rate
    @property
    def start_time(self):
        return self._start_time
    @property
    def stop_time(self):
        return self._stop_time
    @property
    def ground_truth_transcript(self):
        return self._ground_truth_transcript
    
    
    def print_loading_report(self):
        print()
        print('Lexicon Speech: "{}" successfully loaded to memory location:'.format(self._speech_id), self)
        print('Speaker Id:', self._speaker_id)
        print('Source Filepath:', self._source_file)
        print('Audio Filepath:', self._audio_file)
        print('Ground Truth Transcript: {}'.format(self._ground_truth_transcript))
        print('Timestamps: {0}-{1}'.format(self._start_time, self._stop_time))
        print()
        print()
    
    def cache_sph2wav(self):
        """
        Converts an audio file in SPH format to WAV format, for sending to Google Cloud Speech API)
        """
        wav_cache_dir = os.path.join(os.getcwd(), 'datacache', 'speech_objects','wav/')
        if not os.path.exists(wav_cache_dir):
            os.makedirs(wav_cache_dir, exist_ok=True)
        
        cache_file = os.path.join(wav_cache_dir, '{}.wav'.format(self._speech_id.strip()))
        if not os.path.exists(cache_file):
            sph =SPHFile(self._source_file)

            # write out a wav file with content from {start_time} to {stop_time} seconds
            sph.write_wav(cache_file, self._start_time, self._start_time)
        
        return cache_file

        
        