import numpy as np
import os
import unicodedata
import pickle
from sphfile import SPHFile


class Speech(object):
    def load_preprocess(cache_file):
        """
        Load the Preprocessed Training data and return them
        """
        if os.path.exists(cache_file):
            return pickle.load(open(cache_file, mode='rb'))
        else:
            print('Nothing saved in the preprocess directory')
            return None
        
    def __init__(self, speaker_id, speech_id, source_file, ground_truth, start = None, stop = None, audio_type='LINEAR16',
                 sample_rate=16000, print_report=False):
        cache_file = os.path.join(os.getcwd(), 'datacache', 'speech_objects',
                                       '{}_preprocess.p'.format(speech_id.strip()))

        if os.path.exists(cache_file):
            (self._speech_id,
             self._speaker_id,
             self._source_file,
             self._audio_file,
             self._candidate_transcripts,
             self._candidate_timestamps,
             self._audio_type,
             self._sample_rate, 
             self._start_time, 
             self._stop_time, 
             self._ground_truth_transcript) = Speech.load_preprocess(cache_file)
        else:
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
            
            # Cache Object
            self.preprocess_and_save()
        
        
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
        print('Candidate Transcripts: {}'.format(self._candidate_transcripts))
        print('Candidate Timestamps: {}'.format(self._candidate_timestamps))
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
            sph.write_wav(cache_file, self._start_time, self._stop_time)
        
        return cache_file
    
    
    def populate_gcs_results(self, api_result):
        alternatives = api_result.results[0].alternatives
        for alternative in alternatives:
            self._candidate_transcripts.append({"transcript": alternative.transcript, "confidence": alternative.confidence})
           
            for word_info in alternative.words:
                word = word_info.word
                start_time = word_info.start_time
                end_time = word_info.end_time
                start = start_time.seconds + start_time.nanos * 1e-9
                end = end_time.seconds + end_time.nanos * 1e-9
                delta = end - start
                self._candidate_timestamps.append({"word": word, "start_time": start, "end_time": end, "total_time": delta})
     
    
    def preprocess_and_save(self):
        """
        Preprocess Speech Data
        """
        cache_directory = os.path.join(os.getcwd(), 'datacache', 'speech_objects')
        if not os.path.exists(cache_directory):
            os.mkdir(cache_directory)
            
        pickle.dump((self._speech_id,
                     self._speaker_id,
                     self._source_file,
                     self._audio_file,
                     self._candidate_transcripts,
                     self._candidate_timestamps,
                     self._audio_type,
                     self._sample_rate, 
                     self._start_time, 
                     self._stop_time, 
                     self._ground_truth_transcript), open(os.path.join(cache_directory,
                                                                       '{}_preprocess.p'.format(self._speech_id)), 'wb'))



