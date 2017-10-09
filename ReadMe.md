
# Lexicon - Orchestrator


## Overview

For this project, I will build a simple custom ochestrator that processes data objects from the "Lexicon" class.
    - These objects are custom datasets that are modeled after the Ted Talk speakers. 
    - Each Lexicon has a corpus and some helper methods aimed at training and prediction
    - Lexicon class will also have a preprocessing and caching function.
    - Each object will have two methods of prediction, n-gram language model and a recurrent neural network model
    - Each object has a custom reporting function that reports the results of training
    - Each object will be able to learn from any text data provided, and return a transcript with confidence values from input posed in speech utterances. 
        - I will use Google's cloud-based services to preprocess the input audio data and transcribe into an initial guess. Then I will train a model to improve on Google cloud speech API's response.



```python
## Use to reload modules
from importlib import reload
%reload_ext autoreload
%autoreload 2
```


```python
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile

librispeech_dataset_folder_path = 'LibriSpeech'
tar_gz_path = 'dev-clean.tar.gz'

books_path = 'original-books.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(books_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Librispeech Book Texts') as pbar:
        urlretrieve(
            'http://www.openslr.org/resources/12/original-books.tar.gz',
            books_path,
            pbar.hook)

if not isdir(librispeech_dataset_folder_path+'/books'):
    with tarfile.open(books_path) as tar:
        tar.extractall()
        tar.close()
        
        
        
if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Librispeech dev-clean.tar.gz') as pbar:
        urlretrieve(
            'http://www.openslr.org/resources/12/dev-clean.tar.gz',
            tar_gz_path,
            pbar.hook)

if not isdir(librispeech_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        tar.extractall()
        tar.close()
        
        
        
```


```python
import io

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

# Instantiates a client
client = speech.SpeechClient()

# The name of the dev-test audio file to transcribe
dev_file_name_0 = os.path.join(
    os.getcwd(),
    'LibriSpeech',
    'dev-clean',
    '84',
    '121123',
    '84-121123-0000.flac')
gt0 = 'GO DO YOU HEAR'

dev_file_name_1 = os.path.join(
    os.getcwd(),
    'LibriSpeech',
    'dev-clean',
    '84',
    '121123',
    '84-121123-0001.flac')
gt1 = 'BUT IN LESS THAN FIVE MINUTES THE STAIRCASE GROANED BENEATH AN EXTRAORDINARY WEIGHT'

# The name of the test audio file to transcribe
dev_file_name_2 = os.path.join(
    os.getcwd(),
    'LibriSpeech',
    'dev-clean',
    '84',
    '121123',
    '84-121123-0002.flac')
gt2 = 'AT THIS MOMENT THE WHOLE SOUL OF THE OLD MAN SEEMED CENTRED IN HIS EYES WHICH BECAME BLOODSHOT THE VEINS OF THE THROAT SWELLED HIS CHEEKS AND TEMPLES BECAME PURPLE AS THOUGH HE WAS STRUCK WITH EPILEPSY NOTHING WAS WANTING TO COMPLETE THIS BUT THE UTTERANCE OF A CRY'

dev_file_name_3 = os.path.join(
    os.getcwd(),
    'LibriSpeech',
    'dev-clean',
    '84',
    '121123',
    '84-121123-0003.flac')
gt3 = 'AND THE CRY ISSUED FROM HIS PORES IF WE MAY THUS SPEAK A CRY FRIGHTFUL IN ITS SILENCE'

dev_file_name_4 = os.path.join(
    os.getcwd(),
    'LibriSpeech',
    'dev-clean',
    '84',
    '121123',
    '84-121123-0004.flac')
gt4 = "D'AVRIGNY RUSHED TOWARDS THE OLD MAN AND MADE HIM INHALE A POWERFUL RESTORATIVE"


test_file_name_1 = os.path.join(
    os.getcwd(),
    'RNN-Tutorial-master',
    'data',
    'raw',
    'librivox',
    'LibriSpeech',
    'test-clean-wav',
    '4507-16021-0019.wav')


audio_files = {dev_file_name_0:gt0, dev_file_name_1:gt1, dev_file_name_2:gt2, dev_file_name_3:gt3, dev_file_name_4:gt4}

```


    ---------------------------------------------------------------------------

    DefaultCredentialsError                   Traceback (most recent call last)

    <ipython-input-3-f969dd3ea995> in <module>()
          7 
          8 # Instantiates a client
    ----> 9 client = speech.SpeechClient()
         10 
         11 # The name of the dev-test audio file to transcribe


    ~/miniconda3/envs/tf-gpu/lib/python3.5/site-packages/google/cloud/gapic/speech/v1/speech_client.py in __init__(self, service_path, port, channel, credentials, ssl_credentials, scopes, client_config, app_name, app_version, lib_name, lib_version, metrics_headers)
        144             credentials=credentials,
        145             scopes=scopes,
    --> 146             ssl_credentials=ssl_credentials)
        147 
        148         self.operations_client = operations_client.OperationsClient(


    ~/miniconda3/envs/tf-gpu/lib/python3.5/site-packages/google/gax/grpc.py in create_stub(generated_create_stub, channel, service_path, service_port, credentials, scopes, ssl_credentials)
        104 
        105         if credentials is None:
    --> 106             credentials = _grpc_google_auth.get_default_credentials(scopes)
        107 
        108         channel = _grpc_google_auth.secure_authorized_channel(


    ~/miniconda3/envs/tf-gpu/lib/python3.5/site-packages/google/gax/_grpc_google_auth.py in get_default_credentials(scopes)
         60 def get_default_credentials(scopes):
         61     """Gets the Application Default Credentials."""
    ---> 62     credentials, _ = google.auth.default(scopes=scopes)
         63     return credentials
         64 


    ~/miniconda3/envs/tf-gpu/lib/python3.5/site-packages/google/auth/_default.py in default(scopes, request)
        284             return credentials, explicit_project_id or project_id
        285 
    --> 286     raise exceptions.DefaultCredentialsError(_HELP_MESSAGE)
    

    DefaultCredentialsError: Could not automatically determine credentials. Please set GOOGLE_APPLICATION_CREDENTIALS or
    explicitly create credential and re-run the application. For more
    information, please see
    https://developers.google.com/accounts/docs/application-default-credentials.



```python
# Prepare a plain text corpus from which we train a languague model
import glob
import os
import utils
import nltk

# Gather all text files from directory
LIBRISPEECH_DIRECTORY = os.path.join(os.getcwd(),'LibriSpeech/')
TEDLIUM_DIRECTORY = os.path.join(os.getcwd(),'TEDLIUM_release1/')

# TRAINING_DIRECTORY = os.path.abspath(os.path.join(os.sep,'Volumes',"My\ Passport\ for\ Mac",'lexicon','LibriSpeech'))
dev_librispeech_path = "{}{}{}{}".format(LIBRISPEECH_DIRECTORY, 'dev-clean/', '**/', '*.txt*')
train_librispeech_path = "{}{}{}{}{}".format(LIBRISPEECH_DIRECTORY, 'books/', 'utf-8/', '**/', '*.txt*')
TED_path = "{}{}{}{}".format(TEDLIUM_DIRECTORY,'train/','**/', '*.stm')

text_paths = sorted(glob.glob(train_librispeech_path, recursive=True))
segmented_text_paths = sorted(glob.glob(dev_librispeech_path, recursive=True))
stm_paths = sorted(glob.glob(TED_path, recursive=True))

print('Found:',len(text_paths),"text files in the directories {0}\n{1} segmented text files in the {2} directory and \n{3} stm files in directory: {4}:".format(train_librispeech_path, 
        len(segmented_text_paths), dev_librispeech_path, len(stm_paths),TED_path ))
```

    Found: 41 text files in the directories /src/lexicon/LibriSpeech/books/utf-8/**/*.txt*
    97 segmented text files in the /src/lexicon/LibriSpeech/dev-clean/**/*.txt* directory and 
    774 stm files in directory: /src/lexicon/TEDLIUM_release1/train/**/*.stm:


### Build Text Corpuses for Training


```python
import tensorflow as tf
import re
import codecs
import string
from lexicon import Lexicon
from speech import Speech
      
librispeech_corpus = u""
stm_segments = []
lexicons = {} # {speaker_id: lexicon_object}
speeches = {} # {speech_id: speech_object}
segmented_librispeeches = {}

for book_filename in text_paths[:10]: # 1 Book
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        lines = book_file.read()
        librispeech_corpus += lines
for stm_filename in stm_paths: # Process STM files (Tedlium)
        stm_segments.append(utils.parse_stm_file(stm_filename))
        

# Train on 3 speakers
for segments in stm_segments[15:17]: 
    for segment in segments:
        segment_key = "{0}_{1}_{2}".format(segment.speaker_id.strip(), str(segment.start_time).replace('.','_'),
                                          str(segment.stop_time).replace('.','_'))
        if segment.speaker_id not in speeches.keys():
            source_file = os.path.join(os.getcwd(), 'TEDLIUM_release1',
                                       'train','sph', '{}.sph'.format(segment.filename))
            speech = Speech(speaker_id=segment.speaker_id,
                                           speech_id = segment_key,
                                           source_file=source_file,
                                           ground_truth = ' '.join(segment.transcript.split()[:-1]),
                                           start = segment.start_time,
                                           stop = segment.stop_time,
                                           audio_type = 'LINEAR16')
        else:
            speech = speeches[segment.speaker_id.strip()]
            print('Already found speech in list at location: ', speech)
        
        speeches[segment_key] = speech

        if segment.speaker_id not in lexicons.keys():
            lexicon = Lexicon(base_corpus=librispeech_corpus, name=segment.speaker_id)
            lexicons[segment.speaker_id.strip()] = lexicon
        else:
            lexicon = lexicons[segment.speaker_id.strip()]
        
        # Add Speech to Lexicon
        if speech not in lexicon.speeches:
            lexicon.add_speech(speech)

```

### Load GCS Transcripts using GCS Wrapper


```python
import numpy as np
view_sentence_range = (0, 10)

for speaker_id, lexicon in lexicons.items():
    print('Dataset Stats')
    print('Roughly the number of unique words: {}'.format(lexicon.vocab_size))
    
    word_counts = [len(sentence.split()) for sentence in lexicon.corpus_sentences]
    print('Number of sentences: {}'.format(len(lexicon.corpus_sentences)))
    print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

    print()
    print('Transcript sentences {} to {}:'.format(*view_sentence_range))
    print('\n'.join(lexicon.training_set[0][view_sentence_range[0]:view_sentence_range[1]]))
    print()
    print('Ground Truth sentences {} to {}:'.format(*view_sentence_range))
    print('\n'.join(lexicon.training_set[1][view_sentence_range[0]:view_sentence_range[1]]))
    print()
```

    Dataset Stats
    Roughly the number of unique words: 58051
    Number of sentences: 27600
    Average number of words in a sentence: 24.12873188405797
    
    Transcript sentences 0 to 10:
    
    
    "I shall never be better," said Jane Merrick, sternly
     "The end is not
    far off now
    "
    
    "Oh, I'm sorry to hear you say that!" said Patsy; "but I hope it is
    not true
     Why, here are we four newly found relations all beginning to
    get acquainted, and to love one another, and we can't have our little
    party broken up, auntie dear
    "
    
    "Five of us--five relations," cried Uncle John, coming around the
    corner of the hedge
     "Don't I count, Patsy, you rogue? Why you're
    looking as bright and as bonny as can be
     I wouldn't be surprised if
    you could toddle
    "
    
    "Not yet," she answered, cheerfully
     "But I'm doing finely, Uncle
    John, and it won't be long before I can get about as well as ever
    "
    
    "And to think," said Aunt Jane, bitterly, "that all this trouble was
    caused by that miserable boy! If I knew where to send him he'd not
    stay at Elmhurst a day longer
    
    Ground Truth sentences 0 to 10:
    
    
    "I shall never be better," said Jane Merrick, sternly
     "The end is not
    far off now
    "
    
    "Oh, I'm sorry to hear you say that!" said Patsy; "but I hope it is
    not true
     Why, here are we four newly found relations all beginning to
    get acquainted, and to love one another, and we can't have our little
    party broken up, auntie dear
    "
    
    "Five of us--five relations," cried Uncle John, coming around the
    corner of the hedge
     "Don't I count, Patsy, you rogue? Why you're
    looking as bright and as bonny as can be
     I wouldn't be surprised if
    you could toddle
    "
    
    "Not yet," she answered, cheerfully
     "But I'm doing finely, Uncle
    John, and it won't be long before I can get about as well as ever
    "
    
    "And to think," said Aunt Jane, bitterly, "that all this trouble was
    caused by that miserable boy! If I knew where to send him he'd not
    stay at Elmhurst a day longer
    
    Dataset Stats
    Roughly the number of unique words: 57965
    Number of sentences: 27540
    Average number of words in a sentence: 24.005228758169935
    
    Transcript sentences 0 to 10:
    
    
    "I shall never be better," said Jane Merrick, sternly
     "The end is not
    far off now
    "
    
    "Oh, I'm sorry to hear you say that!" said Patsy; "but I hope it is
    not true
     Why, here are we four newly found relations all beginning to
    get acquainted, and to love one another, and we can't have our little
    party broken up, auntie dear
    "
    
    "Five of us--five relations," cried Uncle John, coming around the
    corner of the hedge
     "Don't I count, Patsy, you rogue? Why you're
    looking as bright and as bonny as can be
     I wouldn't be surprised if
    you could toddle
    "
    
    "Not yet," she answered, cheerfully
     "But I'm doing finely, Uncle
    John, and it won't be long before I can get about as well as ever
    "
    
    "And to think," said Aunt Jane, bitterly, "that all this trouble was
    caused by that miserable boy! If I knew where to send him he'd not
    stay at Elmhurst a day longer
    
    Ground Truth sentences 0 to 10:
    
    
    "I shall never be better," said Jane Merrick, sternly
     "The end is not
    far off now
    "
    
    "Oh, I'm sorry to hear you say that!" said Patsy; "but I hope it is
    not true
     Why, here are we four newly found relations all beginning to
    get acquainted, and to love one another, and we can't have our little
    party broken up, auntie dear
    "
    
    "Five of us--five relations," cried Uncle John, coming around the
    corner of the hedge
     "Don't I count, Patsy, you rogue? Why you're
    looking as bright and as bonny as can be
     I wouldn't be surprised if
    you could toddle
    "
    
    "Not yet," she answered, cheerfully
     "But I'm doing finely, Uncle
    John, and it won't be long before I can get about as well as ever
    "
    
    "And to think," said Aunt Jane, bitterly, "that all this trouble was
    caused by that miserable boy! If I knew where to send him he'd not
    stay at Elmhurst a day longer
    


### Preprocess Dataset - Tokenize Corpus


```python
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
import re
import codecs
import string

# reading the file in unicode format using codecs library    
stoplist = set(stopwords.words('english'))
# Strip punctuation
translate_table = dict((ord(char), None) for char in string.punctuation) 
        
corpus_raw = u""
for book_filename in text_paths:
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        lines = book_file.read()
        corpus_raw += lines.translate(translate_table) # remove punctuations 

               
# Tokenize
tokenized_words = nltk.tokenize.word_tokenize(corpus_raw)

## Clean the tokens ##
# Remove stop words
tokenized_words = [word for word in tokenized_words if word not in stoplist]

# Remove single-character tokens (mostly punctuation)
tokenized_words = [word for word in tokenized_words if len(word) > 1]

# Remove numbers
tokenized_words = [word for word in tokenized_words if not word.isnumeric()]

# Lowercase all words (default_stopwords are lowercase too)
tokenized_words = [word.lower() for word in tokenized_words]
```

### Preprocess Dataset - Extract N-Gram Model


```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.collocations import *
from nltk.probability import FreqDist
import nltk

# extracting the bi-grams and sorting them according to their frequencies
finder = BigramCollocationFinder.from_words(tokenized_words)
# finder.apply_freq_filter(3)

bigram_model = nltk.bigrams(tokenized_words)
bigram_model = sorted(bigram_model, key=lambda item: item[1], reverse=True)  
# print(bigram_model)
print('')
print('')
print('')
np.save("lang_model.npy",bigram_model)
```


```python
fdist = nltk.FreqDist(bigram_model)

# Output top 50 words
print("Word|Freq:")
for word, frequency in fdist.most_common(50):
    print(u'{}|{}'.format(word, frequency))
```


```python
cfreq_2gram = nltk.ConditionalFreqDist(bigram_model)
# print('Conditional Frequency Conditions:\n', cfreq_2gram)
print()

# First access the FreqDist associated with "one", then the keys in that FreqDist
print("Listing the words that can follow after 'greater':\n", cfreq_2gram["greater"].keys())
print()

# Determine Most common in conditional frequency
print("Listing 20 most frequent words to come after 'greater':\n", cfreq_2gram["greater"].most_common(20))
```


```python
# For each word in the evaluation list:
# Select word and determine its frequency distribution
# Grab probability of second word in the list
# Continue this process until the sentence is scored

# Add small epsilon value to avoid division by zero
epsilon = 0.0000001

# Loads the audio into memory
for audio, ground_truth in audio_files.items():
    with io.open(audio, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code='en-US',
        max_alternatives=10,
        profanity_filter=False,
        enable_word_time_offsets=True)

    # Detects speech and words in the audio file
    operation = client.long_running_recognize(config, audio)

    print('Waiting for operation to complete...')
    result = operation.result(timeout=90)

    alternatives = result.results[0].alternatives


    #print("API Results: ", alternatives)
    print()
    print()

    rerank_results = {}
    for alternative in alternatives:
        sent = alternative.transcript

        words = nltk.tokenize.word_tokenize(sent)
        probs = np.ones_like(words, dtype=np.float32)*epsilon
        # print(words,'\n',probs)
        for word in words:
            if words.index(word) < len(words)-1: 
                freq = cfreq_2gram[word].freq(words[words.index(word)+1])
                probs[words.index(word)] = freq
            # print(probs)

        lexicon_score = np.sum(probs)
        # print(word_score)

        # Re-rank alternatives using a weighted average of the two scores
        api_weight = 0.90
        confidence_score = alternative.confidence*api_weight + lexicon_score*(1-api_weight)
        rerank_results[alternative.transcript] = confidence_score

    print("RE-RANKED Results: \n", rerank_results)
    print()
    print()

    import operator
    index, value = max(enumerate(list(rerank_results.values())), key=operator.itemgetter(1))
    # Select Corresponding Transcript:
    script=''
    for trnscript, confidence in rerank_results.items():
        if confidence == value:
            script = trnscript

    # Evaluate the differences between the Original and the Reranked transcript:
    print("ORIGINAL Transcript: \n'{0}' \nwith a confidence_score of: {1}".format(alternative.transcript, alternative.confidence))
    
    
    print()
    print()
    print("RE-RANKED Transcript: \n'{0}' \nwith a confidence_score of: {1}".format(script, value))
    
    print()
    print()
    print("GROUND TRUTH TRANSCRIPT: \n{0}".format(ground_truth))
    print()
    ranked_differences = list(set(nltk.tokenize.word_tokenize(alternative.transcript.lower())) -
                              set(nltk.tokenize.word_tokenize(script.lower())))
    if len(ranked_differences) == 0:  
        print("No reranking was performed. The transcripts match!")
    else:
        print("The original transcript was RE-RANKED. The transcripts do not match!")
        print("Differences between original and re-ranked: ", ranked_differences)
    print()
    print()
    
    # Evaluate Differences between the Original and Ground Truth:
    gt_orig_diff = list(set(nltk.tokenize.word_tokenize(alternative.transcript.lower())) -
                              set(nltk.tokenize.word_tokenize(ground_truth.lower())))
    if len(gt_orig_diff) == 0:  
        print("The ORIGINAL transcript matches ground truth!")
    else:
        print("The original transcript DOES NOT MATCH ground truth.")
        print("Differences between original and ground truth: ", gt_orig_diff)
    print()
    print()
    
    
    gt_rr_diff = list(set(nltk.tokenize.word_tokenize(script.lower())) -
                              set(nltk.tokenize.word_tokenize(ground_truth.lower())))
    if len(gt_rr_diff) == 0:  
        print("The RE-RANKED transcript matches ground truth!")
    else:
        print("The RE_RANKED transcript DOES NOT MATCH ground truth.")
        print("Differences between Reranked and ground truth: ", gt_rr_diff)
    print()
    print()
    
    print()
    print()
    
    
    # Compute the Levenshtein Distance (a.k.a. Edit Distance)
#     import nltk.metrics.distance as lev_dist
    
    # Google API Edit Distance
    goog_edit_distance = nltk.edit_distance(alternative.transcript.lower(), ground_truth.lower())
    
    # Re-Ranked Edit Distance
    rr_edit_distance = nltk.edit_distance(script.lower(), ground_truth.lower())

    
    print("ORIGINAL Edit Distance: \n{0}".format(goog_edit_distance))
    print("RE-RANKED Edit Distance: \n{0}".format(rr_edit_distance))
    print()
    print()
    
```

### Evaluate N-Gram Model on Dataset


```python
# Gather all samples, load into dictionary
# Prepare a plain text corpus from which we train a languague model
import glob
import operator

# Gather all text files from directory
WORKING_DIRECTORY = os.path.join(os.getcwd(),'LibriSpeech/')

dev_path = "{}{}{}{}".format(WORKING_DIRECTORY, 'dev-clean/', '**/', '*.txt')
train_path = "{}{}{}{}{}".format(WORKING_DIRECTORY, 'books/', 'utf-8/', '**/', '*.txt*')

text_paths = sorted(glob.glob(dev_path, recursive=True))
print('Found',len(text_paths),'text files in the directory:', dev_path)

transcripts = {}
for document in text_paths:
    with codecs.open(document, 'r', 'utf-8') as filep:
        for i,line in enumerate(filep):
            transcripts[line.split()[0]] = ' '.join(line.split()[1:])

## Evaluate all samples found ##
cloud_speech_api_accuracy = []
custom_lang_model_accuracy = []
epsilon = 0.000000001
api_weight = 0.85
steps = 0
# Pull In Audio File
for filename, gt_transcript in transcripts.items():
    steps += 1
    dirs = filename.split('-')
    
    audio_filepath = dev_file_name_0 = os.path.join(
    os.getcwd(),
    'LibriSpeech',
    'dev-clean',
    dirs[0],
    dirs[1],
    "{0}.flac".format(filename))
    
    

    # Load the audio into memory
    with io.open(audio_filepath, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code='en-US',
        max_alternatives=10,
        profanity_filter=False,
        enable_word_time_offsets=True)

    # Detects speech and words in the audio file
    operation = client.long_running_recognize(config, audio)
    result = operation.result(timeout=90)
    alternatives = result.results[0].alternatives


    # Evaluate API Results for Re-Ranking:
    rerank_results = {}
    for alternative in alternatives:
        sent = alternative.transcript
        
        # Strip punctuation
        translate_table = dict((ord(char), None) for char in string.punctuation)        
        sent = sent.translate(translate_table) # remove punctuations

        words = nltk.tokenize.word_tokenize(sent)
        probs = np.ones_like(words, dtype=np.float32)*epsilon

        for word in words:
            if words.index(word) < len(words)-1: 
                freq = cfreq_2gram[word].freq(words[words.index(word)+1])
                probs[words.index(word)] = freq

        lexicon_score = np.sum(probs)

        # Re-rank alternatives using a weighted average of the two scores
        confidence_score = alternative.confidence*api_weight + lexicon_score*(1-api_weight)
        rerank_results[alternative.transcript] = confidence_score


    
    index, value = max(enumerate(list(rerank_results.values())), key=operator.itemgetter(1))
    # Select Corresponding Transcript:
    script=''
    for trnscript, confidence in rerank_results.items():
        if confidence == value:
            script = trnscript
                
    # Compute the Accuracy, based on the Levenshtein Distance (a.k.a. Edit Distance)
    gcs_ed = nltk.edit_distance(alternative.transcript.lower(), gt_transcript.lower())
    gcs_upper_bound = max(len(alternative.transcript),len(gt_transcript))
    gcs_accuracy = (1.0 - gcs_ed/gcs_upper_bound)
    
    clm_ed = nltk.edit_distance(script.lower(), gt_transcript.lower())
    clm_upper_bound = max(len(script),len(gt_transcript))
    clm_accuracy = (1.0 - clm_ed/clm_upper_bound)
    
    cloud_speech_api_accuracy.append(gcs_accuracy)
    custom_lang_model_accuracy.append(clm_accuracy)

    if steps % 100 == 0:
        print("{0} Transcripts Processed.".format(steps))
        print('Average API Accuracy:', np.mean(cloud_speech_api_accuracy))
        print('Average Custom Model Accuracy:', np.mean(custom_lang_model_accuracy))
        print()

```


```python
# Use other TED speeches for building test set
test_speeches = {}
for segments in stm_segments:
    for segment in segments:
        segment_key = "{0}_{1}_{2}".format(segment.speaker_id.strip(), str(segment.start_time).replace('.','_'),
                                          str(segment.stop_time).replace('.','_'))

        speech = None
        # If not already exist
        if segment.speaker_id not in test_speeches.keys():
            # Connect to Cloud API to get Candidate Transcripts
            source_file = os.path.join(os.getcwd(), 'TEDLIUM_release1', 'train','sph', '{}.sph'.format(segment.filename))
            speech = Speech(speaker_id=segment.speaker_id,
                                           speech_id = segment_key,
                                           source_file=source_file,
                                           ground_truth = ' '.join(segment.transcript.split()[:-1]),
                                           start = segment.start_time,
                                           stop = segment.stop_time,
                                           audio_type = 'LINEAR16')
        else:
            speech = test_speeches[segment.speaker_id.strip()]
            print('Already found speech in list at location: ', speech)
        
        
        
        test_speeches[segment_key] = speech
```

### Get Cloud Speech API Results


```python
def get_audio_size(audio_filepath):
    statinfo = os.stat(audio_filepath)
    return statinfo.st_size
```


```python
speaker_id, lexicon = list(lexicons.items())[0]
gcs = GCSWrapper()
cache_directory = os.path.join(os.getcwd(), 'datacache', 'speech_objects')
for speech_id, speech in test_speeches.items():
    # Not already saved in prepocess cache
    cache_file = os.path.join(cache_directory,'{}_preprocess.p'.format(speech.speech_id))
    if not speech.candidate_transcripts: 
        size = get_audio_size(speech.audio_file)
        
        #TODO: Split large audio file into new files, build new speech objects
        if size < 10485760:
            try:
                result = gcs.transcribe_speech(speech.audio_file)
            except:
                result = None
            if result:
                speech.populate_gcs_results(result)
                speech.preprocess_and_save()
                print('Adding speech with candidate_transcripts to lexicon')
                lexicon.add_speech(speech)
```

### Train LSTM Net and Evaluate


```python
speaker_id, lexicon = list(lexicons.items())[0]
lexicon.optimize(early_stop=True)
#lexicon.evaluate_testset()
```

    Epoch   0 Batch  100/2536 - Train Accuracy: 0.5909, Validation Accuracy: 0.6339, Loss: 3.7819
    Epoch   0 Batch  200/2536 - Train Accuracy: 0.7076, Validation Accuracy: 0.6339, Loss: 2.5078
    Epoch   0 Batch  300/2536 - Train Accuracy: 0.7054, Validation Accuracy: 0.6339, Loss: 2.3029
    Epoch   0 Batch  400/2536 - Train Accuracy: 0.5996, Validation Accuracy: 0.6339, Loss: 2.8934
    Epoch   0 Batch  500/2536 - Train Accuracy: 0.5458, Validation Accuracy: 0.6339, Loss: 3.6510
    Epoch   0 Batch  600/2536 - Train Accuracy: 0.5646, Validation Accuracy: 0.6339, Loss: 3.4727
    Epoch   0 Batch  700/2536 - Train Accuracy: 0.5769, Validation Accuracy: 0.6339, Loss: 3.4185
    Epoch   0 Batch  800/2536 - Train Accuracy: 0.5625, Validation Accuracy: 0.6339, Loss: 3.5207
    Epoch   0 Batch  900/2536 - Train Accuracy: 0.4487, Validation Accuracy: 0.6339, Loss: 4.0872
    Epoch   0 Batch 1000/2536 - Train Accuracy: 0.5062, Validation Accuracy: 0.6339, Loss: 3.4609
    Epoch   0 Batch 1100/2536 - Train Accuracy: 0.3638, Validation Accuracy: 0.6339, Loss: 4.5591
    Epoch   0 Batch 1200/2536 - Train Accuracy: 0.4668, Validation Accuracy: 0.6339, Loss: 3.6390
    Epoch   0 Batch 1300/2536 - Train Accuracy: 0.7236, Validation Accuracy: 0.6339, Loss: 2.4886
    Epoch   0 Batch 1400/2536 - Train Accuracy: 0.4813, Validation Accuracy: 0.6339, Loss: 3.5806
    Epoch   0 Batch 1500/2536 - Train Accuracy: 0.6466, Validation Accuracy: 0.6339, Loss: 2.6339
    Epoch   0 Batch 1600/2536 - Train Accuracy: 0.5521, Validation Accuracy: 0.6339, Loss: 3.1899
    Epoch   0 Batch 1700/2536 - Train Accuracy: 0.6278, Validation Accuracy: 0.6339, Loss: 2.5285
    Epoch   0 Batch 1800/2536 - Train Accuracy: 0.7358, Validation Accuracy: 0.6339, Loss: 1.8110
    Epoch   0 Batch 1900/2536 - Train Accuracy: 0.5599, Validation Accuracy: 0.6339, Loss: 2.8119
    Epoch   0 Batch 2000/2536 - Train Accuracy: 0.5889, Validation Accuracy: 0.6339, Loss: 2.9511
    Epoch   0 Batch 2100/2536 - Train Accuracy: 0.4375, Validation Accuracy: 0.6339, Loss: 3.6075
    Epoch   0 Batch 2200/2536 - Train Accuracy: 0.5813, Validation Accuracy: 0.6339, Loss: 2.7566
    Epoch   0 Batch 2300/2536 - Train Accuracy: 0.3750, Validation Accuracy: 0.6339, Loss: 4.4988
    Epoch   0 Batch 2400/2536 - Train Accuracy: 0.5216, Validation Accuracy: 0.6339, Loss: 3.3584
    Epoch   0 Batch 2500/2536 - Train Accuracy: 0.4183, Validation Accuracy: 0.6339, Loss: 4.0160
    Epoch   1 Batch  100/2536 - Train Accuracy: 0.6534, Validation Accuracy: 0.6339, Loss: 2.3992
    Epoch   1 Batch  200/2536 - Train Accuracy: 0.7321, Validation Accuracy: 0.6339, Loss: 1.9085
    Epoch   1 Batch  300/2536 - Train Accuracy: 0.7188, Validation Accuracy: 0.6339, Loss: 1.7597
    Epoch   1 Batch  400/2536 - Train Accuracy: 0.6680, Validation Accuracy: 0.6339, Loss: 2.2870
    Epoch   1 Batch  500/2536 - Train Accuracy: 0.5708, Validation Accuracy: 0.6339, Loss: 2.9395
    Epoch   1 Batch  600/2536 - Train Accuracy: 0.5958, Validation Accuracy: 0.6339, Loss: 2.7216
    Epoch   1 Batch  700/2536 - Train Accuracy: 0.5938, Validation Accuracy: 0.6339, Loss: 2.7720
    Epoch   1 Batch  800/2536 - Train Accuracy: 0.5841, Validation Accuracy: 0.6339, Loss: 2.8416
    Epoch   1 Batch  900/2536 - Train Accuracy: 0.4955, Validation Accuracy: 0.6339, Loss: 3.1415
    Epoch   1 Batch 1000/2536 - Train Accuracy: 0.5417, Validation Accuracy: 0.6339, Loss: 2.7730
    Epoch   1 Batch 1100/2536 - Train Accuracy: 0.3817, Validation Accuracy: 0.6339, Loss: 3.7213
    Epoch   1 Batch 1200/2536 - Train Accuracy: 0.5020, Validation Accuracy: 0.6339, Loss: 2.8784
    Epoch   1 Batch 1300/2536 - Train Accuracy: 0.7308, Validation Accuracy: 0.6339, Loss: 1.8544
    Epoch   1 Batch 1400/2536 - Train Accuracy: 0.5146, Validation Accuracy: 0.6339, Loss: 2.9559
    Epoch   1 Batch 1500/2536 - Train Accuracy: 0.6755, Validation Accuracy: 0.6339, Loss: 2.0428
    Epoch   1 Batch 1600/2536 - Train Accuracy: 0.6042, Validation Accuracy: 0.6339, Loss: 2.5360
    Epoch   1 Batch 1700/2536 - Train Accuracy: 0.6619, Validation Accuracy: 0.6339, Loss: 2.0350
    Epoch   1 Batch 1800/2536 - Train Accuracy: 0.7642, Validation Accuracy: 0.6339, Loss: 1.4795
    Epoch   1 Batch 1900/2536 - Train Accuracy: 0.5990, Validation Accuracy: 0.6339, Loss: 2.2648
    Epoch   1 Batch 2000/2536 - Train Accuracy: 0.6274, Validation Accuracy: 0.6339, Loss: 2.3492
    Epoch   1 Batch 2100/2536 - Train Accuracy: 0.5078, Validation Accuracy: 0.6339, Loss: 2.7127
    Epoch   1 Batch 2200/2536 - Train Accuracy: 0.6354, Validation Accuracy: 0.6339, Loss: 2.2033
    Epoch   1 Batch 2300/2536 - Train Accuracy: 0.4399, Validation Accuracy: 0.6339, Loss: 3.8612
    Epoch   1 Batch 2400/2536 - Train Accuracy: 0.5288, Validation Accuracy: 0.6339, Loss: 2.7622
    Epoch   1 Batch 2500/2536 - Train Accuracy: 0.4567, Validation Accuracy: 0.6339, Loss: 3.3669
    Epoch   2 Batch  100/2536 - Train Accuracy: 0.6733, Validation Accuracy: 0.6339, Loss: 2.0062
    Epoch   2 Batch  200/2536 - Train Accuracy: 0.7411, Validation Accuracy: 0.6339, Loss: 1.6330
    Epoch   2 Batch  300/2536 - Train Accuracy: 0.7388, Validation Accuracy: 0.6339, Loss: 1.4664
    Epoch   2 Batch  400/2536 - Train Accuracy: 0.6973, Validation Accuracy: 0.6339, Loss: 1.9647
    Epoch   2 Batch  500/2536 - Train Accuracy: 0.5771, Validation Accuracy: 0.6339, Loss: 2.4975
    Epoch   2 Batch  600/2536 - Train Accuracy: 0.6021, Validation Accuracy: 0.6339, Loss: 2.3414
    Epoch   2 Batch  700/2536 - Train Accuracy: 0.6058, Validation Accuracy: 0.6339, Loss: 2.4317
    Epoch   2 Batch  800/2536 - Train Accuracy: 0.6082, Validation Accuracy: 0.6339, Loss: 2.4752
    Epoch   2 Batch  900/2536 - Train Accuracy: 0.5112, Validation Accuracy: 0.6339, Loss: 2.6803
    Epoch   2 Batch 1000/2536 - Train Accuracy: 0.5938, Validation Accuracy: 0.6339, Loss: 2.3807
    Epoch   2 Batch 1100/2536 - Train Accuracy: 0.4286, Validation Accuracy: 0.6339, Loss: 3.2201
    Epoch   2 Batch 1200/2536 - Train Accuracy: 0.5078, Validation Accuracy: 0.6339, Loss: 2.4335
    Epoch   2 Batch 1300/2536 - Train Accuracy: 0.7524, Validation Accuracy: 0.6339, Loss: 1.4841
    Epoch   2 Batch 1400/2536 - Train Accuracy: 0.5521, Validation Accuracy: 0.6339, Loss: 2.6113
    Epoch   2 Batch 1500/2536 - Train Accuracy: 0.6971, Validation Accuracy: 0.6339, Loss: 1.7170
    Epoch   2 Batch 1600/2536 - Train Accuracy: 0.6146, Validation Accuracy: 0.6339, Loss: 2.1507
    Epoch   2 Batch 1700/2536 - Train Accuracy: 0.7102, Validation Accuracy: 0.6339, Loss: 1.7170
    Epoch   2 Batch 1800/2536 - Train Accuracy: 0.7812, Validation Accuracy: 0.6339, Loss: 1.2533
    Epoch   2 Batch 1900/2536 - Train Accuracy: 0.6641, Validation Accuracy: 0.6339, Loss: 1.9008
    Epoch   2 Batch 2000/2536 - Train Accuracy: 0.6514, Validation Accuracy: 0.6339, Loss: 2.0053
    Epoch   2 Batch 2100/2536 - Train Accuracy: 0.5417, Validation Accuracy: 0.6339, Loss: 2.0230
    Epoch   2 Batch 2200/2536 - Train Accuracy: 0.6500, Validation Accuracy: 0.6339, Loss: 1.9123
    Epoch   2 Batch 2300/2536 - Train Accuracy: 0.4543, Validation Accuracy: 0.6339, Loss: 3.3482
    Epoch   2 Batch 2400/2536 - Train Accuracy: 0.5721, Validation Accuracy: 0.6339, Loss: 2.4116
    Epoch   2 Batch 2500/2536 - Train Accuracy: 0.4760, Validation Accuracy: 0.6339, Loss: 2.9548
    Epoch   3 Batch  100/2536 - Train Accuracy: 0.6989, Validation Accuracy: 0.6339, Loss: 1.6845
    Epoch   3 Batch  200/2536 - Train Accuracy: 0.7545, Validation Accuracy: 0.6339, Loss: 1.4443
    Epoch   3 Batch  300/2536 - Train Accuracy: 0.7500, Validation Accuracy: 0.6339, Loss: 1.2922
    Epoch   3 Batch  400/2536 - Train Accuracy: 0.6914, Validation Accuracy: 0.6339, Loss: 1.7133
    Epoch   3 Batch  500/2536 - Train Accuracy: 0.6312, Validation Accuracy: 0.6339, Loss: 2.2221
    Epoch   3 Batch  600/2536 - Train Accuracy: 0.6375, Validation Accuracy: 0.6339, Loss: 2.0591
    Epoch   3 Batch  700/2536 - Train Accuracy: 0.6178, Validation Accuracy: 0.6339, Loss: 2.1516
    Epoch   3 Batch  800/2536 - Train Accuracy: 0.6298, Validation Accuracy: 0.6339, Loss: 2.2143
    Epoch   3 Batch  900/2536 - Train Accuracy: 0.5223, Validation Accuracy: 0.6339, Loss: 2.3201
    Epoch   3 Batch 1000/2536 - Train Accuracy: 0.6042, Validation Accuracy: 0.6339, Loss: 2.0954
    Epoch   3 Batch 1100/2536 - Train Accuracy: 0.4308, Validation Accuracy: 0.6339, Loss: 2.8656
    Epoch   3 Batch 1200/2536 - Train Accuracy: 0.5547, Validation Accuracy: 0.6339, Loss: 2.1372
    Epoch   3 Batch 1300/2536 - Train Accuracy: 0.7740, Validation Accuracy: 0.6339, Loss: 1.2151
    Epoch   3 Batch 1400/2536 - Train Accuracy: 0.5896, Validation Accuracy: 0.6339, Loss: 2.3213
    Epoch   3 Batch 1500/2536 - Train Accuracy: 0.7212, Validation Accuracy: 0.6339, Loss: 1.5167
    Epoch   3 Batch 1600/2536 - Train Accuracy: 0.6589, Validation Accuracy: 0.6339, Loss: 1.8671
    Epoch   3 Batch 1700/2536 - Train Accuracy: 0.7273, Validation Accuracy: 0.6339, Loss: 1.5243
    Epoch   3 Batch 1800/2536 - Train Accuracy: 0.7955, Validation Accuracy: 0.6339, Loss: 1.0825
    Epoch   3 Batch 1900/2536 - Train Accuracy: 0.6719, Validation Accuracy: 0.6339, Loss: 1.6258
    Epoch   3 Batch 2000/2536 - Train Accuracy: 0.6755, Validation Accuracy: 0.6339, Loss: 1.7238
    Epoch   3 Batch 2100/2536 - Train Accuracy: 0.6510, Validation Accuracy: 0.6339, Loss: 1.4436
    Epoch   3 Batch 2200/2536 - Train Accuracy: 0.6646, Validation Accuracy: 0.6339, Loss: 1.6853
    Epoch   3 Batch 2300/2536 - Train Accuracy: 0.4856, Validation Accuracy: 0.6339, Loss: 2.9567
    Epoch   3 Batch 2400/2536 - Train Accuracy: 0.5721, Validation Accuracy: 0.6362, Loss: 2.1496
    Epoch   3 Batch 2500/2536 - Train Accuracy: 0.5072, Validation Accuracy: 0.6339, Loss: 2.6819
    Epoch   4 Batch  100/2536 - Train Accuracy: 0.7500, Validation Accuracy: 0.6339, Loss: 1.4768
    Epoch   4 Batch  200/2536 - Train Accuracy: 0.7790, Validation Accuracy: 0.6339, Loss: 1.3142
    Epoch   4 Batch  300/2536 - Train Accuracy: 0.7612, Validation Accuracy: 0.6339, Loss: 1.1365
    Epoch   4 Batch  400/2536 - Train Accuracy: 0.7109, Validation Accuracy: 0.6339, Loss: 1.5150
    Epoch   4 Batch  500/2536 - Train Accuracy: 0.6521, Validation Accuracy: 0.6339, Loss: 1.9431
    Epoch   4 Batch  600/2536 - Train Accuracy: 0.6542, Validation Accuracy: 0.6339, Loss: 1.8600
    Epoch   4 Batch  700/2536 - Train Accuracy: 0.6298, Validation Accuracy: 0.6339, Loss: 1.9284
    Epoch   4 Batch  800/2536 - Train Accuracy: 0.6442, Validation Accuracy: 0.6339, Loss: 1.9823
    Epoch   4 Batch  900/2536 - Train Accuracy: 0.5625, Validation Accuracy: 0.6339, Loss: 2.1051
    Epoch   4 Batch 1000/2536 - Train Accuracy: 0.6042, Validation Accuracy: 0.6339, Loss: 1.8825
    Epoch   4 Batch 1100/2536 - Train Accuracy: 0.4866, Validation Accuracy: 0.6339, Loss: 2.6423
    Epoch   4 Batch 1200/2536 - Train Accuracy: 0.5547, Validation Accuracy: 0.6339, Loss: 1.9372
    Epoch   4 Batch 1300/2536 - Train Accuracy: 0.8029, Validation Accuracy: 0.6339, Loss: 0.9568
    Epoch   4 Batch 1400/2536 - Train Accuracy: 0.5854, Validation Accuracy: 0.6339, Loss: 2.0906
    Epoch   4 Batch 1500/2536 - Train Accuracy: 0.7476, Validation Accuracy: 0.6339, Loss: 1.3430
    Epoch   4 Batch 1600/2536 - Train Accuracy: 0.6875, Validation Accuracy: 0.6339, Loss: 1.6284
    Epoch   4 Batch 1700/2536 - Train Accuracy: 0.7642, Validation Accuracy: 0.6339, Loss: 1.3230
    Epoch   4 Batch 1800/2536 - Train Accuracy: 0.8182, Validation Accuracy: 0.6339, Loss: 0.9572
    Epoch   4 Batch 1900/2536 - Train Accuracy: 0.6719, Validation Accuracy: 0.6339, Loss: 1.4391
    Epoch   4 Batch 2000/2536 - Train Accuracy: 0.6875, Validation Accuracy: 0.6339, Loss: 1.5467
    Epoch   4 Batch 2100/2536 - Train Accuracy: 0.7995, Validation Accuracy: 0.6362, Loss: 0.9944
    Epoch   4 Batch 2200/2536 - Train Accuracy: 0.6875, Validation Accuracy: 0.6339, Loss: 1.5083
    Epoch   4 Batch 2300/2536 - Train Accuracy: 0.5120, Validation Accuracy: 0.6362, Loss: 2.6629
    Epoch   4 Batch 2400/2536 - Train Accuracy: 0.6178, Validation Accuracy: 0.6362, Loss: 1.9105
    Epoch   4 Batch 2500/2536 - Train Accuracy: 0.5192, Validation Accuracy: 0.6339, Loss: 2.4228
    Epoch   5 Batch  100/2536 - Train Accuracy: 0.7472, Validation Accuracy: 0.6339, Loss: 1.2848
    Epoch   5 Batch  200/2536 - Train Accuracy: 0.7746, Validation Accuracy: 0.6339, Loss: 1.1667
    Epoch   5 Batch  300/2536 - Train Accuracy: 0.7835, Validation Accuracy: 0.6339, Loss: 0.9963
    Epoch   5 Batch  400/2536 - Train Accuracy: 0.7344, Validation Accuracy: 0.6339, Loss: 1.3268
    Epoch   5 Batch  500/2536 - Train Accuracy: 0.6646, Validation Accuracy: 0.6339, Loss: 1.7425
    Epoch   5 Batch  600/2536 - Train Accuracy: 0.6562, Validation Accuracy: 0.6339, Loss: 1.6514
    Epoch   5 Batch  700/2536 - Train Accuracy: 0.6370, Validation Accuracy: 0.6339, Loss: 1.7199
    Epoch   5 Batch  800/2536 - Train Accuracy: 0.6587, Validation Accuracy: 0.6339, Loss: 1.7668
    Epoch   5 Batch  900/2536 - Train Accuracy: 0.6071, Validation Accuracy: 0.6339, Loss: 1.9057
    Epoch   5 Batch 1000/2536 - Train Accuracy: 0.6229, Validation Accuracy: 0.6339, Loss: 1.7293
    Epoch   5 Batch 1100/2536 - Train Accuracy: 0.4710, Validation Accuracy: 0.6339, Loss: 2.4221
    Epoch   5 Batch 1200/2536 - Train Accuracy: 0.5664, Validation Accuracy: 0.6339, Loss: 1.7613
    Epoch   5 Batch 1300/2536 - Train Accuracy: 0.8293, Validation Accuracy: 0.6339, Loss: 0.7494
    Epoch   5 Batch 1400/2536 - Train Accuracy: 0.6104, Validation Accuracy: 0.6339, Loss: 1.9493
    Epoch   5 Batch 1500/2536 - Train Accuracy: 0.7572, Validation Accuracy: 0.6339, Loss: 1.1972
    Epoch   5 Batch 1600/2536 - Train Accuracy: 0.6901, Validation Accuracy: 0.6339, Loss: 1.4559
    Epoch   5 Batch 1700/2536 - Train Accuracy: 0.7614, Validation Accuracy: 0.6339, Loss: 1.1539
    Epoch   5 Batch 1800/2536 - Train Accuracy: 0.8381, Validation Accuracy: 0.6339, Loss: 0.8309
    Epoch   5 Batch 1900/2536 - Train Accuracy: 0.6901, Validation Accuracy: 0.6339, Loss: 1.2847
    Epoch   5 Batch 2000/2536 - Train Accuracy: 0.6707, Validation Accuracy: 0.6339, Loss: 1.3231
    Epoch   5 Batch 2100/2536 - Train Accuracy: 0.8594, Validation Accuracy: 0.6339, Loss: 0.6827
    Epoch   5 Batch 2200/2536 - Train Accuracy: 0.7271, Validation Accuracy: 0.6339, Loss: 1.3427
    Epoch   5 Batch 2300/2536 - Train Accuracy: 0.5337, Validation Accuracy: 0.6339, Loss: 2.3964
    Epoch   5 Batch 2400/2536 - Train Accuracy: 0.6178, Validation Accuracy: 0.6339, Loss: 1.7478
    Epoch   5 Batch 2500/2536 - Train Accuracy: 0.5312, Validation Accuracy: 0.6362, Loss: 2.1873
    Epoch   6 Batch  100/2536 - Train Accuracy: 0.7869, Validation Accuracy: 0.6339, Loss: 1.0875
    Epoch   6 Batch  200/2536 - Train Accuracy: 0.7879, Validation Accuracy: 0.6339, Loss: 1.0442
    Epoch   6 Batch  300/2536 - Train Accuracy: 0.7991, Validation Accuracy: 0.6339, Loss: 0.8817
    Epoch   6 Batch  400/2536 - Train Accuracy: 0.7305, Validation Accuracy: 0.6339, Loss: 1.1920
    Epoch   6 Batch  500/2536 - Train Accuracy: 0.6771, Validation Accuracy: 0.6339, Loss: 1.5779
    Epoch   6 Batch  600/2536 - Train Accuracy: 0.6833, Validation Accuracy: 0.6339, Loss: 1.4556
    Epoch   6 Batch  700/2536 - Train Accuracy: 0.6659, Validation Accuracy: 0.6339, Loss: 1.5407
    Epoch   6 Batch  800/2536 - Train Accuracy: 0.6707, Validation Accuracy: 0.6339, Loss: 1.6016
    Epoch   6 Batch  900/2536 - Train Accuracy: 0.6049, Validation Accuracy: 0.6339, Loss: 1.6992
    Epoch   6 Batch 1000/2536 - Train Accuracy: 0.6417, Validation Accuracy: 0.6339, Loss: 1.5995
    Epoch   6 Batch 1100/2536 - Train Accuracy: 0.5089, Validation Accuracy: 0.6339, Loss: 2.2003
    Epoch   6 Batch 1200/2536 - Train Accuracy: 0.6133, Validation Accuracy: 0.6339, Loss: 1.5787
    Epoch   6 Batch 1300/2536 - Train Accuracy: 0.8726, Validation Accuracy: 0.6339, Loss: 0.5489
    Epoch   6 Batch 1400/2536 - Train Accuracy: 0.6146, Validation Accuracy: 0.6339, Loss: 1.7816
    Epoch   6 Batch 1500/2536 - Train Accuracy: 0.7740, Validation Accuracy: 0.6339, Loss: 1.0663
    Epoch   6 Batch 1600/2536 - Train Accuracy: 0.7057, Validation Accuracy: 0.6339, Loss: 1.2852
    Epoch   6 Batch 1700/2536 - Train Accuracy: 0.7670, Validation Accuracy: 0.6339, Loss: 1.0108
    Epoch   6 Batch 1800/2536 - Train Accuracy: 0.8409, Validation Accuracy: 0.6339, Loss: 0.7354
    Epoch   6 Batch 1900/2536 - Train Accuracy: 0.7292, Validation Accuracy: 0.6339, Loss: 1.1217
    Epoch   6 Batch 2000/2536 - Train Accuracy: 0.7308, Validation Accuracy: 0.6339, Loss: 1.1484
    Epoch   6 Batch 2100/2536 - Train Accuracy: 0.8906, Validation Accuracy: 0.6339, Loss: 0.4710
    Epoch   6 Batch 2200/2536 - Train Accuracy: 0.7188, Validation Accuracy: 0.6339, Loss: 1.2289
    Epoch   6 Batch 2300/2536 - Train Accuracy: 0.5625, Validation Accuracy: 0.6362, Loss: 2.1687
    Epoch   6 Batch 2400/2536 - Train Accuracy: 0.6538, Validation Accuracy: 0.6362, Loss: 1.5524
    Epoch   6 Batch 2500/2536 - Train Accuracy: 0.5409, Validation Accuracy: 0.6362, Loss: 2.0072
    Epoch   7 Batch  100/2536 - Train Accuracy: 0.7841, Validation Accuracy: 0.6362, Loss: 0.9128
    Epoch   7 Batch  200/2536 - Train Accuracy: 0.7969, Validation Accuracy: 0.6339, Loss: 0.9725
    Epoch   7 Batch  300/2536 - Train Accuracy: 0.8192, Validation Accuracy: 0.6339, Loss: 0.7962
    Epoch   7 Batch  400/2536 - Train Accuracy: 0.7461, Validation Accuracy: 0.6339, Loss: 1.0636
    Epoch   7 Batch  500/2536 - Train Accuracy: 0.6875, Validation Accuracy: 0.6339, Loss: 1.4067
    Epoch   7 Batch  600/2536 - Train Accuracy: 0.6917, Validation Accuracy: 0.6339, Loss: 1.3458
    Epoch   7 Batch  700/2536 - Train Accuracy: 0.6947, Validation Accuracy: 0.6339, Loss: 1.3765
    Epoch   7 Batch  800/2536 - Train Accuracy: 0.6971, Validation Accuracy: 0.6339, Loss: 1.4218
    Epoch   7 Batch  900/2536 - Train Accuracy: 0.6429, Validation Accuracy: 0.6339, Loss: 1.5395
    Epoch   7 Batch 1000/2536 - Train Accuracy: 0.6521, Validation Accuracy: 0.6339, Loss: 1.4689
    Epoch   7 Batch 1100/2536 - Train Accuracy: 0.5268, Validation Accuracy: 0.6339, Loss: 2.0079
    Epoch   7 Batch 1200/2536 - Train Accuracy: 0.6406, Validation Accuracy: 0.6339, Loss: 1.4564
    Epoch   7 Batch 1300/2536 - Train Accuracy: 0.9159, Validation Accuracy: 0.6339, Loss: 0.4087
    Epoch   7 Batch 1400/2536 - Train Accuracy: 0.6104, Validation Accuracy: 0.6339, Loss: 1.6274
    Epoch   7 Batch 1500/2536 - Train Accuracy: 0.8005, Validation Accuracy: 0.6339, Loss: 0.9113
    Epoch   7 Batch 1600/2536 - Train Accuracy: 0.7396, Validation Accuracy: 0.6339, Loss: 1.1367
    Epoch   7 Batch 1700/2536 - Train Accuracy: 0.7642, Validation Accuracy: 0.6339, Loss: 0.9012
    Epoch   7 Batch 1800/2536 - Train Accuracy: 0.8551, Validation Accuracy: 0.6339, Loss: 0.6265
    Epoch   7 Batch 1900/2536 - Train Accuracy: 0.7526, Validation Accuracy: 0.6339, Loss: 0.9725
    Epoch   7 Batch 2000/2536 - Train Accuracy: 0.7476, Validation Accuracy: 0.6339, Loss: 1.0058
    Epoch   7 Batch 2100/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6339, Loss: 0.3652
    Epoch   7 Batch 2200/2536 - Train Accuracy: 0.7333, Validation Accuracy: 0.6339, Loss: 1.0731
    Epoch   7 Batch 2300/2536 - Train Accuracy: 0.5962, Validation Accuracy: 0.6339, Loss: 1.9688
    Epoch   7 Batch 2400/2536 - Train Accuracy: 0.6587, Validation Accuracy: 0.6362, Loss: 1.3791
    Epoch   7 Batch 2500/2536 - Train Accuracy: 0.5697, Validation Accuracy: 0.6362, Loss: 1.8299
    Epoch   8 Batch  100/2536 - Train Accuracy: 0.8125, Validation Accuracy: 0.6339, Loss: 0.7784
    Epoch   8 Batch  200/2536 - Train Accuracy: 0.7991, Validation Accuracy: 0.6339, Loss: 0.8513
    Epoch   8 Batch  300/2536 - Train Accuracy: 0.8393, Validation Accuracy: 0.6339, Loss: 0.6838
    Epoch   8 Batch  400/2536 - Train Accuracy: 0.7617, Validation Accuracy: 0.6339, Loss: 0.9369
    Epoch   8 Batch  500/2536 - Train Accuracy: 0.7083, Validation Accuracy: 0.6339, Loss: 1.2918
    Epoch   8 Batch  600/2536 - Train Accuracy: 0.7500, Validation Accuracy: 0.6339, Loss: 1.1843
    Epoch   8 Batch  700/2536 - Train Accuracy: 0.7139, Validation Accuracy: 0.6339, Loss: 1.2867
    Epoch   8 Batch  800/2536 - Train Accuracy: 0.7212, Validation Accuracy: 0.6339, Loss: 1.2979
    Epoch   8 Batch  900/2536 - Train Accuracy: 0.6585, Validation Accuracy: 0.6339, Loss: 1.4299
    Epoch   8 Batch 1000/2536 - Train Accuracy: 0.6687, Validation Accuracy: 0.6339, Loss: 1.3261
    Epoch   8 Batch 1100/2536 - Train Accuracy: 0.5871, Validation Accuracy: 0.6339, Loss: 1.8941
    Epoch   8 Batch 1200/2536 - Train Accuracy: 0.6426, Validation Accuracy: 0.6339, Loss: 1.3430
    Epoch   8 Batch 1300/2536 - Train Accuracy: 0.9423, Validation Accuracy: 0.6339, Loss: 0.2871
    Epoch   8 Batch 1400/2536 - Train Accuracy: 0.6396, Validation Accuracy: 0.6339, Loss: 1.4911
    Epoch   8 Batch 1500/2536 - Train Accuracy: 0.8293, Validation Accuracy: 0.6339, Loss: 0.8203
    Epoch   8 Batch 1600/2536 - Train Accuracy: 0.7552, Validation Accuracy: 0.6339, Loss: 1.0032
    Epoch   8 Batch 1700/2536 - Train Accuracy: 0.8182, Validation Accuracy: 0.6339, Loss: 0.7849
    Epoch   8 Batch 1800/2536 - Train Accuracy: 0.8608, Validation Accuracy: 0.6339, Loss: 0.5227
    Epoch   8 Batch 1900/2536 - Train Accuracy: 0.8177, Validation Accuracy: 0.6339, Loss: 0.8714
    Epoch   8 Batch 2000/2536 - Train Accuracy: 0.7548, Validation Accuracy: 0.6339, Loss: 0.9017
    Epoch   8 Batch 2100/2536 - Train Accuracy: 0.9297, Validation Accuracy: 0.6339, Loss: 0.3078
    Epoch   8 Batch 2200/2536 - Train Accuracy: 0.7438, Validation Accuracy: 0.6339, Loss: 0.9836
    Epoch   8 Batch 2300/2536 - Train Accuracy: 0.6154, Validation Accuracy: 0.6339, Loss: 1.7652
    Epoch   8 Batch 2400/2536 - Train Accuracy: 0.6971, Validation Accuracy: 0.6339, Loss: 1.2424
    Epoch   8 Batch 2500/2536 - Train Accuracy: 0.6010, Validation Accuracy: 0.6362, Loss: 1.6186
    Epoch   9 Batch  100/2536 - Train Accuracy: 0.8324, Validation Accuracy: 0.6362, Loss: 0.6458
    Epoch   9 Batch  200/2536 - Train Accuracy: 0.8147, Validation Accuracy: 0.6339, Loss: 0.7763
    Epoch   9 Batch  300/2536 - Train Accuracy: 0.8438, Validation Accuracy: 0.6339, Loss: 0.6093
    Epoch   9 Batch  400/2536 - Train Accuracy: 0.7988, Validation Accuracy: 0.6339, Loss: 0.8444
    Epoch   9 Batch  500/2536 - Train Accuracy: 0.7021, Validation Accuracy: 0.6339, Loss: 1.1807
    Epoch   9 Batch  600/2536 - Train Accuracy: 0.7396, Validation Accuracy: 0.6339, Loss: 1.0820
    Epoch   9 Batch  700/2536 - Train Accuracy: 0.7139, Validation Accuracy: 0.6339, Loss: 1.1265
    Epoch   9 Batch  800/2536 - Train Accuracy: 0.7236, Validation Accuracy: 0.6339, Loss: 1.1705
    Epoch   9 Batch  900/2536 - Train Accuracy: 0.6942, Validation Accuracy: 0.6339, Loss: 1.2653
    Epoch   9 Batch 1000/2536 - Train Accuracy: 0.7083, Validation Accuracy: 0.6339, Loss: 1.1997
    Epoch   9 Batch 1100/2536 - Train Accuracy: 0.5826, Validation Accuracy: 0.6339, Loss: 1.7286
    Epoch   9 Batch 1200/2536 - Train Accuracy: 0.6641, Validation Accuracy: 0.6339, Loss: 1.1676
    Epoch   9 Batch 1300/2536 - Train Accuracy: 0.9495, Validation Accuracy: 0.6339, Loss: 0.2076
    Epoch   9 Batch 1400/2536 - Train Accuracy: 0.6687, Validation Accuracy: 0.6339, Loss: 1.3317
    Epoch   9 Batch 1500/2536 - Train Accuracy: 0.8341, Validation Accuracy: 0.6339, Loss: 0.6981
    Epoch   9 Batch 1600/2536 - Train Accuracy: 0.7760, Validation Accuracy: 0.6339, Loss: 0.8754
    Epoch   9 Batch 1700/2536 - Train Accuracy: 0.8352, Validation Accuracy: 0.6339, Loss: 0.6527
    Epoch   9 Batch 1800/2536 - Train Accuracy: 0.8807, Validation Accuracy: 0.6339, Loss: 0.4398
    Epoch   9 Batch 1900/2536 - Train Accuracy: 0.8047, Validation Accuracy: 0.6339, Loss: 0.7624
    Epoch   9 Batch 2000/2536 - Train Accuracy: 0.7837, Validation Accuracy: 0.6362, Loss: 0.8088
    Epoch   9 Batch 2100/2536 - Train Accuracy: 0.9479, Validation Accuracy: 0.6362, Loss: 0.2412
    Epoch   9 Batch 2200/2536 - Train Accuracy: 0.7854, Validation Accuracy: 0.6339, Loss: 0.9118
    Epoch   9 Batch 2300/2536 - Train Accuracy: 0.6274, Validation Accuracy: 0.6339, Loss: 1.5524
    Epoch   9 Batch 2400/2536 - Train Accuracy: 0.7067, Validation Accuracy: 0.6339, Loss: 1.1247
    Epoch   9 Batch 2500/2536 - Train Accuracy: 0.6226, Validation Accuracy: 0.6362, Loss: 1.4897
    Epoch  10 Batch  100/2536 - Train Accuracy: 0.8722, Validation Accuracy: 0.6339, Loss: 0.5428
    Epoch  10 Batch  200/2536 - Train Accuracy: 0.8192, Validation Accuracy: 0.6339, Loss: 0.7102
    Epoch  10 Batch  300/2536 - Train Accuracy: 0.8571, Validation Accuracy: 0.6339, Loss: 0.5346
    Epoch  10 Batch  400/2536 - Train Accuracy: 0.8027, Validation Accuracy: 0.6339, Loss: 0.7629
    Epoch  10 Batch  500/2536 - Train Accuracy: 0.7042, Validation Accuracy: 0.6339, Loss: 1.0385
    Epoch  10 Batch  600/2536 - Train Accuracy: 0.7750, Validation Accuracy: 0.6339, Loss: 0.9570
    Epoch  10 Batch  700/2536 - Train Accuracy: 0.7716, Validation Accuracy: 0.6339, Loss: 0.9985
    Epoch  10 Batch  800/2536 - Train Accuracy: 0.7476, Validation Accuracy: 0.6339, Loss: 1.0381
    Epoch  10 Batch  900/2536 - Train Accuracy: 0.7165, Validation Accuracy: 0.6339, Loss: 1.1555
    Epoch  10 Batch 1000/2536 - Train Accuracy: 0.7021, Validation Accuracy: 0.6339, Loss: 1.1061
    Epoch  10 Batch 1100/2536 - Train Accuracy: 0.5804, Validation Accuracy: 0.6339, Loss: 1.6040
    Epoch  10 Batch 1200/2536 - Train Accuracy: 0.7070, Validation Accuracy: 0.6339, Loss: 1.0475
    Epoch  10 Batch 1300/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6339, Loss: 0.1692
    Epoch  10 Batch 1400/2536 - Train Accuracy: 0.6833, Validation Accuracy: 0.6339, Loss: 1.2304
    Epoch  10 Batch 1500/2536 - Train Accuracy: 0.8558, Validation Accuracy: 0.6339, Loss: 0.5882
    Epoch  10 Batch 1600/2536 - Train Accuracy: 0.7917, Validation Accuracy: 0.6339, Loss: 0.7720
    Epoch  10 Batch 1700/2536 - Train Accuracy: 0.8636, Validation Accuracy: 0.6339, Loss: 0.5777
    Epoch  10 Batch 1800/2536 - Train Accuracy: 0.8977, Validation Accuracy: 0.6339, Loss: 0.3838
    Epoch  10 Batch 1900/2536 - Train Accuracy: 0.7995, Validation Accuracy: 0.6339, Loss: 0.6861
    Epoch  10 Batch 2000/2536 - Train Accuracy: 0.8149, Validation Accuracy: 0.6339, Loss: 0.7019
    Epoch  10 Batch 2100/2536 - Train Accuracy: 0.9479, Validation Accuracy: 0.6339, Loss: 0.1919
    Epoch  10 Batch 2200/2536 - Train Accuracy: 0.7812, Validation Accuracy: 0.6339, Loss: 0.8546
    Epoch  10 Batch 2300/2536 - Train Accuracy: 0.6394, Validation Accuracy: 0.6339, Loss: 1.3787
    Epoch  10 Batch 2400/2536 - Train Accuracy: 0.7308, Validation Accuracy: 0.6339, Loss: 1.0443
    Epoch  10 Batch 2500/2536 - Train Accuracy: 0.6587, Validation Accuracy: 0.6339, Loss: 1.3067
    Epoch  11 Batch  100/2536 - Train Accuracy: 0.8778, Validation Accuracy: 0.6362, Loss: 0.4634
    Epoch  11 Batch  200/2536 - Train Accuracy: 0.8192, Validation Accuracy: 0.6362, Loss: 0.6294
    Epoch  11 Batch  300/2536 - Train Accuracy: 0.8728, Validation Accuracy: 0.6339, Loss: 0.4885
    Epoch  11 Batch  400/2536 - Train Accuracy: 0.8242, Validation Accuracy: 0.6339, Loss: 0.6609
    Epoch  11 Batch  500/2536 - Train Accuracy: 0.7083, Validation Accuracy: 0.6339, Loss: 0.9651
    Epoch  11 Batch  600/2536 - Train Accuracy: 0.7688, Validation Accuracy: 0.6339, Loss: 0.8411
    Epoch  11 Batch  700/2536 - Train Accuracy: 0.7981, Validation Accuracy: 0.6339, Loss: 0.8960
    Epoch  11 Batch  800/2536 - Train Accuracy: 0.7668, Validation Accuracy: 0.6339, Loss: 0.9194
    Epoch  11 Batch  900/2536 - Train Accuracy: 0.7366, Validation Accuracy: 0.6339, Loss: 1.0451
    Epoch  11 Batch 1000/2536 - Train Accuracy: 0.7333, Validation Accuracy: 0.6339, Loss: 1.0312
    Epoch  11 Batch 1100/2536 - Train Accuracy: 0.6138, Validation Accuracy: 0.6339, Loss: 1.4867
    Epoch  11 Batch 1200/2536 - Train Accuracy: 0.7012, Validation Accuracy: 0.6339, Loss: 0.9629
    Epoch  11 Batch 1300/2536 - Train Accuracy: 0.9712, Validation Accuracy: 0.6339, Loss: 0.1233
    Epoch  11 Batch 1400/2536 - Train Accuracy: 0.7042, Validation Accuracy: 0.6339, Loss: 1.1274
    Epoch  11 Batch 1500/2536 - Train Accuracy: 0.8822, Validation Accuracy: 0.6339, Loss: 0.5209
    Epoch  11 Batch 1600/2536 - Train Accuracy: 0.8099, Validation Accuracy: 0.6339, Loss: 0.6645
    Epoch  11 Batch 1700/2536 - Train Accuracy: 0.8693, Validation Accuracy: 0.6339, Loss: 0.5186
    Epoch  11 Batch 1800/2536 - Train Accuracy: 0.9091, Validation Accuracy: 0.6339, Loss: 0.3332
    Epoch  11 Batch 1900/2536 - Train Accuracy: 0.8490, Validation Accuracy: 0.6339, Loss: 0.5979
    Epoch  11 Batch 2000/2536 - Train Accuracy: 0.7957, Validation Accuracy: 0.6339, Loss: 0.6329
    Epoch  11 Batch 2100/2536 - Train Accuracy: 0.9505, Validation Accuracy: 0.6362, Loss: 0.1619
    Epoch  11 Batch 2200/2536 - Train Accuracy: 0.7958, Validation Accuracy: 0.6339, Loss: 0.7380
    Epoch  11 Batch 2300/2536 - Train Accuracy: 0.6731, Validation Accuracy: 0.6339, Loss: 1.2446
    Epoch  11 Batch 2400/2536 - Train Accuracy: 0.7572, Validation Accuracy: 0.6339, Loss: 0.9074
    Epoch  11 Batch 2500/2536 - Train Accuracy: 0.6346, Validation Accuracy: 0.6362, Loss: 1.2094
    Epoch  12 Batch  100/2536 - Train Accuracy: 0.8835, Validation Accuracy: 0.6362, Loss: 0.4094
    Epoch  12 Batch  200/2536 - Train Accuracy: 0.8326, Validation Accuracy: 0.6339, Loss: 0.6349
    Epoch  12 Batch  300/2536 - Train Accuracy: 0.8616, Validation Accuracy: 0.6339, Loss: 0.4610
    Epoch  12 Batch  400/2536 - Train Accuracy: 0.8223, Validation Accuracy: 0.6339, Loss: 0.5801
    Epoch  12 Batch  500/2536 - Train Accuracy: 0.7312, Validation Accuracy: 0.6339, Loss: 0.8859
    Epoch  12 Batch  600/2536 - Train Accuracy: 0.7875, Validation Accuracy: 0.6339, Loss: 0.7784
    Epoch  12 Batch  700/2536 - Train Accuracy: 0.7668, Validation Accuracy: 0.6339, Loss: 0.8487
    Epoch  12 Batch  800/2536 - Train Accuracy: 0.7788, Validation Accuracy: 0.6339, Loss: 0.8324
    Epoch  12 Batch  900/2536 - Train Accuracy: 0.7165, Validation Accuracy: 0.6339, Loss: 0.9440
    Epoch  12 Batch 1000/2536 - Train Accuracy: 0.7312, Validation Accuracy: 0.6339, Loss: 0.9816
    Epoch  12 Batch 1100/2536 - Train Accuracy: 0.6384, Validation Accuracy: 0.6339, Loss: 1.3769
    Epoch  12 Batch 1200/2536 - Train Accuracy: 0.7305, Validation Accuracy: 0.6339, Loss: 0.8984
    Epoch  12 Batch 1300/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6339, Loss: 0.1023
    Epoch  12 Batch 1400/2536 - Train Accuracy: 0.7083, Validation Accuracy: 0.6339, Loss: 1.0539
    Epoch  12 Batch 1500/2536 - Train Accuracy: 0.8750, Validation Accuracy: 0.6339, Loss: 0.4562
    Epoch  12 Batch 1600/2536 - Train Accuracy: 0.8854, Validation Accuracy: 0.6339, Loss: 0.5901
    Epoch  12 Batch 1700/2536 - Train Accuracy: 0.8892, Validation Accuracy: 0.6339, Loss: 0.4436
    Epoch  12 Batch 1800/2536 - Train Accuracy: 0.9347, Validation Accuracy: 0.6339, Loss: 0.2875
    Epoch  12 Batch 1900/2536 - Train Accuracy: 0.8646, Validation Accuracy: 0.6339, Loss: 0.5281
    Epoch  12 Batch 2000/2536 - Train Accuracy: 0.8029, Validation Accuracy: 0.6339, Loss: 0.5623
    Epoch  12 Batch 2100/2536 - Train Accuracy: 0.9661, Validation Accuracy: 0.6339, Loss: 0.1434
    Epoch  12 Batch 2200/2536 - Train Accuracy: 0.8000, Validation Accuracy: 0.6339, Loss: 0.6421
    Epoch  12 Batch 2300/2536 - Train Accuracy: 0.6803, Validation Accuracy: 0.6339, Loss: 1.1443
    Epoch  12 Batch 2400/2536 - Train Accuracy: 0.7620, Validation Accuracy: 0.6362, Loss: 0.8064
    Epoch  12 Batch 2500/2536 - Train Accuracy: 0.7212, Validation Accuracy: 0.6362, Loss: 1.1051
    Epoch  13 Batch  100/2536 - Train Accuracy: 0.9233, Validation Accuracy: 0.6362, Loss: 0.3470
    Epoch  13 Batch  200/2536 - Train Accuracy: 0.8482, Validation Accuracy: 0.6339, Loss: 0.5485
    Epoch  13 Batch  300/2536 - Train Accuracy: 0.8772, Validation Accuracy: 0.6339, Loss: 0.4200
    Epoch  13 Batch  400/2536 - Train Accuracy: 0.8105, Validation Accuracy: 0.6339, Loss: 0.5442
    Epoch  13 Batch  500/2536 - Train Accuracy: 0.7854, Validation Accuracy: 0.6339, Loss: 0.7985
    Epoch  13 Batch  600/2536 - Train Accuracy: 0.8083, Validation Accuracy: 0.6339, Loss: 0.7594
    Epoch  13 Batch  700/2536 - Train Accuracy: 0.7861, Validation Accuracy: 0.6339, Loss: 0.7670
    Epoch  13 Batch  800/2536 - Train Accuracy: 0.7716, Validation Accuracy: 0.6339, Loss: 0.7565
    Epoch  13 Batch  900/2536 - Train Accuracy: 0.7723, Validation Accuracy: 0.6339, Loss: 0.8577
    Epoch  13 Batch 1000/2536 - Train Accuracy: 0.7479, Validation Accuracy: 0.6339, Loss: 0.9267
    Epoch  13 Batch 1100/2536 - Train Accuracy: 0.6607, Validation Accuracy: 0.6339, Loss: 1.2514
    Epoch  13 Batch 1200/2536 - Train Accuracy: 0.7188, Validation Accuracy: 0.6339, Loss: 0.8605
    Epoch  13 Batch 1300/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6339, Loss: 0.0910
    Epoch  13 Batch 1400/2536 - Train Accuracy: 0.7229, Validation Accuracy: 0.6362, Loss: 0.9404
    Epoch  13 Batch 1500/2536 - Train Accuracy: 0.9087, Validation Accuracy: 0.6339, Loss: 0.4081
    Epoch  13 Batch 1600/2536 - Train Accuracy: 0.8854, Validation Accuracy: 0.6339, Loss: 0.5269
    Epoch  13 Batch 1700/2536 - Train Accuracy: 0.9091, Validation Accuracy: 0.6339, Loss: 0.3881
    Epoch  13 Batch 1800/2536 - Train Accuracy: 0.9318, Validation Accuracy: 0.6339, Loss: 0.2653
    Epoch  13 Batch 1900/2536 - Train Accuracy: 0.8724, Validation Accuracy: 0.6339, Loss: 0.4934
    Epoch  13 Batch 2000/2536 - Train Accuracy: 0.7933, Validation Accuracy: 0.6339, Loss: 0.5334
    Epoch  13 Batch 2100/2536 - Train Accuracy: 0.9818, Validation Accuracy: 0.6362, Loss: 0.1180
    Epoch  13 Batch 2200/2536 - Train Accuracy: 0.8063, Validation Accuracy: 0.6339, Loss: 0.6204
    Epoch  13 Batch 2300/2536 - Train Accuracy: 0.7308, Validation Accuracy: 0.6339, Loss: 1.0405
    Epoch  13 Batch 2400/2536 - Train Accuracy: 0.7620, Validation Accuracy: 0.6339, Loss: 0.7486
    Epoch  13 Batch 2500/2536 - Train Accuracy: 0.7139, Validation Accuracy: 0.6339, Loss: 1.0378
    Epoch  14 Batch  100/2536 - Train Accuracy: 0.9233, Validation Accuracy: 0.6362, Loss: 0.3123
    Epoch  14 Batch  200/2536 - Train Accuracy: 0.8527, Validation Accuracy: 0.6339, Loss: 0.5304
    Epoch  14 Batch  300/2536 - Train Accuracy: 0.9040, Validation Accuracy: 0.6339, Loss: 0.3835
    Epoch  14 Batch  400/2536 - Train Accuracy: 0.8301, Validation Accuracy: 0.6339, Loss: 0.5265
    Epoch  14 Batch  500/2536 - Train Accuracy: 0.7604, Validation Accuracy: 0.6339, Loss: 0.7369
    Epoch  14 Batch  600/2536 - Train Accuracy: 0.8125, Validation Accuracy: 0.6339, Loss: 0.6646
    Epoch  14 Batch  700/2536 - Train Accuracy: 0.7668, Validation Accuracy: 0.6339, Loss: 0.7496
    Epoch  14 Batch  800/2536 - Train Accuracy: 0.8029, Validation Accuracy: 0.6339, Loss: 0.7437
    Epoch  14 Batch  900/2536 - Train Accuracy: 0.7656, Validation Accuracy: 0.6339, Loss: 0.8702
    Epoch  14 Batch 1000/2536 - Train Accuracy: 0.7271, Validation Accuracy: 0.6339, Loss: 0.8865
    Epoch  14 Batch 1100/2536 - Train Accuracy: 0.6540, Validation Accuracy: 0.6339, Loss: 1.1817
    Epoch  14 Batch 1200/2536 - Train Accuracy: 0.7148, Validation Accuracy: 0.6362, Loss: 0.8075
    Epoch  14 Batch 1300/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6339, Loss: 0.0660
    Epoch  14 Batch 1400/2536 - Train Accuracy: 0.7542, Validation Accuracy: 0.6339, Loss: 0.8757
    Epoch  14 Batch 1500/2536 - Train Accuracy: 0.8894, Validation Accuracy: 0.6339, Loss: 0.3733
    Epoch  14 Batch 1600/2536 - Train Accuracy: 0.8698, Validation Accuracy: 0.6339, Loss: 0.4220
    Epoch  14 Batch 1700/2536 - Train Accuracy: 0.9148, Validation Accuracy: 0.6339, Loss: 0.3423
    Epoch  14 Batch 1800/2536 - Train Accuracy: 0.9290, Validation Accuracy: 0.6339, Loss: 0.2305
    Epoch  14 Batch 1900/2536 - Train Accuracy: 0.8698, Validation Accuracy: 0.6339, Loss: 0.4500
    Epoch  14 Batch 2000/2536 - Train Accuracy: 0.8413, Validation Accuracy: 0.6339, Loss: 0.4870
    Epoch  14 Batch 2100/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6339, Loss: 0.0992
    Epoch  14 Batch 2200/2536 - Train Accuracy: 0.8229, Validation Accuracy: 0.6339, Loss: 0.5295
    Epoch  14 Batch 2300/2536 - Train Accuracy: 0.7067, Validation Accuracy: 0.6339, Loss: 0.9552
    Epoch  14 Batch 2400/2536 - Train Accuracy: 0.7933, Validation Accuracy: 0.6362, Loss: 0.7003
    Epoch  14 Batch 2500/2536 - Train Accuracy: 0.7260, Validation Accuracy: 0.6339, Loss: 0.9450
    Epoch  15 Batch  100/2536 - Train Accuracy: 0.9261, Validation Accuracy: 0.6339, Loss: 0.2557
    Epoch  15 Batch  200/2536 - Train Accuracy: 0.8549, Validation Accuracy: 0.6339, Loss: 0.4461
    Epoch  15 Batch  300/2536 - Train Accuracy: 0.9196, Validation Accuracy: 0.6339, Loss: 0.3409
    Epoch  15 Batch  400/2536 - Train Accuracy: 0.8184, Validation Accuracy: 0.6339, Loss: 0.5306
    Epoch  15 Batch  500/2536 - Train Accuracy: 0.7854, Validation Accuracy: 0.6339, Loss: 0.6825
    Epoch  15 Batch  600/2536 - Train Accuracy: 0.8146, Validation Accuracy: 0.6339, Loss: 0.6035
    Epoch  15 Batch  700/2536 - Train Accuracy: 0.7837, Validation Accuracy: 0.6339, Loss: 0.7035
    Epoch  15 Batch  800/2536 - Train Accuracy: 0.8053, Validation Accuracy: 0.6339, Loss: 0.6015
    Epoch  15 Batch  900/2536 - Train Accuracy: 0.7768, Validation Accuracy: 0.6339, Loss: 0.7679
    Epoch  15 Batch 1000/2536 - Train Accuracy: 0.7250, Validation Accuracy: 0.6339, Loss: 0.8219
    Epoch  15 Batch 1100/2536 - Train Accuracy: 0.7009, Validation Accuracy: 0.6339, Loss: 1.0815
    Epoch  15 Batch 1200/2536 - Train Accuracy: 0.7344, Validation Accuracy: 0.6339, Loss: 0.7431
    Epoch  15 Batch 1300/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6339, Loss: 0.0597
    Epoch  15 Batch 1400/2536 - Train Accuracy: 0.7604, Validation Accuracy: 0.6339, Loss: 0.7996
    Epoch  15 Batch 1500/2536 - Train Accuracy: 0.8822, Validation Accuracy: 0.6362, Loss: 0.3289
    Epoch  15 Batch 1600/2536 - Train Accuracy: 0.8880, Validation Accuracy: 0.6339, Loss: 0.3793
    Epoch  15 Batch 1700/2536 - Train Accuracy: 0.9119, Validation Accuracy: 0.6339, Loss: 0.2980
    Epoch  15 Batch 1800/2536 - Train Accuracy: 0.9318, Validation Accuracy: 0.6339, Loss: 0.2005
    Epoch  15 Batch 1900/2536 - Train Accuracy: 0.8516, Validation Accuracy: 0.6339, Loss: 0.3933
    Epoch  15 Batch 2000/2536 - Train Accuracy: 0.8534, Validation Accuracy: 0.6339, Loss: 0.4141
    Epoch  15 Batch 2100/2536 - Train Accuracy: 0.9896, Validation Accuracy: 0.6339, Loss: 0.0753
    Epoch  15 Batch 2200/2536 - Train Accuracy: 0.8479, Validation Accuracy: 0.6339, Loss: 0.5044
    Epoch  15 Batch 2300/2536 - Train Accuracy: 0.7452, Validation Accuracy: 0.6339, Loss: 0.8828
    Epoch  15 Batch 2400/2536 - Train Accuracy: 0.7909, Validation Accuracy: 0.6362, Loss: 0.6152
    Epoch  15 Batch 2500/2536 - Train Accuracy: 0.7236, Validation Accuracy: 0.6339, Loss: 0.8399
    Epoch  16 Batch  100/2536 - Train Accuracy: 0.9432, Validation Accuracy: 0.6339, Loss: 0.2181
    Epoch  16 Batch  200/2536 - Train Accuracy: 0.8683, Validation Accuracy: 0.6339, Loss: 0.4232
    Epoch  16 Batch  300/2536 - Train Accuracy: 0.9107, Validation Accuracy: 0.6339, Loss: 0.3240
    Epoch  16 Batch  400/2536 - Train Accuracy: 0.8398, Validation Accuracy: 0.6339, Loss: 0.4533
    Epoch  16 Batch  500/2536 - Train Accuracy: 0.7896, Validation Accuracy: 0.6339, Loss: 0.6238
    Epoch  16 Batch  600/2536 - Train Accuracy: 0.8229, Validation Accuracy: 0.6339, Loss: 0.5791
    Epoch  16 Batch  700/2536 - Train Accuracy: 0.7933, Validation Accuracy: 0.6339, Loss: 0.5842
    Epoch  16 Batch  800/2536 - Train Accuracy: 0.8221, Validation Accuracy: 0.6339, Loss: 0.5620
    Epoch  16 Batch  900/2536 - Train Accuracy: 0.7924, Validation Accuracy: 0.6339, Loss: 0.6802
    Epoch  16 Batch 1000/2536 - Train Accuracy: 0.7771, Validation Accuracy: 0.6339, Loss: 0.7689
    Epoch  16 Batch 1100/2536 - Train Accuracy: 0.6585, Validation Accuracy: 0.6339, Loss: 1.0008
    Epoch  16 Batch 1200/2536 - Train Accuracy: 0.7578, Validation Accuracy: 0.6339, Loss: 0.6878
    Epoch  16 Batch 1300/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0497
    Epoch  16 Batch 1400/2536 - Train Accuracy: 0.7438, Validation Accuracy: 0.6339, Loss: 0.7373
    Epoch  16 Batch 1500/2536 - Train Accuracy: 0.8846, Validation Accuracy: 0.6339, Loss: 0.2929
    Epoch  16 Batch 1600/2536 - Train Accuracy: 0.9271, Validation Accuracy: 0.6339, Loss: 0.3387
    Epoch  16 Batch 1700/2536 - Train Accuracy: 0.9205, Validation Accuracy: 0.6339, Loss: 0.2623
    Epoch  16 Batch 1800/2536 - Train Accuracy: 0.9290, Validation Accuracy: 0.6339, Loss: 0.1831
    Epoch  16 Batch 1900/2536 - Train Accuracy: 0.9062, Validation Accuracy: 0.6339, Loss: 0.3655
    Epoch  16 Batch 2000/2536 - Train Accuracy: 0.8606, Validation Accuracy: 0.6339, Loss: 0.3960
    Epoch  16 Batch 2100/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6339, Loss: 0.0702
    Epoch  16 Batch 2200/2536 - Train Accuracy: 0.8479, Validation Accuracy: 0.6339, Loss: 0.4593
    Epoch  16 Batch 2300/2536 - Train Accuracy: 0.7788, Validation Accuracy: 0.6362, Loss: 0.8173
    Epoch  16 Batch 2400/2536 - Train Accuracy: 0.8125, Validation Accuracy: 0.6339, Loss: 0.5823
    Epoch  16 Batch 2500/2536 - Train Accuracy: 0.7692, Validation Accuracy: 0.6362, Loss: 0.7758
    Epoch  17 Batch  100/2536 - Train Accuracy: 0.9347, Validation Accuracy: 0.6339, Loss: 0.2033
    Epoch  17 Batch  200/2536 - Train Accuracy: 0.8683, Validation Accuracy: 0.6339, Loss: 0.4011
    Epoch  17 Batch  300/2536 - Train Accuracy: 0.9174, Validation Accuracy: 0.6339, Loss: 0.2825
    Epoch  17 Batch  400/2536 - Train Accuracy: 0.8574, Validation Accuracy: 0.6339, Loss: 0.3635
    Epoch  17 Batch  500/2536 - Train Accuracy: 0.7667, Validation Accuracy: 0.6339, Loss: 0.5618
    Epoch  17 Batch  600/2536 - Train Accuracy: 0.8646, Validation Accuracy: 0.6339, Loss: 0.5477
    Epoch  17 Batch  700/2536 - Train Accuracy: 0.8510, Validation Accuracy: 0.6339, Loss: 0.5478
    Epoch  17 Batch  800/2536 - Train Accuracy: 0.8389, Validation Accuracy: 0.6339, Loss: 0.5256
    Epoch  17 Batch  900/2536 - Train Accuracy: 0.8036, Validation Accuracy: 0.6339, Loss: 0.6325
    Epoch  17 Batch 1000/2536 - Train Accuracy: 0.7688, Validation Accuracy: 0.6339, Loss: 0.6929
    Epoch  17 Batch 1100/2536 - Train Accuracy: 0.6897, Validation Accuracy: 0.6362, Loss: 0.9234
    Epoch  17 Batch 1200/2536 - Train Accuracy: 0.8008, Validation Accuracy: 0.6339, Loss: 0.5995
    Epoch  17 Batch 1300/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6339, Loss: 0.0491
    Epoch  17 Batch 1400/2536 - Train Accuracy: 0.7667, Validation Accuracy: 0.6339, Loss: 0.7229
    Epoch  17 Batch 1500/2536 - Train Accuracy: 0.9014, Validation Accuracy: 0.6339, Loss: 0.2699
    Epoch  17 Batch 1600/2536 - Train Accuracy: 0.9062, Validation Accuracy: 0.6362, Loss: 0.3079
    Epoch  17 Batch 1700/2536 - Train Accuracy: 0.9318, Validation Accuracy: 0.6339, Loss: 0.2661
    Epoch  17 Batch 1800/2536 - Train Accuracy: 0.9403, Validation Accuracy: 0.6362, Loss: 0.1594
    Epoch  17 Batch 1900/2536 - Train Accuracy: 0.8958, Validation Accuracy: 0.6339, Loss: 0.3527
    Epoch  17 Batch 2000/2536 - Train Accuracy: 0.8654, Validation Accuracy: 0.6339, Loss: 0.3394
    Epoch  17 Batch 2100/2536 - Train Accuracy: 0.9714, Validation Accuracy: 0.6339, Loss: 0.0658
    Epoch  17 Batch 2200/2536 - Train Accuracy: 0.8375, Validation Accuracy: 0.6339, Loss: 0.4305
    Epoch  17 Batch 2300/2536 - Train Accuracy: 0.7812, Validation Accuracy: 0.6339, Loss: 0.7052
    Epoch  17 Batch 2400/2536 - Train Accuracy: 0.7933, Validation Accuracy: 0.6362, Loss: 0.5209
    Epoch  17 Batch 2500/2536 - Train Accuracy: 0.8245, Validation Accuracy: 0.6362, Loss: 0.7085
    Epoch  18 Batch  100/2536 - Train Accuracy: 0.9517, Validation Accuracy: 0.6339, Loss: 0.1786
    Epoch  18 Batch  200/2536 - Train Accuracy: 0.8728, Validation Accuracy: 0.6339, Loss: 0.3486
    Epoch  18 Batch  300/2536 - Train Accuracy: 0.9219, Validation Accuracy: 0.6339, Loss: 0.2769
    Epoch  18 Batch  400/2536 - Train Accuracy: 0.8438, Validation Accuracy: 0.6339, Loss: 0.3719
    Epoch  18 Batch  500/2536 - Train Accuracy: 0.8354, Validation Accuracy: 0.6362, Loss: 0.5055
    Epoch  18 Batch  600/2536 - Train Accuracy: 0.8292, Validation Accuracy: 0.6339, Loss: 0.5301
    Epoch  18 Batch  700/2536 - Train Accuracy: 0.8269, Validation Accuracy: 0.6339, Loss: 0.4858
    Epoch  18 Batch  800/2536 - Train Accuracy: 0.8558, Validation Accuracy: 0.6339, Loss: 0.5173
    Epoch  18 Batch  900/2536 - Train Accuracy: 0.8125, Validation Accuracy: 0.6362, Loss: 0.5784
    Epoch  18 Batch 1000/2536 - Train Accuracy: 0.7875, Validation Accuracy: 0.6362, Loss: 0.6879
    Epoch  18 Batch 1100/2536 - Train Accuracy: 0.7188, Validation Accuracy: 0.6339, Loss: 0.9049
    Epoch  18 Batch 1200/2536 - Train Accuracy: 0.7793, Validation Accuracy: 0.6362, Loss: 0.6097
    Epoch  18 Batch 1300/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0434
    Epoch  18 Batch 1400/2536 - Train Accuracy: 0.7604, Validation Accuracy: 0.6339, Loss: 0.6622
    Epoch  18 Batch 1500/2536 - Train Accuracy: 0.9351, Validation Accuracy: 0.6339, Loss: 0.2577
    Epoch  18 Batch 1600/2536 - Train Accuracy: 0.9505, Validation Accuracy: 0.6362, Loss: 0.2780
    Epoch  18 Batch 1700/2536 - Train Accuracy: 0.9403, Validation Accuracy: 0.6362, Loss: 0.2448
    Epoch  18 Batch 1800/2536 - Train Accuracy: 0.9517, Validation Accuracy: 0.6339, Loss: 0.1569
    Epoch  18 Batch 1900/2536 - Train Accuracy: 0.9089, Validation Accuracy: 0.6339, Loss: 0.3020
    Epoch  18 Batch 2000/2536 - Train Accuracy: 0.8774, Validation Accuracy: 0.6339, Loss: 0.3091
    Epoch  18 Batch 2100/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6339, Loss: 0.0667
    Epoch  18 Batch 2200/2536 - Train Accuracy: 0.8667, Validation Accuracy: 0.6339, Loss: 0.3950
    Epoch  18 Batch 2300/2536 - Train Accuracy: 0.7740, Validation Accuracy: 0.6339, Loss: 0.6586
    Epoch  18 Batch 2400/2536 - Train Accuracy: 0.8654, Validation Accuracy: 0.6339, Loss: 0.4704
    Epoch  18 Batch 2500/2536 - Train Accuracy: 0.8197, Validation Accuracy: 0.6384, Loss: 0.6083
    Epoch  19 Batch  100/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.1648
    Epoch  19 Batch  200/2536 - Train Accuracy: 0.8817, Validation Accuracy: 0.6362, Loss: 0.3001
    Epoch  19 Batch  300/2536 - Train Accuracy: 0.9129, Validation Accuracy: 0.6362, Loss: 0.2291
    Epoch  19 Batch  400/2536 - Train Accuracy: 0.8555, Validation Accuracy: 0.6362, Loss: 0.3493
    Epoch  19 Batch  500/2536 - Train Accuracy: 0.8479, Validation Accuracy: 0.6362, Loss: 0.4636
    Epoch  19 Batch  600/2536 - Train Accuracy: 0.8646, Validation Accuracy: 0.6362, Loss: 0.4849
    Epoch  19 Batch  700/2536 - Train Accuracy: 0.8702, Validation Accuracy: 0.6362, Loss: 0.4771
    Epoch  19 Batch  800/2536 - Train Accuracy: 0.8486, Validation Accuracy: 0.6362, Loss: 0.4571
    Epoch  19 Batch  900/2536 - Train Accuracy: 0.7946, Validation Accuracy: 0.6339, Loss: 0.5637
    Epoch  19 Batch 1000/2536 - Train Accuracy: 0.7792, Validation Accuracy: 0.6339, Loss: 0.6447
    Epoch  19 Batch 1100/2536 - Train Accuracy: 0.7232, Validation Accuracy: 0.6362, Loss: 0.8178
    Epoch  19 Batch 1200/2536 - Train Accuracy: 0.7891, Validation Accuracy: 0.6339, Loss: 0.5549
    Epoch  19 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0304
    Epoch  19 Batch 1400/2536 - Train Accuracy: 0.7750, Validation Accuracy: 0.6339, Loss: 0.6205
    Epoch  19 Batch 1500/2536 - Train Accuracy: 0.9399, Validation Accuracy: 0.6339, Loss: 0.2251
    Epoch  19 Batch 1600/2536 - Train Accuracy: 0.9297, Validation Accuracy: 0.6362, Loss: 0.2524
    Epoch  19 Batch 1700/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6362, Loss: 0.2178
    Epoch  19 Batch 1800/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6339, Loss: 0.1248
    Epoch  19 Batch 1900/2536 - Train Accuracy: 0.9323, Validation Accuracy: 0.6339, Loss: 0.2714
    Epoch  19 Batch 2000/2536 - Train Accuracy: 0.9135, Validation Accuracy: 0.6339, Loss: 0.2738
    Epoch  19 Batch 2100/2536 - Train Accuracy: 0.9818, Validation Accuracy: 0.6339, Loss: 0.0635
    Epoch  19 Batch 2200/2536 - Train Accuracy: 0.8542, Validation Accuracy: 0.6339, Loss: 0.3653
    Epoch  19 Batch 2300/2536 - Train Accuracy: 0.7885, Validation Accuracy: 0.6339, Loss: 0.6071
    Epoch  19 Batch 2400/2536 - Train Accuracy: 0.8365, Validation Accuracy: 0.6339, Loss: 0.4268
    Epoch  19 Batch 2500/2536 - Train Accuracy: 0.8197, Validation Accuracy: 0.6362, Loss: 0.6046
    Epoch  20 Batch  100/2536 - Train Accuracy: 0.9602, Validation Accuracy: 0.6362, Loss: 0.1241
    Epoch  20 Batch  200/2536 - Train Accuracy: 0.8884, Validation Accuracy: 0.6362, Loss: 0.2882
    Epoch  20 Batch  300/2536 - Train Accuracy: 0.9286, Validation Accuracy: 0.6362, Loss: 0.2366
    Epoch  20 Batch  400/2536 - Train Accuracy: 0.8789, Validation Accuracy: 0.6362, Loss: 0.3067
    Epoch  20 Batch  500/2536 - Train Accuracy: 0.8438, Validation Accuracy: 0.6362, Loss: 0.4497
    Epoch  20 Batch  600/2536 - Train Accuracy: 0.8542, Validation Accuracy: 0.6362, Loss: 0.4238
    Epoch  20 Batch  700/2536 - Train Accuracy: 0.8726, Validation Accuracy: 0.6362, Loss: 0.4463
    Epoch  20 Batch  800/2536 - Train Accuracy: 0.8245, Validation Accuracy: 0.6384, Loss: 0.4180
    Epoch  20 Batch  900/2536 - Train Accuracy: 0.8237, Validation Accuracy: 0.6339, Loss: 0.5179
    Epoch  20 Batch 1000/2536 - Train Accuracy: 0.7958, Validation Accuracy: 0.6339, Loss: 0.6184
    Epoch  20 Batch 1100/2536 - Train Accuracy: 0.7522, Validation Accuracy: 0.6362, Loss: 0.7643
    Epoch  20 Batch 1200/2536 - Train Accuracy: 0.8027, Validation Accuracy: 0.6384, Loss: 0.5122
    Epoch  20 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0273
    Epoch  20 Batch 1400/2536 - Train Accuracy: 0.7729, Validation Accuracy: 0.6339, Loss: 0.5700
    Epoch  20 Batch 1500/2536 - Train Accuracy: 0.9423, Validation Accuracy: 0.6339, Loss: 0.1922
    Epoch  20 Batch 1600/2536 - Train Accuracy: 0.9219, Validation Accuracy: 0.6362, Loss: 0.2151
    Epoch  20 Batch 1700/2536 - Train Accuracy: 0.9545, Validation Accuracy: 0.6362, Loss: 0.1935
    Epoch  20 Batch 1800/2536 - Train Accuracy: 0.9631, Validation Accuracy: 0.6339, Loss: 0.1182
    Epoch  20 Batch 1900/2536 - Train Accuracy: 0.9167, Validation Accuracy: 0.6339, Loss: 0.2407
    Epoch  20 Batch 2000/2536 - Train Accuracy: 0.9087, Validation Accuracy: 0.6339, Loss: 0.2474
    Epoch  20 Batch 2100/2536 - Train Accuracy: 0.9870, Validation Accuracy: 0.6362, Loss: 0.0555
    Epoch  20 Batch 2200/2536 - Train Accuracy: 0.8625, Validation Accuracy: 0.6339, Loss: 0.3498
    Epoch  20 Batch 2300/2536 - Train Accuracy: 0.8125, Validation Accuracy: 0.6339, Loss: 0.5935
    Epoch  20 Batch 2400/2536 - Train Accuracy: 0.8702, Validation Accuracy: 0.6339, Loss: 0.3673
    Epoch  20 Batch 2500/2536 - Train Accuracy: 0.8365, Validation Accuracy: 0.6362, Loss: 0.5129
    Epoch  21 Batch  100/2536 - Train Accuracy: 0.9773, Validation Accuracy: 0.6339, Loss: 0.1035
    Epoch  21 Batch  200/2536 - Train Accuracy: 0.8996, Validation Accuracy: 0.6362, Loss: 0.2557
    Epoch  21 Batch  300/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6339, Loss: 0.2562
    Epoch  21 Batch  400/2536 - Train Accuracy: 0.8672, Validation Accuracy: 0.6339, Loss: 0.2883
    Epoch  21 Batch  500/2536 - Train Accuracy: 0.8542, Validation Accuracy: 0.6362, Loss: 0.3948
    Epoch  21 Batch  600/2536 - Train Accuracy: 0.8792, Validation Accuracy: 0.6362, Loss: 0.3936
    Epoch  21 Batch  700/2536 - Train Accuracy: 0.8942, Validation Accuracy: 0.6362, Loss: 0.3886
    Epoch  21 Batch  800/2536 - Train Accuracy: 0.8774, Validation Accuracy: 0.6362, Loss: 0.3857
    Epoch  21 Batch  900/2536 - Train Accuracy: 0.8259, Validation Accuracy: 0.6339, Loss: 0.4728
    Epoch  21 Batch 1000/2536 - Train Accuracy: 0.7937, Validation Accuracy: 0.6362, Loss: 0.5719
    Epoch  21 Batch 1100/2536 - Train Accuracy: 0.7344, Validation Accuracy: 0.6362, Loss: 0.6928
    Epoch  21 Batch 1200/2536 - Train Accuracy: 0.8066, Validation Accuracy: 0.6384, Loss: 0.5119
    Epoch  21 Batch 1300/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6339, Loss: 0.0319
    Epoch  21 Batch 1400/2536 - Train Accuracy: 0.8083, Validation Accuracy: 0.6339, Loss: 0.5520
    Epoch  21 Batch 1500/2536 - Train Accuracy: 0.9327, Validation Accuracy: 0.6362, Loss: 0.1959
    Epoch  21 Batch 1600/2536 - Train Accuracy: 0.9818, Validation Accuracy: 0.6362, Loss: 0.1780
    Epoch  21 Batch 1700/2536 - Train Accuracy: 0.9290, Validation Accuracy: 0.6362, Loss: 0.1937
    Epoch  21 Batch 1800/2536 - Train Accuracy: 0.9460, Validation Accuracy: 0.6362, Loss: 0.0952
    Epoch  21 Batch 1900/2536 - Train Accuracy: 0.9271, Validation Accuracy: 0.6339, Loss: 0.2328
    Epoch  21 Batch 2000/2536 - Train Accuracy: 0.8894, Validation Accuracy: 0.6339, Loss: 0.2032
    Epoch  21 Batch 2100/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6362, Loss: 0.0534
    Epoch  21 Batch 2200/2536 - Train Accuracy: 0.8667, Validation Accuracy: 0.6339, Loss: 0.3149
    Epoch  21 Batch 2300/2536 - Train Accuracy: 0.8125, Validation Accuracy: 0.6339, Loss: 0.5260
    Epoch  21 Batch 2400/2536 - Train Accuracy: 0.8654, Validation Accuracy: 0.6362, Loss: 0.3600
    Epoch  21 Batch 2500/2536 - Train Accuracy: 0.8053, Validation Accuracy: 0.6339, Loss: 0.5206
    Epoch  22 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6339, Loss: 0.0983
    Epoch  22 Batch  200/2536 - Train Accuracy: 0.9129, Validation Accuracy: 0.6339, Loss: 0.2254
    Epoch  22 Batch  300/2536 - Train Accuracy: 0.9330, Validation Accuracy: 0.6362, Loss: 0.2025
    Epoch  22 Batch  400/2536 - Train Accuracy: 0.9004, Validation Accuracy: 0.6339, Loss: 0.2660
    Epoch  22 Batch  500/2536 - Train Accuracy: 0.8313, Validation Accuracy: 0.6384, Loss: 0.3675
    Epoch  22 Batch  600/2536 - Train Accuracy: 0.8792, Validation Accuracy: 0.6362, Loss: 0.3737
    Epoch  22 Batch  700/2536 - Train Accuracy: 0.8678, Validation Accuracy: 0.6362, Loss: 0.3787
    Epoch  22 Batch  800/2536 - Train Accuracy: 0.8750, Validation Accuracy: 0.6384, Loss: 0.3751
    Epoch  22 Batch  900/2536 - Train Accuracy: 0.8504, Validation Accuracy: 0.6362, Loss: 0.4210
    Epoch  22 Batch 1000/2536 - Train Accuracy: 0.8104, Validation Accuracy: 0.6362, Loss: 0.5438
    Epoch  22 Batch 1100/2536 - Train Accuracy: 0.7254, Validation Accuracy: 0.6362, Loss: 0.7025
    Epoch  22 Batch 1200/2536 - Train Accuracy: 0.8105, Validation Accuracy: 0.6362, Loss: 0.4984
    Epoch  22 Batch 1300/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6339, Loss: 0.0359
    Epoch  22 Batch 1400/2536 - Train Accuracy: 0.7979, Validation Accuracy: 0.6339, Loss: 0.4900
    Epoch  22 Batch 1500/2536 - Train Accuracy: 0.9447, Validation Accuracy: 0.6339, Loss: 0.1706
    Epoch  22 Batch 1600/2536 - Train Accuracy: 0.9349, Validation Accuracy: 0.6362, Loss: 0.1649
    Epoch  22 Batch 1700/2536 - Train Accuracy: 0.9432, Validation Accuracy: 0.6362, Loss: 0.1709
    Epoch  22 Batch 1800/2536 - Train Accuracy: 0.9489, Validation Accuracy: 0.6362, Loss: 0.0927
    Epoch  22 Batch 1900/2536 - Train Accuracy: 0.9349, Validation Accuracy: 0.6339, Loss: 0.2143
    Epoch  22 Batch 2000/2536 - Train Accuracy: 0.8870, Validation Accuracy: 0.6339, Loss: 0.2172
    Epoch  22 Batch 2100/2536 - Train Accuracy: 0.9740, Validation Accuracy: 0.6384, Loss: 0.0404
    Epoch  22 Batch 2200/2536 - Train Accuracy: 0.9000, Validation Accuracy: 0.6339, Loss: 0.2709
    Epoch  22 Batch 2300/2536 - Train Accuracy: 0.7764, Validation Accuracy: 0.6362, Loss: 0.4483
    Epoch  22 Batch 2400/2536 - Train Accuracy: 0.8894, Validation Accuracy: 0.6406, Loss: 0.3420
    Epoch  22 Batch 2500/2536 - Train Accuracy: 0.8726, Validation Accuracy: 0.6339, Loss: 0.4742
    Epoch  23 Batch  100/2536 - Train Accuracy: 0.9773, Validation Accuracy: 0.6362, Loss: 0.0945
    Epoch  23 Batch  200/2536 - Train Accuracy: 0.9152, Validation Accuracy: 0.6362, Loss: 0.2095
    Epoch  23 Batch  300/2536 - Train Accuracy: 0.9330, Validation Accuracy: 0.6384, Loss: 0.1904
    Epoch  23 Batch  400/2536 - Train Accuracy: 0.9102, Validation Accuracy: 0.6362, Loss: 0.2922
    Epoch  23 Batch  500/2536 - Train Accuracy: 0.9042, Validation Accuracy: 0.6384, Loss: 0.3396
    Epoch  23 Batch  600/2536 - Train Accuracy: 0.8917, Validation Accuracy: 0.6362, Loss: 0.3648
    Epoch  23 Batch  700/2536 - Train Accuracy: 0.8702, Validation Accuracy: 0.6362, Loss: 0.3564
    Epoch  23 Batch  800/2536 - Train Accuracy: 0.8846, Validation Accuracy: 0.6384, Loss: 0.3483
    Epoch  23 Batch  900/2536 - Train Accuracy: 0.8527, Validation Accuracy: 0.6362, Loss: 0.4382
    Epoch  23 Batch 1000/2536 - Train Accuracy: 0.8250, Validation Accuracy: 0.6362, Loss: 0.5510
    Epoch  23 Batch 1100/2536 - Train Accuracy: 0.7701, Validation Accuracy: 0.6362, Loss: 0.6347
    Epoch  23 Batch 1200/2536 - Train Accuracy: 0.8340, Validation Accuracy: 0.6362, Loss: 0.4687
    Epoch  23 Batch 1300/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6339, Loss: 0.0384
    Epoch  23 Batch 1400/2536 - Train Accuracy: 0.8125, Validation Accuracy: 0.6339, Loss: 0.4695
    Epoch  23 Batch 1500/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6339, Loss: 0.1862
    Epoch  23 Batch 1600/2536 - Train Accuracy: 0.9505, Validation Accuracy: 0.6362, Loss: 0.1802
    Epoch  23 Batch 1700/2536 - Train Accuracy: 0.9574, Validation Accuracy: 0.6362, Loss: 0.1416
    Epoch  23 Batch 1800/2536 - Train Accuracy: 0.9659, Validation Accuracy: 0.6362, Loss: 0.0912
    Epoch  23 Batch 1900/2536 - Train Accuracy: 0.9219, Validation Accuracy: 0.6362, Loss: 0.1960
    Epoch  23 Batch 2000/2536 - Train Accuracy: 0.9615, Validation Accuracy: 0.6362, Loss: 0.1724
    Epoch  23 Batch 2100/2536 - Train Accuracy: 0.9896, Validation Accuracy: 0.6362, Loss: 0.0416
    Epoch  23 Batch 2200/2536 - Train Accuracy: 0.8833, Validation Accuracy: 0.6362, Loss: 0.2783
    Epoch  23 Batch 2300/2536 - Train Accuracy: 0.8293, Validation Accuracy: 0.6362, Loss: 0.4258
    Epoch  23 Batch 2400/2536 - Train Accuracy: 0.8894, Validation Accuracy: 0.6339, Loss: 0.3163
    Epoch  23 Batch 2500/2536 - Train Accuracy: 0.8510, Validation Accuracy: 0.6339, Loss: 0.4227
    Epoch  24 Batch  100/2536 - Train Accuracy: 0.9773, Validation Accuracy: 0.6339, Loss: 0.0907
    Epoch  24 Batch  200/2536 - Train Accuracy: 0.9174, Validation Accuracy: 0.6339, Loss: 0.2172
    Epoch  24 Batch  300/2536 - Train Accuracy: 0.9554, Validation Accuracy: 0.6339, Loss: 0.1992
    Epoch  24 Batch  400/2536 - Train Accuracy: 0.8789, Validation Accuracy: 0.6339, Loss: 0.2362
    Epoch  24 Batch  500/2536 - Train Accuracy: 0.8938, Validation Accuracy: 0.6339, Loss: 0.3240
    Epoch  24 Batch  600/2536 - Train Accuracy: 0.8875, Validation Accuracy: 0.6384, Loss: 0.3643
    Epoch  24 Batch  700/2536 - Train Accuracy: 0.8846, Validation Accuracy: 0.6339, Loss: 0.3287
    Epoch  24 Batch  800/2536 - Train Accuracy: 0.9231, Validation Accuracy: 0.6362, Loss: 0.3264
    Epoch  24 Batch  900/2536 - Train Accuracy: 0.8750, Validation Accuracy: 0.6362, Loss: 0.3737
    Epoch  24 Batch 1000/2536 - Train Accuracy: 0.8083, Validation Accuracy: 0.6362, Loss: 0.4653
    Epoch  24 Batch 1100/2536 - Train Accuracy: 0.7679, Validation Accuracy: 0.6362, Loss: 0.5902
    Epoch  24 Batch 1200/2536 - Train Accuracy: 0.7910, Validation Accuracy: 0.6362, Loss: 0.4493
    Epoch  24 Batch 1300/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6339, Loss: 0.0238
    Epoch  24 Batch 1400/2536 - Train Accuracy: 0.8146, Validation Accuracy: 0.6339, Loss: 0.4508
    Epoch  24 Batch 1500/2536 - Train Accuracy: 0.9207, Validation Accuracy: 0.6362, Loss: 0.1464
    Epoch  24 Batch 1600/2536 - Train Accuracy: 0.9635, Validation Accuracy: 0.6362, Loss: 0.1277
    Epoch  24 Batch 1700/2536 - Train Accuracy: 0.9545, Validation Accuracy: 0.6384, Loss: 0.1311
    Epoch  24 Batch 1800/2536 - Train Accuracy: 0.9574, Validation Accuracy: 0.6362, Loss: 0.0859
    Epoch  24 Batch 1900/2536 - Train Accuracy: 0.9167, Validation Accuracy: 0.6362, Loss: 0.1798
    Epoch  24 Batch 2000/2536 - Train Accuracy: 0.9062, Validation Accuracy: 0.6384, Loss: 0.1601
    Epoch  24 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0293
    Epoch  24 Batch 2200/2536 - Train Accuracy: 0.9208, Validation Accuracy: 0.6362, Loss: 0.2155
    Epoch  24 Batch 2300/2536 - Train Accuracy: 0.8101, Validation Accuracy: 0.6339, Loss: 0.3634
    Epoch  24 Batch 2400/2536 - Train Accuracy: 0.8918, Validation Accuracy: 0.6362, Loss: 0.2705
    Epoch  24 Batch 2500/2536 - Train Accuracy: 0.8558, Validation Accuracy: 0.6362, Loss: 0.3952
    Epoch  25 Batch  100/2536 - Train Accuracy: 0.9830, Validation Accuracy: 0.6362, Loss: 0.0878
    Epoch  25 Batch  200/2536 - Train Accuracy: 0.9286, Validation Accuracy: 0.6339, Loss: 0.1659
    Epoch  25 Batch  300/2536 - Train Accuracy: 0.9464, Validation Accuracy: 0.6384, Loss: 0.1916
    Epoch  25 Batch  400/2536 - Train Accuracy: 0.9238, Validation Accuracy: 0.6362, Loss: 0.2344
    Epoch  25 Batch  500/2536 - Train Accuracy: 0.8896, Validation Accuracy: 0.6362, Loss: 0.3138
    Epoch  25 Batch  600/2536 - Train Accuracy: 0.8979, Validation Accuracy: 0.6384, Loss: 0.3258
    Epoch  25 Batch  700/2536 - Train Accuracy: 0.9135, Validation Accuracy: 0.6384, Loss: 0.2833
    Epoch  25 Batch  800/2536 - Train Accuracy: 0.8918, Validation Accuracy: 0.6362, Loss: 0.2979
    Epoch  25 Batch  900/2536 - Train Accuracy: 0.8728, Validation Accuracy: 0.6384, Loss: 0.3495
    Epoch  25 Batch 1000/2536 - Train Accuracy: 0.8063, Validation Accuracy: 0.6362, Loss: 0.4942
    Epoch  25 Batch 1100/2536 - Train Accuracy: 0.7835, Validation Accuracy: 0.6362, Loss: 0.6128
    Epoch  25 Batch 1200/2536 - Train Accuracy: 0.8379, Validation Accuracy: 0.6362, Loss: 0.3892
    Epoch  25 Batch 1300/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6339, Loss: 0.0223
    Epoch  25 Batch 1400/2536 - Train Accuracy: 0.7896, Validation Accuracy: 0.6339, Loss: 0.3933
    Epoch  25 Batch 1500/2536 - Train Accuracy: 0.9255, Validation Accuracy: 0.6362, Loss: 0.1360
    Epoch  25 Batch 1600/2536 - Train Accuracy: 0.9818, Validation Accuracy: 0.6384, Loss: 0.1294
    Epoch  25 Batch 1700/2536 - Train Accuracy: 0.9716, Validation Accuracy: 0.6362, Loss: 0.1159
    Epoch  25 Batch 1800/2536 - Train Accuracy: 0.9744, Validation Accuracy: 0.6362, Loss: 0.0773
    Epoch  25 Batch 1900/2536 - Train Accuracy: 0.9401, Validation Accuracy: 0.6362, Loss: 0.1629
    Epoch  25 Batch 2000/2536 - Train Accuracy: 0.9543, Validation Accuracy: 0.6339, Loss: 0.1445
    Epoch  25 Batch 2100/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6362, Loss: 0.0351
    Epoch  25 Batch 2200/2536 - Train Accuracy: 0.9313, Validation Accuracy: 0.6362, Loss: 0.2226
    Epoch  25 Batch 2300/2536 - Train Accuracy: 0.8341, Validation Accuracy: 0.6362, Loss: 0.3857
    Epoch  25 Batch 2400/2536 - Train Accuracy: 0.9038, Validation Accuracy: 0.6339, Loss: 0.2502
    Epoch  25 Batch 2500/2536 - Train Accuracy: 0.8894, Validation Accuracy: 0.6362, Loss: 0.3767
    Epoch  26 Batch  100/2536 - Train Accuracy: 0.9858, Validation Accuracy: 0.6339, Loss: 0.0796
    Epoch  26 Batch  200/2536 - Train Accuracy: 0.9420, Validation Accuracy: 0.6339, Loss: 0.1814
    Epoch  26 Batch  300/2536 - Train Accuracy: 0.9420, Validation Accuracy: 0.6339, Loss: 0.1684
    Epoch  26 Batch  400/2536 - Train Accuracy: 0.9062, Validation Accuracy: 0.6339, Loss: 0.2167
    Epoch  26 Batch  500/2536 - Train Accuracy: 0.8958, Validation Accuracy: 0.6339, Loss: 0.2380
    Epoch  26 Batch  600/2536 - Train Accuracy: 0.9083, Validation Accuracy: 0.6362, Loss: 0.3009
    Epoch  26 Batch  700/2536 - Train Accuracy: 0.9255, Validation Accuracy: 0.6339, Loss: 0.2728
    Epoch  26 Batch  800/2536 - Train Accuracy: 0.8942, Validation Accuracy: 0.6362, Loss: 0.2928
    Epoch  26 Batch  900/2536 - Train Accuracy: 0.8951, Validation Accuracy: 0.6339, Loss: 0.3423
    Epoch  26 Batch 1000/2536 - Train Accuracy: 0.8187, Validation Accuracy: 0.6339, Loss: 0.4683
    Epoch  26 Batch 1100/2536 - Train Accuracy: 0.7567, Validation Accuracy: 0.6339, Loss: 0.5089
    Epoch  26 Batch 1200/2536 - Train Accuracy: 0.8125, Validation Accuracy: 0.6362, Loss: 0.3779
    Epoch  26 Batch 1300/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6339, Loss: 0.0282
    Epoch  26 Batch 1400/2536 - Train Accuracy: 0.8375, Validation Accuracy: 0.6339, Loss: 0.3917
    Epoch  26 Batch 1500/2536 - Train Accuracy: 0.9639, Validation Accuracy: 0.6384, Loss: 0.1221
    Epoch  26 Batch 1600/2536 - Train Accuracy: 0.9740, Validation Accuracy: 0.6362, Loss: 0.1007
    Epoch  26 Batch 1700/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.1153
    Epoch  26 Batch 1800/2536 - Train Accuracy: 0.9602, Validation Accuracy: 0.6339, Loss: 0.0797
    Epoch  26 Batch 1900/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6339, Loss: 0.1792
    Epoch  26 Batch 2000/2536 - Train Accuracy: 0.9447, Validation Accuracy: 0.6339, Loss: 0.1251
    Epoch  26 Batch 2100/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6362, Loss: 0.0326
    Epoch  26 Batch 2200/2536 - Train Accuracy: 0.9250, Validation Accuracy: 0.6339, Loss: 0.1942
    Epoch  26 Batch 2300/2536 - Train Accuracy: 0.8534, Validation Accuracy: 0.6339, Loss: 0.3456
    Epoch  26 Batch 2400/2536 - Train Accuracy: 0.9038, Validation Accuracy: 0.6339, Loss: 0.2577
    Epoch  26 Batch 2500/2536 - Train Accuracy: 0.9014, Validation Accuracy: 0.6339, Loss: 0.3213
    Epoch  27 Batch  100/2536 - Train Accuracy: 0.9659, Validation Accuracy: 0.6339, Loss: 0.1010
    Epoch  27 Batch  200/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6339, Loss: 0.1481
    Epoch  27 Batch  300/2536 - Train Accuracy: 0.9554, Validation Accuracy: 0.6362, Loss: 0.1383
    Epoch  27 Batch  400/2536 - Train Accuracy: 0.9180, Validation Accuracy: 0.6339, Loss: 0.1923
    Epoch  27 Batch  500/2536 - Train Accuracy: 0.9021, Validation Accuracy: 0.6362, Loss: 0.2519
    Epoch  27 Batch  600/2536 - Train Accuracy: 0.9146, Validation Accuracy: 0.6362, Loss: 0.3081
    Epoch  27 Batch  700/2536 - Train Accuracy: 0.9279, Validation Accuracy: 0.6406, Loss: 0.2446
    Epoch  27 Batch  800/2536 - Train Accuracy: 0.9159, Validation Accuracy: 0.6362, Loss: 0.2465
    Epoch  27 Batch  900/2536 - Train Accuracy: 0.8705, Validation Accuracy: 0.6362, Loss: 0.3106
    Epoch  27 Batch 1000/2536 - Train Accuracy: 0.8625, Validation Accuracy: 0.6339, Loss: 0.4086
    Epoch  27 Batch 1100/2536 - Train Accuracy: 0.8192, Validation Accuracy: 0.6339, Loss: 0.4936
    Epoch  27 Batch 1200/2536 - Train Accuracy: 0.8320, Validation Accuracy: 0.6339, Loss: 0.3565
    Epoch  27 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0159
    Epoch  27 Batch 1400/2536 - Train Accuracy: 0.8708, Validation Accuracy: 0.6339, Loss: 0.3678
    Epoch  27 Batch 1500/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6362, Loss: 0.0982
    Epoch  27 Batch 1600/2536 - Train Accuracy: 0.9896, Validation Accuracy: 0.6384, Loss: 0.0850
    Epoch  27 Batch 1700/2536 - Train Accuracy: 0.9489, Validation Accuracy: 0.6384, Loss: 0.1018
    Epoch  27 Batch 1800/2536 - Train Accuracy: 0.9716, Validation Accuracy: 0.6362, Loss: 0.0816
    Epoch  27 Batch 1900/2536 - Train Accuracy: 0.9245, Validation Accuracy: 0.6362, Loss: 0.1720
    Epoch  27 Batch 2000/2536 - Train Accuracy: 0.9183, Validation Accuracy: 0.6384, Loss: 0.1204
    Epoch  27 Batch 2100/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6384, Loss: 0.0219
    Epoch  27 Batch 2200/2536 - Train Accuracy: 0.9125, Validation Accuracy: 0.6362, Loss: 0.1905
    Epoch  27 Batch 2300/2536 - Train Accuracy: 0.8726, Validation Accuracy: 0.6339, Loss: 0.2907
    Epoch  27 Batch 2400/2536 - Train Accuracy: 0.8918, Validation Accuracy: 0.6362, Loss: 0.2288
    Epoch  27 Batch 2500/2536 - Train Accuracy: 0.8606, Validation Accuracy: 0.6362, Loss: 0.2789
    Epoch  28 Batch  100/2536 - Train Accuracy: 0.9858, Validation Accuracy: 0.6362, Loss: 0.0751
    Epoch  28 Batch  200/2536 - Train Accuracy: 0.9241, Validation Accuracy: 0.6362, Loss: 0.1421
    Epoch  28 Batch  300/2536 - Train Accuracy: 0.9442, Validation Accuracy: 0.6362, Loss: 0.1320
    Epoch  28 Batch  400/2536 - Train Accuracy: 0.9180, Validation Accuracy: 0.6362, Loss: 0.1787
    Epoch  28 Batch  500/2536 - Train Accuracy: 0.8979, Validation Accuracy: 0.6384, Loss: 0.2251
    Epoch  28 Batch  600/2536 - Train Accuracy: 0.8979, Validation Accuracy: 0.6362, Loss: 0.2653
    Epoch  28 Batch  700/2536 - Train Accuracy: 0.9207, Validation Accuracy: 0.6362, Loss: 0.2183
    Epoch  28 Batch  800/2536 - Train Accuracy: 0.9231, Validation Accuracy: 0.6384, Loss: 0.2425
    Epoch  28 Batch  900/2536 - Train Accuracy: 0.9085, Validation Accuracy: 0.6362, Loss: 0.2848
    Epoch  28 Batch 1000/2536 - Train Accuracy: 0.8542, Validation Accuracy: 0.6339, Loss: 0.4033
    Epoch  28 Batch 1100/2536 - Train Accuracy: 0.7924, Validation Accuracy: 0.6339, Loss: 0.4727
    Epoch  28 Batch 1200/2536 - Train Accuracy: 0.8594, Validation Accuracy: 0.6362, Loss: 0.3190
    Epoch  28 Batch 1300/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6339, Loss: 0.0203
    Epoch  28 Batch 1400/2536 - Train Accuracy: 0.8417, Validation Accuracy: 0.6339, Loss: 0.3628
    Epoch  28 Batch 1500/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6362, Loss: 0.0910
    Epoch  28 Batch 1600/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6362, Loss: 0.0810
    Epoch  28 Batch 1700/2536 - Train Accuracy: 0.9801, Validation Accuracy: 0.6362, Loss: 0.0898
    Epoch  28 Batch 1800/2536 - Train Accuracy: 0.9801, Validation Accuracy: 0.6362, Loss: 0.0717
    Epoch  28 Batch 1900/2536 - Train Accuracy: 0.9349, Validation Accuracy: 0.6362, Loss: 0.1655
    Epoch  28 Batch 2000/2536 - Train Accuracy: 0.9615, Validation Accuracy: 0.6362, Loss: 0.1056
    Epoch  28 Batch 2100/2536 - Train Accuracy: 0.9818, Validation Accuracy: 0.6384, Loss: 0.0216
    Epoch  28 Batch 2200/2536 - Train Accuracy: 0.9250, Validation Accuracy: 0.6339, Loss: 0.2086
    Epoch  28 Batch 2300/2536 - Train Accuracy: 0.8846, Validation Accuracy: 0.6339, Loss: 0.2880
    Epoch  28 Batch 2400/2536 - Train Accuracy: 0.9279, Validation Accuracy: 0.6362, Loss: 0.2281
    Epoch  28 Batch 2500/2536 - Train Accuracy: 0.8726, Validation Accuracy: 0.6362, Loss: 0.2890
    Epoch  29 Batch  100/2536 - Train Accuracy: 0.9886, Validation Accuracy: 0.6339, Loss: 0.0585
    Epoch  29 Batch  200/2536 - Train Accuracy: 0.9643, Validation Accuracy: 0.6339, Loss: 0.1379
    Epoch  29 Batch  300/2536 - Train Accuracy: 0.9621, Validation Accuracy: 0.6362, Loss: 0.1594
    Epoch  29 Batch  400/2536 - Train Accuracy: 0.9043, Validation Accuracy: 0.6384, Loss: 0.1707
    Epoch  29 Batch  500/2536 - Train Accuracy: 0.9062, Validation Accuracy: 0.6406, Loss: 0.2288
    Epoch  29 Batch  600/2536 - Train Accuracy: 0.9229, Validation Accuracy: 0.6362, Loss: 0.2572
    Epoch  29 Batch  700/2536 - Train Accuracy: 0.9062, Validation Accuracy: 0.6384, Loss: 0.2130
    Epoch  29 Batch  800/2536 - Train Accuracy: 0.8942, Validation Accuracy: 0.6384, Loss: 0.2004
    Epoch  29 Batch  900/2536 - Train Accuracy: 0.9062, Validation Accuracy: 0.6384, Loss: 0.2715
    Epoch  29 Batch 1000/2536 - Train Accuracy: 0.8667, Validation Accuracy: 0.6362, Loss: 0.3835
    Epoch  29 Batch 1100/2536 - Train Accuracy: 0.8036, Validation Accuracy: 0.6362, Loss: 0.4370
    Epoch  29 Batch 1200/2536 - Train Accuracy: 0.8340, Validation Accuracy: 0.6362, Loss: 0.2799
    Epoch  29 Batch 1300/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6339, Loss: 0.0238
    Epoch  29 Batch 1400/2536 - Train Accuracy: 0.8500, Validation Accuracy: 0.6362, Loss: 0.3299
    Epoch  29 Batch 1500/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6406, Loss: 0.0856
    Epoch  29 Batch 1600/2536 - Train Accuracy: 0.9766, Validation Accuracy: 0.6384, Loss: 0.0888
    Epoch  29 Batch 1700/2536 - Train Accuracy: 0.9602, Validation Accuracy: 0.6384, Loss: 0.0977
    Epoch  29 Batch 1800/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6362, Loss: 0.0684
    Epoch  29 Batch 1900/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.1537
    Epoch  29 Batch 2000/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6384, Loss: 0.1134
    Epoch  29 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0300
    Epoch  29 Batch 2200/2536 - Train Accuracy: 0.9229, Validation Accuracy: 0.6362, Loss: 0.1668
    Epoch  29 Batch 2300/2536 - Train Accuracy: 0.9038, Validation Accuracy: 0.6339, Loss: 0.2948
    Epoch  29 Batch 2400/2536 - Train Accuracy: 0.9351, Validation Accuracy: 0.6384, Loss: 0.1857
    Epoch  29 Batch 2500/2536 - Train Accuracy: 0.8942, Validation Accuracy: 0.6384, Loss: 0.2343
    Epoch  30 Batch  100/2536 - Train Accuracy: 0.9858, Validation Accuracy: 0.6384, Loss: 0.0564
    Epoch  30 Batch  200/2536 - Train Accuracy: 0.9464, Validation Accuracy: 0.6384, Loss: 0.1034
    Epoch  30 Batch  300/2536 - Train Accuracy: 0.9353, Validation Accuracy: 0.6384, Loss: 0.1607
    Epoch  30 Batch  400/2536 - Train Accuracy: 0.9277, Validation Accuracy: 0.6384, Loss: 0.1622
    Epoch  30 Batch  500/2536 - Train Accuracy: 0.9062, Validation Accuracy: 0.6384, Loss: 0.2097
    Epoch  30 Batch  600/2536 - Train Accuracy: 0.9271, Validation Accuracy: 0.6384, Loss: 0.2409
    Epoch  30 Batch  700/2536 - Train Accuracy: 0.9399, Validation Accuracy: 0.6384, Loss: 0.2225
    Epoch  30 Batch  800/2536 - Train Accuracy: 0.9207, Validation Accuracy: 0.6384, Loss: 0.2181
    Epoch  30 Batch  900/2536 - Train Accuracy: 0.8817, Validation Accuracy: 0.6362, Loss: 0.2531
    Epoch  30 Batch 1000/2536 - Train Accuracy: 0.8250, Validation Accuracy: 0.6362, Loss: 0.3637
    Epoch  30 Batch 1100/2536 - Train Accuracy: 0.8125, Validation Accuracy: 0.6362, Loss: 0.3783
    Epoch  30 Batch 1200/2536 - Train Accuracy: 0.8750, Validation Accuracy: 0.6384, Loss: 0.3113
    Epoch  30 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0168
    Epoch  30 Batch 1400/2536 - Train Accuracy: 0.8479, Validation Accuracy: 0.6362, Loss: 0.2978
    Epoch  30 Batch 1500/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6339, Loss: 0.0800
    Epoch  30 Batch 1600/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0828
    Epoch  30 Batch 1700/2536 - Train Accuracy: 0.9716, Validation Accuracy: 0.6384, Loss: 0.0838
    Epoch  30 Batch 1800/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6406, Loss: 0.0665
    Epoch  30 Batch 1900/2536 - Train Accuracy: 0.9531, Validation Accuracy: 0.6362, Loss: 0.1200
    Epoch  30 Batch 2000/2536 - Train Accuracy: 0.9447, Validation Accuracy: 0.6362, Loss: 0.1042
    Epoch  30 Batch 2100/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6406, Loss: 0.0291
    Epoch  30 Batch 2200/2536 - Train Accuracy: 0.9333, Validation Accuracy: 0.6406, Loss: 0.1868
    Epoch  30 Batch 2300/2536 - Train Accuracy: 0.8966, Validation Accuracy: 0.6362, Loss: 0.2401
    Epoch  30 Batch 2400/2536 - Train Accuracy: 0.9327, Validation Accuracy: 0.6384, Loss: 0.1999
    Epoch  30 Batch 2500/2536 - Train Accuracy: 0.8966, Validation Accuracy: 0.6406, Loss: 0.2520
    Epoch  31 Batch  100/2536 - Train Accuracy: 0.9716, Validation Accuracy: 0.6406, Loss: 0.0709
    Epoch  31 Batch  200/2536 - Train Accuracy: 0.9509, Validation Accuracy: 0.6384, Loss: 0.0875
    Epoch  31 Batch  300/2536 - Train Accuracy: 0.9420, Validation Accuracy: 0.6384, Loss: 0.1173
    Epoch  31 Batch  400/2536 - Train Accuracy: 0.9434, Validation Accuracy: 0.6362, Loss: 0.1640
    Epoch  31 Batch  500/2536 - Train Accuracy: 0.9104, Validation Accuracy: 0.6384, Loss: 0.2176
    Epoch  31 Batch  600/2536 - Train Accuracy: 0.9479, Validation Accuracy: 0.6384, Loss: 0.2243
    Epoch  31 Batch  700/2536 - Train Accuracy: 0.8966, Validation Accuracy: 0.6362, Loss: 0.1931
    Epoch  31 Batch  800/2536 - Train Accuracy: 0.9111, Validation Accuracy: 0.6384, Loss: 0.2261
    Epoch  31 Batch  900/2536 - Train Accuracy: 0.9129, Validation Accuracy: 0.6362, Loss: 0.2199
    Epoch  31 Batch 1000/2536 - Train Accuracy: 0.8521, Validation Accuracy: 0.6384, Loss: 0.3585
    Epoch  31 Batch 1100/2536 - Train Accuracy: 0.8728, Validation Accuracy: 0.6384, Loss: 0.3762
    Epoch  31 Batch 1200/2536 - Train Accuracy: 0.8535, Validation Accuracy: 0.6384, Loss: 0.2674
    Epoch  31 Batch 1300/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6362, Loss: 0.0272
    Epoch  31 Batch 1400/2536 - Train Accuracy: 0.8917, Validation Accuracy: 0.6362, Loss: 0.2443
    Epoch  31 Batch 1500/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0883
    Epoch  31 Batch 1600/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6406, Loss: 0.0783
    Epoch  31 Batch 1700/2536 - Train Accuracy: 0.9517, Validation Accuracy: 0.6384, Loss: 0.0917
    Epoch  31 Batch 1800/2536 - Train Accuracy: 0.9773, Validation Accuracy: 0.6362, Loss: 0.0516
    Epoch  31 Batch 1900/2536 - Train Accuracy: 0.9453, Validation Accuracy: 0.6362, Loss: 0.1277
    Epoch  31 Batch 2000/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6384, Loss: 0.0945
    Epoch  31 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0221
    Epoch  31 Batch 2200/2536 - Train Accuracy: 0.9333, Validation Accuracy: 0.6362, Loss: 0.1685
    Epoch  31 Batch 2300/2536 - Train Accuracy: 0.8702, Validation Accuracy: 0.6384, Loss: 0.2255
    Epoch  31 Batch 2400/2536 - Train Accuracy: 0.9111, Validation Accuracy: 0.6384, Loss: 0.1735
    Epoch  31 Batch 2500/2536 - Train Accuracy: 0.9135, Validation Accuracy: 0.6384, Loss: 0.2086
    Epoch  32 Batch  100/2536 - Train Accuracy: 0.9801, Validation Accuracy: 0.6384, Loss: 0.0574
    Epoch  32 Batch  200/2536 - Train Accuracy: 0.9286, Validation Accuracy: 0.6384, Loss: 0.1084
    Epoch  32 Batch  300/2536 - Train Accuracy: 0.9665, Validation Accuracy: 0.6384, Loss: 0.1222
    Epoch  32 Batch  400/2536 - Train Accuracy: 0.9316, Validation Accuracy: 0.6384, Loss: 0.1531
    Epoch  32 Batch  500/2536 - Train Accuracy: 0.9187, Validation Accuracy: 0.6384, Loss: 0.1827
    Epoch  32 Batch  600/2536 - Train Accuracy: 0.9250, Validation Accuracy: 0.6384, Loss: 0.2324
    Epoch  32 Batch  700/2536 - Train Accuracy: 0.9279, Validation Accuracy: 0.6384, Loss: 0.1886
    Epoch  32 Batch  800/2536 - Train Accuracy: 0.9279, Validation Accuracy: 0.6406, Loss: 0.1827
    Epoch  32 Batch  900/2536 - Train Accuracy: 0.9196, Validation Accuracy: 0.6406, Loss: 0.2163
    Epoch  32 Batch 1000/2536 - Train Accuracy: 0.8292, Validation Accuracy: 0.6384, Loss: 0.3601
    Epoch  32 Batch 1100/2536 - Train Accuracy: 0.8304, Validation Accuracy: 0.6384, Loss: 0.3886
    Epoch  32 Batch 1200/2536 - Train Accuracy: 0.8418, Validation Accuracy: 0.6406, Loss: 0.2619
    Epoch  32 Batch 1300/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6384, Loss: 0.0152
    Epoch  32 Batch 1400/2536 - Train Accuracy: 0.8875, Validation Accuracy: 0.6384, Loss: 0.2715
    Epoch  32 Batch 1500/2536 - Train Accuracy: 0.9639, Validation Accuracy: 0.6362, Loss: 0.0634
    Epoch  32 Batch 1600/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6384, Loss: 0.0700
    Epoch  32 Batch 1700/2536 - Train Accuracy: 0.9801, Validation Accuracy: 0.6362, Loss: 0.0839
    Epoch  32 Batch 1800/2536 - Train Accuracy: 0.9744, Validation Accuracy: 0.6384, Loss: 0.0643
    Epoch  32 Batch 1900/2536 - Train Accuracy: 0.9557, Validation Accuracy: 0.6362, Loss: 0.1220
    Epoch  32 Batch 2000/2536 - Train Accuracy: 0.9471, Validation Accuracy: 0.6384, Loss: 0.1176
    Epoch  32 Batch 2100/2536 - Train Accuracy: 0.9740, Validation Accuracy: 0.6406, Loss: 0.0206
    Epoch  32 Batch 2200/2536 - Train Accuracy: 0.9250, Validation Accuracy: 0.6384, Loss: 0.1574
    Epoch  32 Batch 2300/2536 - Train Accuracy: 0.8870, Validation Accuracy: 0.6362, Loss: 0.2216
    Epoch  32 Batch 2400/2536 - Train Accuracy: 0.9495, Validation Accuracy: 0.6362, Loss: 0.1643
    Epoch  32 Batch 2500/2536 - Train Accuracy: 0.9399, Validation Accuracy: 0.6384, Loss: 0.2335
    Epoch  33 Batch  100/2536 - Train Accuracy: 0.9801, Validation Accuracy: 0.6362, Loss: 0.0547
    Epoch  33 Batch  200/2536 - Train Accuracy: 0.9598, Validation Accuracy: 0.6362, Loss: 0.0986
    Epoch  33 Batch  300/2536 - Train Accuracy: 0.9397, Validation Accuracy: 0.6384, Loss: 0.1237
    Epoch  33 Batch  400/2536 - Train Accuracy: 0.9141, Validation Accuracy: 0.6384, Loss: 0.1752
    Epoch  33 Batch  500/2536 - Train Accuracy: 0.9250, Validation Accuracy: 0.6384, Loss: 0.1813
    Epoch  33 Batch  600/2536 - Train Accuracy: 0.9250, Validation Accuracy: 0.6384, Loss: 0.2453
    Epoch  33 Batch  700/2536 - Train Accuracy: 0.9543, Validation Accuracy: 0.6384, Loss: 0.1606
    Epoch  33 Batch  800/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6384, Loss: 0.1672
    Epoch  33 Batch  900/2536 - Train Accuracy: 0.8973, Validation Accuracy: 0.6384, Loss: 0.2542
    Epoch  33 Batch 1000/2536 - Train Accuracy: 0.8479, Validation Accuracy: 0.6384, Loss: 0.3713
    Epoch  33 Batch 1100/2536 - Train Accuracy: 0.8281, Validation Accuracy: 0.6384, Loss: 0.3700
    Epoch  33 Batch 1200/2536 - Train Accuracy: 0.8555, Validation Accuracy: 0.6384, Loss: 0.2564
    Epoch  33 Batch 1300/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0184
    Epoch  33 Batch 1400/2536 - Train Accuracy: 0.8896, Validation Accuracy: 0.6384, Loss: 0.2508
    Epoch  33 Batch 1500/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6362, Loss: 0.0718
    Epoch  33 Batch 1600/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6384, Loss: 0.0592
    Epoch  33 Batch 1700/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0621
    Epoch  33 Batch 1800/2536 - Train Accuracy: 0.9801, Validation Accuracy: 0.6384, Loss: 0.0394
    Epoch  33 Batch 1900/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.1125
    Epoch  33 Batch 2000/2536 - Train Accuracy: 0.9712, Validation Accuracy: 0.6384, Loss: 0.0930
    Epoch  33 Batch 2100/2536 - Train Accuracy: 0.9870, Validation Accuracy: 0.6384, Loss: 0.0246
    Epoch  33 Batch 2200/2536 - Train Accuracy: 0.9667, Validation Accuracy: 0.6362, Loss: 0.0983
    Epoch  33 Batch 2300/2536 - Train Accuracy: 0.8822, Validation Accuracy: 0.6362, Loss: 0.2235
    Epoch  33 Batch 2400/2536 - Train Accuracy: 0.9712, Validation Accuracy: 0.6362, Loss: 0.1399
    Epoch  33 Batch 2500/2536 - Train Accuracy: 0.9327, Validation Accuracy: 0.6384, Loss: 0.1903
    Epoch  34 Batch  100/2536 - Train Accuracy: 0.9830, Validation Accuracy: 0.6384, Loss: 0.0582
    Epoch  34 Batch  200/2536 - Train Accuracy: 0.9710, Validation Accuracy: 0.6384, Loss: 0.0979
    Epoch  34 Batch  300/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.1185
    Epoch  34 Batch  400/2536 - Train Accuracy: 0.9590, Validation Accuracy: 0.6362, Loss: 0.1413
    Epoch  34 Batch  500/2536 - Train Accuracy: 0.9083, Validation Accuracy: 0.6384, Loss: 0.1693
    Epoch  34 Batch  600/2536 - Train Accuracy: 0.9313, Validation Accuracy: 0.6362, Loss: 0.2050
    Epoch  34 Batch  700/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6384, Loss: 0.1353
    Epoch  34 Batch  800/2536 - Train Accuracy: 0.9303, Validation Accuracy: 0.6384, Loss: 0.1364
    Epoch  34 Batch  900/2536 - Train Accuracy: 0.9196, Validation Accuracy: 0.6384, Loss: 0.2028
    Epoch  34 Batch 1000/2536 - Train Accuracy: 0.8812, Validation Accuracy: 0.6384, Loss: 0.3271
    Epoch  34 Batch 1100/2536 - Train Accuracy: 0.8661, Validation Accuracy: 0.6384, Loss: 0.3311
    Epoch  34 Batch 1200/2536 - Train Accuracy: 0.8418, Validation Accuracy: 0.6406, Loss: 0.2662
    Epoch  34 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0098
    Epoch  34 Batch 1400/2536 - Train Accuracy: 0.8896, Validation Accuracy: 0.6384, Loss: 0.2431
    Epoch  34 Batch 1500/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6406, Loss: 0.0776
    Epoch  34 Batch 1600/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6406, Loss: 0.0715
    Epoch  34 Batch 1700/2536 - Train Accuracy: 0.9858, Validation Accuracy: 0.6384, Loss: 0.0542
    Epoch  34 Batch 1800/2536 - Train Accuracy: 0.9858, Validation Accuracy: 0.6406, Loss: 0.0456
    Epoch  34 Batch 1900/2536 - Train Accuracy: 0.9661, Validation Accuracy: 0.6406, Loss: 0.0994
    Epoch  34 Batch 2000/2536 - Train Accuracy: 0.9471, Validation Accuracy: 0.6384, Loss: 0.0868
    Epoch  34 Batch 2100/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6429, Loss: 0.0138
    Epoch  34 Batch 2200/2536 - Train Accuracy: 0.9563, Validation Accuracy: 0.6362, Loss: 0.0996
    Epoch  34 Batch 2300/2536 - Train Accuracy: 0.9159, Validation Accuracy: 0.6362, Loss: 0.1912
    Epoch  34 Batch 2400/2536 - Train Accuracy: 0.9183, Validation Accuracy: 0.6362, Loss: 0.1163
    Epoch  34 Batch 2500/2536 - Train Accuracy: 0.9207, Validation Accuracy: 0.6362, Loss: 0.1847
    Epoch  35 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6384, Loss: 0.0387
    Epoch  35 Batch  200/2536 - Train Accuracy: 0.9665, Validation Accuracy: 0.6362, Loss: 0.0903
    Epoch  35 Batch  300/2536 - Train Accuracy: 0.9732, Validation Accuracy: 0.6362, Loss: 0.0931
    Epoch  35 Batch  400/2536 - Train Accuracy: 0.9316, Validation Accuracy: 0.6362, Loss: 0.1329
    Epoch  35 Batch  500/2536 - Train Accuracy: 0.9167, Validation Accuracy: 0.6384, Loss: 0.1422
    Epoch  35 Batch  600/2536 - Train Accuracy: 0.9417, Validation Accuracy: 0.6362, Loss: 0.1898
    Epoch  35 Batch  700/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6362, Loss: 0.1103
    Epoch  35 Batch  800/2536 - Train Accuracy: 0.9327, Validation Accuracy: 0.6384, Loss: 0.1756
    Epoch  35 Batch  900/2536 - Train Accuracy: 0.9353, Validation Accuracy: 0.6362, Loss: 0.1832
    Epoch  35 Batch 1000/2536 - Train Accuracy: 0.8479, Validation Accuracy: 0.6384, Loss: 0.3297
    Epoch  35 Batch 1100/2536 - Train Accuracy: 0.8571, Validation Accuracy: 0.6384, Loss: 0.3195
    Epoch  35 Batch 1200/2536 - Train Accuracy: 0.8711, Validation Accuracy: 0.6384, Loss: 0.2431
    Epoch  35 Batch 1300/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6362, Loss: 0.0161
    Epoch  35 Batch 1400/2536 - Train Accuracy: 0.8833, Validation Accuracy: 0.6362, Loss: 0.2305
    Epoch  35 Batch 1500/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6384, Loss: 0.0711
    Epoch  35 Batch 1600/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6384, Loss: 0.0510
    Epoch  35 Batch 1700/2536 - Train Accuracy: 0.9716, Validation Accuracy: 0.6384, Loss: 0.0581
    Epoch  35 Batch 1800/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6362, Loss: 0.0519
    Epoch  35 Batch 1900/2536 - Train Accuracy: 0.9427, Validation Accuracy: 0.6362, Loss: 0.0839
    Epoch  35 Batch 2000/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6384, Loss: 0.0798
    Epoch  35 Batch 2100/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6406, Loss: 0.0122
    Epoch  35 Batch 2200/2536 - Train Accuracy: 0.9604, Validation Accuracy: 0.6384, Loss: 0.1063
    Epoch  35 Batch 2300/2536 - Train Accuracy: 0.9159, Validation Accuracy: 0.6339, Loss: 0.1984
    Epoch  35 Batch 2400/2536 - Train Accuracy: 0.9423, Validation Accuracy: 0.6362, Loss: 0.1133
    Epoch  35 Batch 2500/2536 - Train Accuracy: 0.9399, Validation Accuracy: 0.6384, Loss: 0.1695
    Epoch  36 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6384, Loss: 0.0455
    Epoch  36 Batch  200/2536 - Train Accuracy: 0.9710, Validation Accuracy: 0.6384, Loss: 0.0846
    Epoch  36 Batch  300/2536 - Train Accuracy: 0.9598, Validation Accuracy: 0.6384, Loss: 0.1029
    Epoch  36 Batch  400/2536 - Train Accuracy: 0.9355, Validation Accuracy: 0.6362, Loss: 0.1111
    Epoch  36 Batch  500/2536 - Train Accuracy: 0.9250, Validation Accuracy: 0.6384, Loss: 0.1624
    Epoch  36 Batch  600/2536 - Train Accuracy: 0.9417, Validation Accuracy: 0.6362, Loss: 0.1828
    Epoch  36 Batch  700/2536 - Train Accuracy: 0.9495, Validation Accuracy: 0.6384, Loss: 0.1216
    Epoch  36 Batch  800/2536 - Train Accuracy: 0.9471, Validation Accuracy: 0.6362, Loss: 0.1410
    Epoch  36 Batch  900/2536 - Train Accuracy: 0.9196, Validation Accuracy: 0.6362, Loss: 0.1841
    Epoch  36 Batch 1000/2536 - Train Accuracy: 0.8500, Validation Accuracy: 0.6362, Loss: 0.2839
    Epoch  36 Batch 1100/2536 - Train Accuracy: 0.8549, Validation Accuracy: 0.6384, Loss: 0.2850
    Epoch  36 Batch 1200/2536 - Train Accuracy: 0.8867, Validation Accuracy: 0.6384, Loss: 0.2152
    Epoch  36 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0071
    Epoch  36 Batch 1400/2536 - Train Accuracy: 0.9250, Validation Accuracy: 0.6362, Loss: 0.1884
    Epoch  36 Batch 1500/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0520
    Epoch  36 Batch 1600/2536 - Train Accuracy: 0.9870, Validation Accuracy: 0.6362, Loss: 0.0529
    Epoch  36 Batch 1700/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0805
    Epoch  36 Batch 1800/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6362, Loss: 0.0240
    Epoch  36 Batch 1900/2536 - Train Accuracy: 0.9740, Validation Accuracy: 0.6362, Loss: 0.1120
    Epoch  36 Batch 2000/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0798
    Epoch  36 Batch 2100/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6384, Loss: 0.0144
    Epoch  36 Batch 2200/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0866
    Epoch  36 Batch 2300/2536 - Train Accuracy: 0.9231, Validation Accuracy: 0.6362, Loss: 0.1774
    Epoch  36 Batch 2400/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6362, Loss: 0.1111
    Epoch  36 Batch 2500/2536 - Train Accuracy: 0.9423, Validation Accuracy: 0.6384, Loss: 0.1796
    Epoch  37 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6384, Loss: 0.0272
    Epoch  37 Batch  200/2536 - Train Accuracy: 0.9665, Validation Accuracy: 0.6362, Loss: 0.0625
    Epoch  37 Batch  300/2536 - Train Accuracy: 0.9420, Validation Accuracy: 0.6384, Loss: 0.1182
    Epoch  37 Batch  400/2536 - Train Accuracy: 0.9316, Validation Accuracy: 0.6362, Loss: 0.1166
    Epoch  37 Batch  500/2536 - Train Accuracy: 0.9167, Validation Accuracy: 0.6406, Loss: 0.1662
    Epoch  37 Batch  600/2536 - Train Accuracy: 0.9521, Validation Accuracy: 0.6384, Loss: 0.1951
    Epoch  37 Batch  700/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6384, Loss: 0.1204
    Epoch  37 Batch  800/2536 - Train Accuracy: 0.9471, Validation Accuracy: 0.6406, Loss: 0.1374
    Epoch  37 Batch  900/2536 - Train Accuracy: 0.8929, Validation Accuracy: 0.6384, Loss: 0.1980
    Epoch  37 Batch 1000/2536 - Train Accuracy: 0.8708, Validation Accuracy: 0.6384, Loss: 0.2697
    Epoch  37 Batch 1100/2536 - Train Accuracy: 0.8750, Validation Accuracy: 0.6384, Loss: 0.3085
    Epoch  37 Batch 1200/2536 - Train Accuracy: 0.8809, Validation Accuracy: 0.6406, Loss: 0.2590
    Epoch  37 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0074
    Epoch  37 Batch 1400/2536 - Train Accuracy: 0.8688, Validation Accuracy: 0.6384, Loss: 0.1835
    Epoch  37 Batch 1500/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6384, Loss: 0.0594
    Epoch  37 Batch 1600/2536 - Train Accuracy: 0.9896, Validation Accuracy: 0.6406, Loss: 0.0733
    Epoch  37 Batch 1700/2536 - Train Accuracy: 0.9886, Validation Accuracy: 0.6384, Loss: 0.0487
    Epoch  37 Batch 1800/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6384, Loss: 0.0311
    Epoch  37 Batch 1900/2536 - Train Accuracy: 0.9401, Validation Accuracy: 0.6384, Loss: 0.0932
    Epoch  37 Batch 2000/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6384, Loss: 0.0613
    Epoch  37 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0103
    Epoch  37 Batch 2200/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0987
    Epoch  37 Batch 2300/2536 - Train Accuracy: 0.9255, Validation Accuracy: 0.6362, Loss: 0.1638
    Epoch  37 Batch 2400/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6384, Loss: 0.1168
    Epoch  37 Batch 2500/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6384, Loss: 0.1321
    Epoch  38 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6384, Loss: 0.0333
    Epoch  38 Batch  200/2536 - Train Accuracy: 0.9754, Validation Accuracy: 0.6384, Loss: 0.0515
    Epoch  38 Batch  300/2536 - Train Accuracy: 0.9576, Validation Accuracy: 0.6384, Loss: 0.1118
    Epoch  38 Batch  400/2536 - Train Accuracy: 0.9355, Validation Accuracy: 0.6384, Loss: 0.1298
    Epoch  38 Batch  500/2536 - Train Accuracy: 0.9417, Validation Accuracy: 0.6384, Loss: 0.1307
    Epoch  38 Batch  600/2536 - Train Accuracy: 0.9563, Validation Accuracy: 0.6384, Loss: 0.1720
    Epoch  38 Batch  700/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6384, Loss: 0.1122
    Epoch  38 Batch  800/2536 - Train Accuracy: 0.9519, Validation Accuracy: 0.6384, Loss: 0.1112
    Epoch  38 Batch  900/2536 - Train Accuracy: 0.9040, Validation Accuracy: 0.6384, Loss: 0.1345
    Epoch  38 Batch 1000/2536 - Train Accuracy: 0.8667, Validation Accuracy: 0.6406, Loss: 0.2639
    Epoch  38 Batch 1100/2536 - Train Accuracy: 0.8326, Validation Accuracy: 0.6384, Loss: 0.3031
    Epoch  38 Batch 1200/2536 - Train Accuracy: 0.8516, Validation Accuracy: 0.6384, Loss: 0.2247
    Epoch  38 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0105
    Epoch  38 Batch 1400/2536 - Train Accuracy: 0.9167, Validation Accuracy: 0.6384, Loss: 0.1938
    Epoch  38 Batch 1500/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6384, Loss: 0.0553
    Epoch  38 Batch 1600/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6384, Loss: 0.0654
    Epoch  38 Batch 1700/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6384, Loss: 0.0457
    Epoch  38 Batch 1800/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6384, Loss: 0.0325
    Epoch  38 Batch 1900/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0968
    Epoch  38 Batch 2000/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6384, Loss: 0.0585
    Epoch  38 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0207
    Epoch  38 Batch 2200/2536 - Train Accuracy: 0.9500, Validation Accuracy: 0.6362, Loss: 0.0817
    Epoch  38 Batch 2300/2536 - Train Accuracy: 0.9087, Validation Accuracy: 0.6384, Loss: 0.1734
    Epoch  38 Batch 2400/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6384, Loss: 0.0909
    Epoch  38 Batch 2500/2536 - Train Accuracy: 0.9639, Validation Accuracy: 0.6406, Loss: 0.1380
    Epoch  39 Batch  100/2536 - Train Accuracy: 0.9886, Validation Accuracy: 0.6384, Loss: 0.0376
    Epoch  39 Batch  200/2536 - Train Accuracy: 0.9621, Validation Accuracy: 0.6384, Loss: 0.0587
    Epoch  39 Batch  300/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0849
    Epoch  39 Batch  400/2536 - Train Accuracy: 0.9629, Validation Accuracy: 0.6384, Loss: 0.1114
    Epoch  39 Batch  500/2536 - Train Accuracy: 0.9292, Validation Accuracy: 0.6406, Loss: 0.1066
    Epoch  39 Batch  600/2536 - Train Accuracy: 0.9354, Validation Accuracy: 0.6384, Loss: 0.1710
    Epoch  39 Batch  700/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6384, Loss: 0.0807
    Epoch  39 Batch  800/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6384, Loss: 0.1359
    Epoch  39 Batch  900/2536 - Train Accuracy: 0.9040, Validation Accuracy: 0.6384, Loss: 0.1552
    Epoch  39 Batch 1000/2536 - Train Accuracy: 0.9042, Validation Accuracy: 0.6384, Loss: 0.2652
    Epoch  39 Batch 1100/2536 - Train Accuracy: 0.8594, Validation Accuracy: 0.6384, Loss: 0.2587
    Epoch  39 Batch 1200/2536 - Train Accuracy: 0.8789, Validation Accuracy: 0.6384, Loss: 0.2172
    Epoch  39 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0082
    Epoch  39 Batch 1400/2536 - Train Accuracy: 0.8938, Validation Accuracy: 0.6384, Loss: 0.1778
    Epoch  39 Batch 1500/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6384, Loss: 0.0498
    Epoch  39 Batch 1600/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6384, Loss: 0.0411
    Epoch  39 Batch 1700/2536 - Train Accuracy: 0.9858, Validation Accuracy: 0.6384, Loss: 0.0427
    Epoch  39 Batch 1800/2536 - Train Accuracy: 0.9744, Validation Accuracy: 0.6384, Loss: 0.0456
    Epoch  39 Batch 1900/2536 - Train Accuracy: 0.9609, Validation Accuracy: 0.6362, Loss: 0.0913
    Epoch  39 Batch 2000/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6384, Loss: 0.0490
    Epoch  39 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0107
    Epoch  39 Batch 2200/2536 - Train Accuracy: 0.9333, Validation Accuracy: 0.6362, Loss: 0.0851
    Epoch  39 Batch 2300/2536 - Train Accuracy: 0.9062, Validation Accuracy: 0.6384, Loss: 0.1367
    Epoch  39 Batch 2400/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6384, Loss: 0.0906
    Epoch  39 Batch 2500/2536 - Train Accuracy: 0.9471, Validation Accuracy: 0.6384, Loss: 0.1237
    Epoch  40 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6384, Loss: 0.0238
    Epoch  40 Batch  200/2536 - Train Accuracy: 0.9888, Validation Accuracy: 0.6384, Loss: 0.0545
    Epoch  40 Batch  300/2536 - Train Accuracy: 0.9754, Validation Accuracy: 0.6384, Loss: 0.0906
    Epoch  40 Batch  400/2536 - Train Accuracy: 0.9512, Validation Accuracy: 0.6384, Loss: 0.1325
    Epoch  40 Batch  500/2536 - Train Accuracy: 0.9250, Validation Accuracy: 0.6384, Loss: 0.1347
    Epoch  40 Batch  600/2536 - Train Accuracy: 0.9437, Validation Accuracy: 0.6384, Loss: 0.1618
    Epoch  40 Batch  700/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6384, Loss: 0.0789
    Epoch  40 Batch  800/2536 - Train Accuracy: 0.9639, Validation Accuracy: 0.6384, Loss: 0.1037
    Epoch  40 Batch  900/2536 - Train Accuracy: 0.9263, Validation Accuracy: 0.6384, Loss: 0.1632
    Epoch  40 Batch 1000/2536 - Train Accuracy: 0.8917, Validation Accuracy: 0.6384, Loss: 0.2278
    Epoch  40 Batch 1100/2536 - Train Accuracy: 0.8549, Validation Accuracy: 0.6406, Loss: 0.2533
    Epoch  40 Batch 1200/2536 - Train Accuracy: 0.8887, Validation Accuracy: 0.6384, Loss: 0.2437
    Epoch  40 Batch 1300/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6362, Loss: 0.0097
    Epoch  40 Batch 1400/2536 - Train Accuracy: 0.9458, Validation Accuracy: 0.6384, Loss: 0.1700
    Epoch  40 Batch 1500/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6384, Loss: 0.0400
    Epoch  40 Batch 1600/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6384, Loss: 0.0420
    Epoch  40 Batch 1700/2536 - Train Accuracy: 0.9801, Validation Accuracy: 0.6384, Loss: 0.0481
    Epoch  40 Batch 1800/2536 - Train Accuracy: 0.9773, Validation Accuracy: 0.6384, Loss: 0.0292
    Epoch  40 Batch 1900/2536 - Train Accuracy: 0.9714, Validation Accuracy: 0.6384, Loss: 0.0886
    Epoch  40 Batch 2000/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6384, Loss: 0.0512
    Epoch  40 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0085
    Epoch  40 Batch 2200/2536 - Train Accuracy: 0.9771, Validation Accuracy: 0.6384, Loss: 0.0692
    Epoch  40 Batch 2300/2536 - Train Accuracy: 0.9303, Validation Accuracy: 0.6384, Loss: 0.1422
    Epoch  40 Batch 2400/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6362, Loss: 0.0832
    Epoch  40 Batch 2500/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6384, Loss: 0.1315
    Epoch  41 Batch  100/2536 - Train Accuracy: 0.9858, Validation Accuracy: 0.6406, Loss: 0.0501
    Epoch  41 Batch  200/2536 - Train Accuracy: 0.9777, Validation Accuracy: 0.6384, Loss: 0.0434
    Epoch  41 Batch  300/2536 - Train Accuracy: 0.9732, Validation Accuracy: 0.6384, Loss: 0.0587
    Epoch  41 Batch  400/2536 - Train Accuracy: 0.9609, Validation Accuracy: 0.6384, Loss: 0.0945
    Epoch  41 Batch  500/2536 - Train Accuracy: 0.9354, Validation Accuracy: 0.6406, Loss: 0.1042
    Epoch  41 Batch  600/2536 - Train Accuracy: 0.9396, Validation Accuracy: 0.6384, Loss: 0.1500
    Epoch  41 Batch  700/2536 - Train Accuracy: 0.9567, Validation Accuracy: 0.6384, Loss: 0.0903
    Epoch  41 Batch  800/2536 - Train Accuracy: 0.9615, Validation Accuracy: 0.6384, Loss: 0.1034
    Epoch  41 Batch  900/2536 - Train Accuracy: 0.9085, Validation Accuracy: 0.6384, Loss: 0.1104
    Epoch  41 Batch 1000/2536 - Train Accuracy: 0.8625, Validation Accuracy: 0.6406, Loss: 0.2465
    Epoch  41 Batch 1100/2536 - Train Accuracy: 0.8304, Validation Accuracy: 0.6384, Loss: 0.2336
    Epoch  41 Batch 1200/2536 - Train Accuracy: 0.8926, Validation Accuracy: 0.6384, Loss: 0.1871
    Epoch  41 Batch 1300/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6384, Loss: 0.0107
    Epoch  41 Batch 1400/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6384, Loss: 0.1681
    Epoch  41 Batch 1500/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6429, Loss: 0.0493
    Epoch  41 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0497
    Epoch  41 Batch 1700/2536 - Train Accuracy: 0.9886, Validation Accuracy: 0.6362, Loss: 0.0413
    Epoch  41 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0315
    Epoch  41 Batch 1900/2536 - Train Accuracy: 0.9818, Validation Accuracy: 0.6339, Loss: 0.0805
    Epoch  41 Batch 2000/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6384, Loss: 0.0477
    Epoch  41 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0096
    Epoch  41 Batch 2200/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6384, Loss: 0.0767
    Epoch  41 Batch 2300/2536 - Train Accuracy: 0.9351, Validation Accuracy: 0.6384, Loss: 0.1480
    Epoch  41 Batch 2400/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6384, Loss: 0.0940
    Epoch  41 Batch 2500/2536 - Train Accuracy: 0.9615, Validation Accuracy: 0.6384, Loss: 0.1255
    Epoch  42 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6406, Loss: 0.0527
    Epoch  42 Batch  200/2536 - Train Accuracy: 0.9933, Validation Accuracy: 0.6384, Loss: 0.0516
    Epoch  42 Batch  300/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0645
    Epoch  42 Batch  400/2536 - Train Accuracy: 0.9160, Validation Accuracy: 0.6384, Loss: 0.1031
    Epoch  42 Batch  500/2536 - Train Accuracy: 0.9354, Validation Accuracy: 0.6384, Loss: 0.0996
    Epoch  42 Batch  600/2536 - Train Accuracy: 0.9458, Validation Accuracy: 0.6384, Loss: 0.1485
    Epoch  42 Batch  700/2536 - Train Accuracy: 0.9447, Validation Accuracy: 0.6384, Loss: 0.0920
    Epoch  42 Batch  800/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6406, Loss: 0.1404
    Epoch  42 Batch  900/2536 - Train Accuracy: 0.9598, Validation Accuracy: 0.6384, Loss: 0.1176
    Epoch  42 Batch 1000/2536 - Train Accuracy: 0.8688, Validation Accuracy: 0.6384, Loss: 0.2100
    Epoch  42 Batch 1100/2536 - Train Accuracy: 0.8996, Validation Accuracy: 0.6384, Loss: 0.2342
    Epoch  42 Batch 1200/2536 - Train Accuracy: 0.8887, Validation Accuracy: 0.6384, Loss: 0.1972
    Epoch  42 Batch 1300/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6384, Loss: 0.0156
    Epoch  42 Batch 1400/2536 - Train Accuracy: 0.9542, Validation Accuracy: 0.6384, Loss: 0.1533
    Epoch  42 Batch 1500/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6406, Loss: 0.0380
    Epoch  42 Batch 1600/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6362, Loss: 0.0309
    Epoch  42 Batch 1700/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6362, Loss: 0.0209
    Epoch  42 Batch 1800/2536 - Train Accuracy: 0.9744, Validation Accuracy: 0.6384, Loss: 0.0518
    Epoch  42 Batch 1900/2536 - Train Accuracy: 0.9818, Validation Accuracy: 0.6362, Loss: 0.0639
    Epoch  42 Batch 2000/2536 - Train Accuracy: 0.9712, Validation Accuracy: 0.6362, Loss: 0.0519
    Epoch  42 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0138
    Epoch  42 Batch 2200/2536 - Train Accuracy: 0.9646, Validation Accuracy: 0.6384, Loss: 0.0818
    Epoch  42 Batch 2300/2536 - Train Accuracy: 0.9447, Validation Accuracy: 0.6384, Loss: 0.1517
    Epoch  42 Batch 2400/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6384, Loss: 0.0725
    Epoch  42 Batch 2500/2536 - Train Accuracy: 0.9712, Validation Accuracy: 0.6362, Loss: 0.1173
    Epoch  43 Batch  100/2536 - Train Accuracy: 0.9886, Validation Accuracy: 0.6384, Loss: 0.0190
    Epoch  43 Batch  200/2536 - Train Accuracy: 0.9821, Validation Accuracy: 0.6384, Loss: 0.0701
    Epoch  43 Batch  300/2536 - Train Accuracy: 0.9621, Validation Accuracy: 0.6384, Loss: 0.0594
    Epoch  43 Batch  400/2536 - Train Accuracy: 0.9473, Validation Accuracy: 0.6384, Loss: 0.1060
    Epoch  43 Batch  500/2536 - Train Accuracy: 0.9521, Validation Accuracy: 0.6384, Loss: 0.0943
    Epoch  43 Batch  600/2536 - Train Accuracy: 0.9396, Validation Accuracy: 0.6384, Loss: 0.1525
    Epoch  43 Batch  700/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0631
    Epoch  43 Batch  800/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6384, Loss: 0.0957
    Epoch  43 Batch  900/2536 - Train Accuracy: 0.9598, Validation Accuracy: 0.6406, Loss: 0.1164
    Epoch  43 Batch 1000/2536 - Train Accuracy: 0.9062, Validation Accuracy: 0.6384, Loss: 0.2199
    Epoch  43 Batch 1100/2536 - Train Accuracy: 0.9263, Validation Accuracy: 0.6384, Loss: 0.2280
    Epoch  43 Batch 1200/2536 - Train Accuracy: 0.9004, Validation Accuracy: 0.6384, Loss: 0.1724
    Epoch  43 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0073
    Epoch  43 Batch 1400/2536 - Train Accuracy: 0.9271, Validation Accuracy: 0.6384, Loss: 0.1248
    Epoch  43 Batch 1500/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6384, Loss: 0.0423
    Epoch  43 Batch 1600/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6384, Loss: 0.0397
    Epoch  43 Batch 1700/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6384, Loss: 0.0275
    Epoch  43 Batch 1800/2536 - Train Accuracy: 0.9830, Validation Accuracy: 0.6384, Loss: 0.0390
    Epoch  43 Batch 1900/2536 - Train Accuracy: 0.9818, Validation Accuracy: 0.6384, Loss: 0.0666
    Epoch  43 Batch 2000/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6384, Loss: 0.0354
    Epoch  43 Batch 2100/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6406, Loss: 0.0104
    Epoch  43 Batch 2200/2536 - Train Accuracy: 0.9625, Validation Accuracy: 0.6406, Loss: 0.0594
    Epoch  43 Batch 2300/2536 - Train Accuracy: 0.9543, Validation Accuracy: 0.6362, Loss: 0.1222
    Epoch  43 Batch 2400/2536 - Train Accuracy: 0.9447, Validation Accuracy: 0.6384, Loss: 0.0747
    Epoch  43 Batch 2500/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6384, Loss: 0.1001
    Epoch  44 Batch  100/2536 - Train Accuracy: 0.9830, Validation Accuracy: 0.6384, Loss: 0.0478
    Epoch  44 Batch  200/2536 - Train Accuracy: 0.9888, Validation Accuracy: 0.6384, Loss: 0.0385
    Epoch  44 Batch  300/2536 - Train Accuracy: 0.9598, Validation Accuracy: 0.6384, Loss: 0.0650
    Epoch  44 Batch  400/2536 - Train Accuracy: 0.9355, Validation Accuracy: 0.6384, Loss: 0.1008
    Epoch  44 Batch  500/2536 - Train Accuracy: 0.9396, Validation Accuracy: 0.6384, Loss: 0.1031
    Epoch  44 Batch  600/2536 - Train Accuracy: 0.9542, Validation Accuracy: 0.6384, Loss: 0.1330
    Epoch  44 Batch  700/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6384, Loss: 0.0662
    Epoch  44 Batch  800/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6384, Loss: 0.0989
    Epoch  44 Batch  900/2536 - Train Accuracy: 0.9219, Validation Accuracy: 0.6384, Loss: 0.1233
    Epoch  44 Batch 1000/2536 - Train Accuracy: 0.8750, Validation Accuracy: 0.6384, Loss: 0.2282
    Epoch  44 Batch 1100/2536 - Train Accuracy: 0.9085, Validation Accuracy: 0.6384, Loss: 0.2059
    Epoch  44 Batch 1200/2536 - Train Accuracy: 0.8848, Validation Accuracy: 0.6406, Loss: 0.1653
    Epoch  44 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0066
    Epoch  44 Batch 1400/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6384, Loss: 0.1516
    Epoch  44 Batch 1500/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6406, Loss: 0.0444
    Epoch  44 Batch 1600/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6384, Loss: 0.0434
    Epoch  44 Batch 1700/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6384, Loss: 0.0290
    Epoch  44 Batch 1800/2536 - Train Accuracy: 0.9858, Validation Accuracy: 0.6384, Loss: 0.0230
    Epoch  44 Batch 1900/2536 - Train Accuracy: 0.9714, Validation Accuracy: 0.6362, Loss: 0.0588
    Epoch  44 Batch 2000/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6362, Loss: 0.0446
    Epoch  44 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0119
    Epoch  44 Batch 2200/2536 - Train Accuracy: 0.9771, Validation Accuracy: 0.6362, Loss: 0.0534
    Epoch  44 Batch 2300/2536 - Train Accuracy: 0.9399, Validation Accuracy: 0.6362, Loss: 0.1490
    Epoch  44 Batch 2400/2536 - Train Accuracy: 0.9712, Validation Accuracy: 0.6384, Loss: 0.0791
    Epoch  44 Batch 2500/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6384, Loss: 0.1135
    Epoch  45 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6384, Loss: 0.0262
    Epoch  45 Batch  200/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0422
    Epoch  45 Batch  300/2536 - Train Accuracy: 0.9598, Validation Accuracy: 0.6384, Loss: 0.0473
    Epoch  45 Batch  400/2536 - Train Accuracy: 0.9453, Validation Accuracy: 0.6362, Loss: 0.0703
    Epoch  45 Batch  500/2536 - Train Accuracy: 0.9583, Validation Accuracy: 0.6362, Loss: 0.0964
    Epoch  45 Batch  600/2536 - Train Accuracy: 0.9437, Validation Accuracy: 0.6362, Loss: 0.1357
    Epoch  45 Batch  700/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6384, Loss: 0.0675
    Epoch  45 Batch  800/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6384, Loss: 0.0824
    Epoch  45 Batch  900/2536 - Train Accuracy: 0.9598, Validation Accuracy: 0.6362, Loss: 0.1179
    Epoch  45 Batch 1000/2536 - Train Accuracy: 0.8979, Validation Accuracy: 0.6384, Loss: 0.1849
    Epoch  45 Batch 1100/2536 - Train Accuracy: 0.9129, Validation Accuracy: 0.6406, Loss: 0.1890
    Epoch  45 Batch 1200/2536 - Train Accuracy: 0.8984, Validation Accuracy: 0.6384, Loss: 0.1769
    Epoch  45 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0092
    Epoch  45 Batch 1400/2536 - Train Accuracy: 0.9333, Validation Accuracy: 0.6384, Loss: 0.1222
    Epoch  45 Batch 1500/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6362, Loss: 0.0319
    Epoch  45 Batch 1600/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6339, Loss: 0.0312
    Epoch  45 Batch 1700/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6339, Loss: 0.0338
    Epoch  45 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0149
    Epoch  45 Batch 1900/2536 - Train Accuracy: 0.9635, Validation Accuracy: 0.6339, Loss: 0.0660
    Epoch  45 Batch 2000/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6339, Loss: 0.0508
    Epoch  45 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0068
    Epoch  45 Batch 2200/2536 - Train Accuracy: 0.9938, Validation Accuracy: 0.6339, Loss: 0.0474
    Epoch  45 Batch 2300/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6339, Loss: 0.1205
    Epoch  45 Batch 2400/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6384, Loss: 0.0782
    Epoch  45 Batch 2500/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6339, Loss: 0.1029
    Epoch  46 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6362, Loss: 0.0349
    Epoch  46 Batch  200/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6362, Loss: 0.0516
    Epoch  46 Batch  300/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6362, Loss: 0.0533
    Epoch  46 Batch  400/2536 - Train Accuracy: 0.9492, Validation Accuracy: 0.6362, Loss: 0.0744
    Epoch  46 Batch  500/2536 - Train Accuracy: 0.9583, Validation Accuracy: 0.6362, Loss: 0.0828
    Epoch  46 Batch  600/2536 - Train Accuracy: 0.9625, Validation Accuracy: 0.6339, Loss: 0.1399
    Epoch  46 Batch  700/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6384, Loss: 0.0783
    Epoch  46 Batch  800/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0902
    Epoch  46 Batch  900/2536 - Train Accuracy: 0.9732, Validation Accuracy: 0.6384, Loss: 0.0839
    Epoch  46 Batch 1000/2536 - Train Accuracy: 0.9146, Validation Accuracy: 0.6384, Loss: 0.1882
    Epoch  46 Batch 1100/2536 - Train Accuracy: 0.9241, Validation Accuracy: 0.6384, Loss: 0.1728
    Epoch  46 Batch 1200/2536 - Train Accuracy: 0.9023, Validation Accuracy: 0.6384, Loss: 0.1701
    Epoch  46 Batch 1300/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6362, Loss: 0.0059
    Epoch  46 Batch 1400/2536 - Train Accuracy: 0.9563, Validation Accuracy: 0.6362, Loss: 0.1370
    Epoch  46 Batch 1500/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6362, Loss: 0.0255
    Epoch  46 Batch 1600/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6362, Loss: 0.0408
    Epoch  46 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0262
    Epoch  46 Batch 1800/2536 - Train Accuracy: 0.9801, Validation Accuracy: 0.6384, Loss: 0.0149
    Epoch  46 Batch 1900/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6362, Loss: 0.0575
    Epoch  46 Batch 2000/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6384, Loss: 0.0588
    Epoch  46 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0129
    Epoch  46 Batch 2200/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0597
    Epoch  46 Batch 2300/2536 - Train Accuracy: 0.9255, Validation Accuracy: 0.6384, Loss: 0.1005
    Epoch  46 Batch 2400/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0543
    Epoch  46 Batch 2500/2536 - Train Accuracy: 0.9639, Validation Accuracy: 0.6362, Loss: 0.1321
    Epoch  47 Batch  100/2536 - Train Accuracy: 0.9858, Validation Accuracy: 0.6384, Loss: 0.0245
    Epoch  47 Batch  200/2536 - Train Accuracy: 0.9933, Validation Accuracy: 0.6384, Loss: 0.0336
    Epoch  47 Batch  300/2536 - Train Accuracy: 0.9710, Validation Accuracy: 0.6384, Loss: 0.0554
    Epoch  47 Batch  400/2536 - Train Accuracy: 0.9902, Validation Accuracy: 0.6362, Loss: 0.0662
    Epoch  47 Batch  500/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0643
    Epoch  47 Batch  600/2536 - Train Accuracy: 0.9563, Validation Accuracy: 0.6362, Loss: 0.1084
    Epoch  47 Batch  700/2536 - Train Accuracy: 0.9615, Validation Accuracy: 0.6384, Loss: 0.0524
    Epoch  47 Batch  800/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6384, Loss: 0.0915
    Epoch  47 Batch  900/2536 - Train Accuracy: 0.9598, Validation Accuracy: 0.6406, Loss: 0.1063
    Epoch  47 Batch 1000/2536 - Train Accuracy: 0.8958, Validation Accuracy: 0.6384, Loss: 0.1750
    Epoch  47 Batch 1100/2536 - Train Accuracy: 0.9397, Validation Accuracy: 0.6384, Loss: 0.2023
    Epoch  47 Batch 1200/2536 - Train Accuracy: 0.9492, Validation Accuracy: 0.6406, Loss: 0.1379
    Epoch  47 Batch 1300/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6362, Loss: 0.0068
    Epoch  47 Batch 1400/2536 - Train Accuracy: 0.9500, Validation Accuracy: 0.6384, Loss: 0.1153
    Epoch  47 Batch 1500/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6384, Loss: 0.0319
    Epoch  47 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0354
    Epoch  47 Batch 1700/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6362, Loss: 0.0294
    Epoch  47 Batch 1800/2536 - Train Accuracy: 0.9773, Validation Accuracy: 0.6362, Loss: 0.0201
    Epoch  47 Batch 1900/2536 - Train Accuracy: 0.9714, Validation Accuracy: 0.6362, Loss: 0.0764
    Epoch  47 Batch 2000/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6384, Loss: 0.0403
    Epoch  47 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0095
    Epoch  47 Batch 2200/2536 - Train Accuracy: 0.9708, Validation Accuracy: 0.6362, Loss: 0.0449
    Epoch  47 Batch 2300/2536 - Train Accuracy: 0.9447, Validation Accuracy: 0.6362, Loss: 0.0990
    Epoch  47 Batch 2400/2536 - Train Accuracy: 0.9712, Validation Accuracy: 0.6362, Loss: 0.0786
    Epoch  47 Batch 2500/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6339, Loss: 0.0744
    Epoch  48 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6362, Loss: 0.0307
    Epoch  48 Batch  200/2536 - Train Accuracy: 0.9754, Validation Accuracy: 0.6384, Loss: 0.0395
    Epoch  48 Batch  300/2536 - Train Accuracy: 0.9710, Validation Accuracy: 0.6384, Loss: 0.0574
    Epoch  48 Batch  400/2536 - Train Accuracy: 0.9609, Validation Accuracy: 0.6384, Loss: 0.0705
    Epoch  48 Batch  500/2536 - Train Accuracy: 0.9479, Validation Accuracy: 0.6362, Loss: 0.0712
    Epoch  48 Batch  600/2536 - Train Accuracy: 0.9479, Validation Accuracy: 0.6362, Loss: 0.0999
    Epoch  48 Batch  700/2536 - Train Accuracy: 0.9447, Validation Accuracy: 0.6362, Loss: 0.0664
    Epoch  48 Batch  800/2536 - Train Accuracy: 0.9615, Validation Accuracy: 0.6362, Loss: 0.0983
    Epoch  48 Batch  900/2536 - Train Accuracy: 0.9665, Validation Accuracy: 0.6362, Loss: 0.0907
    Epoch  48 Batch 1000/2536 - Train Accuracy: 0.9042, Validation Accuracy: 0.6384, Loss: 0.1619
    Epoch  48 Batch 1100/2536 - Train Accuracy: 0.9330, Validation Accuracy: 0.6384, Loss: 0.2033
    Epoch  48 Batch 1200/2536 - Train Accuracy: 0.9043, Validation Accuracy: 0.6406, Loss: 0.1350
    Epoch  48 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0055
    Epoch  48 Batch 1400/2536 - Train Accuracy: 0.9313, Validation Accuracy: 0.6384, Loss: 0.1195
    Epoch  48 Batch 1500/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6362, Loss: 0.0246
    Epoch  48 Batch 1600/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6362, Loss: 0.0314
    Epoch  48 Batch 1700/2536 - Train Accuracy: 0.9830, Validation Accuracy: 0.6406, Loss: 0.0312
    Epoch  48 Batch 1800/2536 - Train Accuracy: 0.9773, Validation Accuracy: 0.6362, Loss: 0.0238
    Epoch  48 Batch 1900/2536 - Train Accuracy: 0.9766, Validation Accuracy: 0.6384, Loss: 0.0580
    Epoch  48 Batch 2000/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6384, Loss: 0.0390
    Epoch  48 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0107
    Epoch  48 Batch 2200/2536 - Train Accuracy: 0.9917, Validation Accuracy: 0.6384, Loss: 0.0598
    Epoch  48 Batch 2300/2536 - Train Accuracy: 0.9303, Validation Accuracy: 0.6362, Loss: 0.0952
    Epoch  48 Batch 2400/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6362, Loss: 0.0663
    Epoch  48 Batch 2500/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6362, Loss: 0.1049
    Epoch  49 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6384, Loss: 0.0248
    Epoch  49 Batch  200/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0347
    Epoch  49 Batch  300/2536 - Train Accuracy: 0.9710, Validation Accuracy: 0.6384, Loss: 0.0672
    Epoch  49 Batch  400/2536 - Train Accuracy: 0.9492, Validation Accuracy: 0.6339, Loss: 0.0807
    Epoch  49 Batch  500/2536 - Train Accuracy: 0.9500, Validation Accuracy: 0.6362, Loss: 0.0593
    Epoch  49 Batch  600/2536 - Train Accuracy: 0.9500, Validation Accuracy: 0.6384, Loss: 0.1066
    Epoch  49 Batch  700/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6384, Loss: 0.0558
    Epoch  49 Batch  800/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6362, Loss: 0.0932
    Epoch  49 Batch  900/2536 - Train Accuracy: 0.9420, Validation Accuracy: 0.6384, Loss: 0.0960
    Epoch  49 Batch 1000/2536 - Train Accuracy: 0.9187, Validation Accuracy: 0.6384, Loss: 0.1802
    Epoch  49 Batch 1100/2536 - Train Accuracy: 0.8817, Validation Accuracy: 0.6384, Loss: 0.1867
    Epoch  49 Batch 1200/2536 - Train Accuracy: 0.9199, Validation Accuracy: 0.6384, Loss: 0.1366
    Epoch  49 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0062
    Epoch  49 Batch 1400/2536 - Train Accuracy: 0.9542, Validation Accuracy: 0.6384, Loss: 0.1182
    Epoch  49 Batch 1500/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0322
    Epoch  49 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0316
    Epoch  49 Batch 1700/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6362, Loss: 0.0316
    Epoch  49 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0216
    Epoch  49 Batch 1900/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6384, Loss: 0.0643
    Epoch  49 Batch 2000/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6384, Loss: 0.0438
    Epoch  49 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0043
    Epoch  49 Batch 2200/2536 - Train Accuracy: 0.9854, Validation Accuracy: 0.6362, Loss: 0.0497
    Epoch  49 Batch 2300/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6362, Loss: 0.0980
    Epoch  49 Batch 2400/2536 - Train Accuracy: 0.9712, Validation Accuracy: 0.6384, Loss: 0.0674
    Epoch  49 Batch 2500/2536 - Train Accuracy: 0.9567, Validation Accuracy: 0.6362, Loss: 0.0626
    Epoch  50 Batch  100/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6384, Loss: 0.0232
    Epoch  50 Batch  200/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6406, Loss: 0.0312
    Epoch  50 Batch  300/2536 - Train Accuracy: 0.9777, Validation Accuracy: 0.6384, Loss: 0.0522
    Epoch  50 Batch  400/2536 - Train Accuracy: 0.9395, Validation Accuracy: 0.6406, Loss: 0.0658
    Epoch  50 Batch  500/2536 - Train Accuracy: 0.9750, Validation Accuracy: 0.6384, Loss: 0.0708
    Epoch  50 Batch  600/2536 - Train Accuracy: 0.9437, Validation Accuracy: 0.6362, Loss: 0.1046
    Epoch  50 Batch  700/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6362, Loss: 0.0487
    Epoch  50 Batch  800/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6362, Loss: 0.0822
    Epoch  50 Batch  900/2536 - Train Accuracy: 0.9643, Validation Accuracy: 0.6362, Loss: 0.0799
    Epoch  50 Batch 1000/2536 - Train Accuracy: 0.8938, Validation Accuracy: 0.6362, Loss: 0.1926
    Epoch  50 Batch 1100/2536 - Train Accuracy: 0.9219, Validation Accuracy: 0.6384, Loss: 0.1994
    Epoch  50 Batch 1200/2536 - Train Accuracy: 0.9512, Validation Accuracy: 0.6384, Loss: 0.1234
    Epoch  50 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0100
    Epoch  50 Batch 1400/2536 - Train Accuracy: 0.9625, Validation Accuracy: 0.6362, Loss: 0.0916
    Epoch  50 Batch 1500/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6362, Loss: 0.0226
    Epoch  50 Batch 1600/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6362, Loss: 0.0245
    Epoch  50 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0195
    Epoch  50 Batch 1800/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6362, Loss: 0.0153
    Epoch  50 Batch 1900/2536 - Train Accuracy: 0.9635, Validation Accuracy: 0.6384, Loss: 0.0497
    Epoch  50 Batch 2000/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6362, Loss: 0.0270
    Epoch  50 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0105
    Epoch  50 Batch 2200/2536 - Train Accuracy: 0.9896, Validation Accuracy: 0.6339, Loss: 0.0387
    Epoch  50 Batch 2300/2536 - Train Accuracy: 0.9495, Validation Accuracy: 0.6384, Loss: 0.1016
    Epoch  50 Batch 2400/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6362, Loss: 0.0632
    Epoch  50 Batch 2500/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6362, Loss: 0.0612
    Epoch  51 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6362, Loss: 0.0203
    Epoch  51 Batch  200/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6362, Loss: 0.0386
    Epoch  51 Batch  300/2536 - Train Accuracy: 0.9643, Validation Accuracy: 0.6384, Loss: 0.0511
    Epoch  51 Batch  400/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6362, Loss: 0.0670
    Epoch  51 Batch  500/2536 - Train Accuracy: 0.9500, Validation Accuracy: 0.6362, Loss: 0.0792
    Epoch  51 Batch  600/2536 - Train Accuracy: 0.9479, Validation Accuracy: 0.6362, Loss: 0.1028
    Epoch  51 Batch  700/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6384, Loss: 0.0433
    Epoch  51 Batch  800/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0757
    Epoch  51 Batch  900/2536 - Train Accuracy: 0.9777, Validation Accuracy: 0.6384, Loss: 0.0764
    Epoch  51 Batch 1000/2536 - Train Accuracy: 0.9104, Validation Accuracy: 0.6384, Loss: 0.1672
    Epoch  51 Batch 1100/2536 - Train Accuracy: 0.8951, Validation Accuracy: 0.6406, Loss: 0.1660
    Epoch  51 Batch 1200/2536 - Train Accuracy: 0.9219, Validation Accuracy: 0.6384, Loss: 0.1343
    Epoch  51 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0066
    Epoch  51 Batch 1400/2536 - Train Accuracy: 0.9583, Validation Accuracy: 0.6384, Loss: 0.1039
    Epoch  51 Batch 1500/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6384, Loss: 0.0363
    Epoch  51 Batch 1600/2536 - Train Accuracy: 0.9870, Validation Accuracy: 0.6384, Loss: 0.0293
    Epoch  51 Batch 1700/2536 - Train Accuracy: 0.9830, Validation Accuracy: 0.6384, Loss: 0.0414
    Epoch  51 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0221
    Epoch  51 Batch 1900/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6406, Loss: 0.0570
    Epoch  51 Batch 2000/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0339
    Epoch  51 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0055
    Epoch  51 Batch 2200/2536 - Train Accuracy: 0.9437, Validation Accuracy: 0.6429, Loss: 0.0440
    Epoch  51 Batch 2300/2536 - Train Accuracy: 0.9567, Validation Accuracy: 0.6384, Loss: 0.0927
    Epoch  51 Batch 2400/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6384, Loss: 0.0596
    Epoch  51 Batch 2500/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6362, Loss: 0.0662
    Epoch  52 Batch  100/2536 - Train Accuracy: 0.9801, Validation Accuracy: 0.6384, Loss: 0.0262
    Epoch  52 Batch  200/2536 - Train Accuracy: 0.9754, Validation Accuracy: 0.6406, Loss: 0.0389
    Epoch  52 Batch  300/2536 - Train Accuracy: 0.9754, Validation Accuracy: 0.6406, Loss: 0.0653
    Epoch  52 Batch  400/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0674
    Epoch  52 Batch  500/2536 - Train Accuracy: 0.9479, Validation Accuracy: 0.6384, Loss: 0.0730
    Epoch  52 Batch  600/2536 - Train Accuracy: 0.9563, Validation Accuracy: 0.6362, Loss: 0.0870
    Epoch  52 Batch  700/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6384, Loss: 0.0564
    Epoch  52 Batch  800/2536 - Train Accuracy: 0.9712, Validation Accuracy: 0.6384, Loss: 0.0784
    Epoch  52 Batch  900/2536 - Train Accuracy: 0.9754, Validation Accuracy: 0.6362, Loss: 0.0808
    Epoch  52 Batch 1000/2536 - Train Accuracy: 0.9146, Validation Accuracy: 0.6406, Loss: 0.1858
    Epoch  52 Batch 1100/2536 - Train Accuracy: 0.9286, Validation Accuracy: 0.6406, Loss: 0.1743
    Epoch  52 Batch 1200/2536 - Train Accuracy: 0.9395, Validation Accuracy: 0.6384, Loss: 0.1345
    Epoch  52 Batch 1300/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6384, Loss: 0.0039
    Epoch  52 Batch 1400/2536 - Train Accuracy: 0.9542, Validation Accuracy: 0.6362, Loss: 0.1036
    Epoch  52 Batch 1500/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6362, Loss: 0.0225
    Epoch  52 Batch 1600/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6384, Loss: 0.0240
    Epoch  52 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0173
    Epoch  52 Batch 1800/2536 - Train Accuracy: 0.9773, Validation Accuracy: 0.6384, Loss: 0.0206
    Epoch  52 Batch 1900/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6384, Loss: 0.0939
    Epoch  52 Batch 2000/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6384, Loss: 0.0559
    Epoch  52 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0067
    Epoch  52 Batch 2200/2536 - Train Accuracy: 0.9833, Validation Accuracy: 0.6406, Loss: 0.0380
    Epoch  52 Batch 2300/2536 - Train Accuracy: 0.9399, Validation Accuracy: 0.6362, Loss: 0.0808
    Epoch  52 Batch 2400/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6384, Loss: 0.0643
    Epoch  52 Batch 2500/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6406, Loss: 0.0652
    Epoch  53 Batch  100/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6384, Loss: 0.0220
    Epoch  53 Batch  200/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6384, Loss: 0.0329
    Epoch  53 Batch  300/2536 - Train Accuracy: 0.9799, Validation Accuracy: 0.6384, Loss: 0.0548
    Epoch  53 Batch  400/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6406, Loss: 0.0638
    Epoch  53 Batch  500/2536 - Train Accuracy: 0.9771, Validation Accuracy: 0.6362, Loss: 0.0646
    Epoch  53 Batch  600/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6362, Loss: 0.1011
    Epoch  53 Batch  700/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6384, Loss: 0.0621
    Epoch  53 Batch  800/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6362, Loss: 0.0666
    Epoch  53 Batch  900/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6384, Loss: 0.0805
    Epoch  53 Batch 1000/2536 - Train Accuracy: 0.9146, Validation Accuracy: 0.6406, Loss: 0.1708
    Epoch  53 Batch 1100/2536 - Train Accuracy: 0.9353, Validation Accuracy: 0.6384, Loss: 0.1240
    Epoch  53 Batch 1200/2536 - Train Accuracy: 0.9336, Validation Accuracy: 0.6362, Loss: 0.1284
    Epoch  53 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0029
    Epoch  53 Batch 1400/2536 - Train Accuracy: 0.9479, Validation Accuracy: 0.6384, Loss: 0.0818
    Epoch  53 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0196
    Epoch  53 Batch 1600/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6362, Loss: 0.0316
    Epoch  53 Batch 1700/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6339, Loss: 0.0212
    Epoch  53 Batch 1800/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6429, Loss: 0.0117
    Epoch  53 Batch 1900/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6362, Loss: 0.0510
    Epoch  53 Batch 2000/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6384, Loss: 0.0442
    Epoch  53 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0113
    Epoch  53 Batch 2200/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6384, Loss: 0.0518
    Epoch  53 Batch 2300/2536 - Train Accuracy: 0.9423, Validation Accuracy: 0.6339, Loss: 0.1194
    Epoch  53 Batch 2400/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6362, Loss: 0.0521
    Epoch  53 Batch 2500/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6362, Loss: 0.0629
    Epoch  54 Batch  100/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6384, Loss: 0.0202
    Epoch  54 Batch  200/2536 - Train Accuracy: 0.9888, Validation Accuracy: 0.6384, Loss: 0.0381
    Epoch  54 Batch  300/2536 - Train Accuracy: 0.9799, Validation Accuracy: 0.6384, Loss: 0.0620
    Epoch  54 Batch  400/2536 - Train Accuracy: 0.9336, Validation Accuracy: 0.6362, Loss: 0.0649
    Epoch  54 Batch  500/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6362, Loss: 0.0584
    Epoch  54 Batch  600/2536 - Train Accuracy: 0.9563, Validation Accuracy: 0.6362, Loss: 0.0747
    Epoch  54 Batch  700/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6362, Loss: 0.0415
    Epoch  54 Batch  800/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6384, Loss: 0.0683
    Epoch  54 Batch  900/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6406, Loss: 0.0667
    Epoch  54 Batch 1000/2536 - Train Accuracy: 0.9500, Validation Accuracy: 0.6384, Loss: 0.1033
    Epoch  54 Batch 1100/2536 - Train Accuracy: 0.9241, Validation Accuracy: 0.6384, Loss: 0.1352
    Epoch  54 Batch 1200/2536 - Train Accuracy: 0.9277, Validation Accuracy: 0.6384, Loss: 0.0848
    Epoch  54 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0041
    Epoch  54 Batch 1400/2536 - Train Accuracy: 0.9417, Validation Accuracy: 0.6362, Loss: 0.0794
    Epoch  54 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0236
    Epoch  54 Batch 1600/2536 - Train Accuracy: 0.9818, Validation Accuracy: 0.6362, Loss: 0.0198
    Epoch  54 Batch 1700/2536 - Train Accuracy: 0.9886, Validation Accuracy: 0.6339, Loss: 0.0189
    Epoch  54 Batch 1800/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6384, Loss: 0.0106
    Epoch  54 Batch 1900/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6339, Loss: 0.0519
    Epoch  54 Batch 2000/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6384, Loss: 0.0432
    Epoch  54 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0112
    Epoch  54 Batch 2200/2536 - Train Accuracy: 0.9812, Validation Accuracy: 0.6384, Loss: 0.0338
    Epoch  54 Batch 2300/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6362, Loss: 0.0843
    Epoch  54 Batch 2400/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0741
    Epoch  54 Batch 2500/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6384, Loss: 0.0567
    Epoch  55 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0127
    Epoch  55 Batch  200/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6406, Loss: 0.0432
    Epoch  55 Batch  300/2536 - Train Accuracy: 0.9821, Validation Accuracy: 0.6384, Loss: 0.0507
    Epoch  55 Batch  400/2536 - Train Accuracy: 0.9707, Validation Accuracy: 0.6406, Loss: 0.0672
    Epoch  55 Batch  500/2536 - Train Accuracy: 0.9771, Validation Accuracy: 0.6406, Loss: 0.0609
    Epoch  55 Batch  600/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6362, Loss: 0.0717
    Epoch  55 Batch  700/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6362, Loss: 0.0576
    Epoch  55 Batch  800/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6384, Loss: 0.0908
    Epoch  55 Batch  900/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0655
    Epoch  55 Batch 1000/2536 - Train Accuracy: 0.9208, Validation Accuracy: 0.6384, Loss: 0.1225
    Epoch  55 Batch 1100/2536 - Train Accuracy: 0.9353, Validation Accuracy: 0.6384, Loss: 0.1270
    Epoch  55 Batch 1200/2536 - Train Accuracy: 0.9629, Validation Accuracy: 0.6406, Loss: 0.0907
    Epoch  55 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0033
    Epoch  55 Batch 1400/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0863
    Epoch  55 Batch 1500/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6384, Loss: 0.0172
    Epoch  55 Batch 1600/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6384, Loss: 0.0354
    Epoch  55 Batch 1700/2536 - Train Accuracy: 0.9858, Validation Accuracy: 0.6362, Loss: 0.0424
    Epoch  55 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0205
    Epoch  55 Batch 1900/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6362, Loss: 0.0303
    Epoch  55 Batch 2000/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0305
    Epoch  55 Batch 2100/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6384, Loss: 0.0045
    Epoch  55 Batch 2200/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0411
    Epoch  55 Batch 2300/2536 - Train Accuracy: 0.9567, Validation Accuracy: 0.6362, Loss: 0.1097
    Epoch  55 Batch 2400/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6362, Loss: 0.0401
    Epoch  55 Batch 2500/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6362, Loss: 0.0569
    Epoch  56 Batch  100/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6362, Loss: 0.0161
    Epoch  56 Batch  200/2536 - Train Accuracy: 0.9821, Validation Accuracy: 0.6362, Loss: 0.0260
    Epoch  56 Batch  300/2536 - Train Accuracy: 0.9799, Validation Accuracy: 0.6339, Loss: 0.0700
    Epoch  56 Batch  400/2536 - Train Accuracy: 0.9648, Validation Accuracy: 0.6339, Loss: 0.0546
    Epoch  56 Batch  500/2536 - Train Accuracy: 0.9604, Validation Accuracy: 0.6339, Loss: 0.0640
    Epoch  56 Batch  600/2536 - Train Accuracy: 0.9708, Validation Accuracy: 0.6362, Loss: 0.0881
    Epoch  56 Batch  700/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6362, Loss: 0.0352
    Epoch  56 Batch  800/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0493
    Epoch  56 Batch  900/2536 - Train Accuracy: 0.9732, Validation Accuracy: 0.6384, Loss: 0.0584
    Epoch  56 Batch 1000/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6384, Loss: 0.1112
    Epoch  56 Batch 1100/2536 - Train Accuracy: 0.9330, Validation Accuracy: 0.6362, Loss: 0.1409
    Epoch  56 Batch 1200/2536 - Train Accuracy: 0.9395, Validation Accuracy: 0.6339, Loss: 0.0824
    Epoch  56 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0036
    Epoch  56 Batch 1400/2536 - Train Accuracy: 0.9604, Validation Accuracy: 0.6362, Loss: 0.0875
    Epoch  56 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0271
    Epoch  56 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0182
    Epoch  56 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0159
    Epoch  56 Batch 1800/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6362, Loss: 0.0157
    Epoch  56 Batch 1900/2536 - Train Accuracy: 0.9818, Validation Accuracy: 0.6362, Loss: 0.0313
    Epoch  56 Batch 2000/2536 - Train Accuracy: 0.9591, Validation Accuracy: 0.6339, Loss: 0.0300
    Epoch  56 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0036
    Epoch  56 Batch 2200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0386
    Epoch  56 Batch 2300/2536 - Train Accuracy: 0.9423, Validation Accuracy: 0.6339, Loss: 0.0874
    Epoch  56 Batch 2400/2536 - Train Accuracy: 0.9471, Validation Accuracy: 0.6339, Loss: 0.0357
    Epoch  56 Batch 2500/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6339, Loss: 0.0556
    Epoch  57 Batch  100/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6384, Loss: 0.0116
    Epoch  57 Batch  200/2536 - Train Accuracy: 0.9799, Validation Accuracy: 0.6362, Loss: 0.0346
    Epoch  57 Batch  300/2536 - Train Accuracy: 0.9754, Validation Accuracy: 0.6339, Loss: 0.0537
    Epoch  57 Batch  400/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6362, Loss: 0.0470
    Epoch  57 Batch  500/2536 - Train Accuracy: 0.9479, Validation Accuracy: 0.6384, Loss: 0.0485
    Epoch  57 Batch  600/2536 - Train Accuracy: 0.9563, Validation Accuracy: 0.6362, Loss: 0.0880
    Epoch  57 Batch  700/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6339, Loss: 0.0356
    Epoch  57 Batch  800/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6339, Loss: 0.0454
    Epoch  57 Batch  900/2536 - Train Accuracy: 0.9554, Validation Accuracy: 0.6339, Loss: 0.0702
    Epoch  57 Batch 1000/2536 - Train Accuracy: 0.9187, Validation Accuracy: 0.6362, Loss: 0.1307
    Epoch  57 Batch 1100/2536 - Train Accuracy: 0.9576, Validation Accuracy: 0.6362, Loss: 0.1393
    Epoch  57 Batch 1200/2536 - Train Accuracy: 0.9297, Validation Accuracy: 0.6384, Loss: 0.1071
    Epoch  57 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0035
    Epoch  57 Batch 1400/2536 - Train Accuracy: 0.9458, Validation Accuracy: 0.6339, Loss: 0.0947
    Epoch  57 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0155
    Epoch  57 Batch 1600/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6362, Loss: 0.0535
    Epoch  57 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0197
    Epoch  57 Batch 1800/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6384, Loss: 0.0098
    Epoch  57 Batch 1900/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6362, Loss: 0.0524
    Epoch  57 Batch 2000/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6362, Loss: 0.0327
    Epoch  57 Batch 2100/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6384, Loss: 0.0035
    Epoch  57 Batch 2200/2536 - Train Accuracy: 0.9750, Validation Accuracy: 0.6384, Loss: 0.0456
    Epoch  57 Batch 2300/2536 - Train Accuracy: 0.9519, Validation Accuracy: 0.6362, Loss: 0.0606
    Epoch  57 Batch 2400/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6384, Loss: 0.0431
    Epoch  57 Batch 2500/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6362, Loss: 0.0822
    Epoch  58 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0207
    Epoch  58 Batch  200/2536 - Train Accuracy: 0.9978, Validation Accuracy: 0.6362, Loss: 0.0505
    Epoch  58 Batch  300/2536 - Train Accuracy: 0.9888, Validation Accuracy: 0.6362, Loss: 0.0425
    Epoch  58 Batch  400/2536 - Train Accuracy: 0.9668, Validation Accuracy: 0.6362, Loss: 0.0697
    Epoch  58 Batch  500/2536 - Train Accuracy: 0.9771, Validation Accuracy: 0.6339, Loss: 0.0678
    Epoch  58 Batch  600/2536 - Train Accuracy: 0.9729, Validation Accuracy: 0.6362, Loss: 0.0937
    Epoch  58 Batch  700/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6339, Loss: 0.0480
    Epoch  58 Batch  800/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6339, Loss: 0.0628
    Epoch  58 Batch  900/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6339, Loss: 0.0606
    Epoch  58 Batch 1000/2536 - Train Accuracy: 0.9167, Validation Accuracy: 0.6362, Loss: 0.1465
    Epoch  58 Batch 1100/2536 - Train Accuracy: 0.9420, Validation Accuracy: 0.6362, Loss: 0.1180
    Epoch  58 Batch 1200/2536 - Train Accuracy: 0.9062, Validation Accuracy: 0.6384, Loss: 0.0938
    Epoch  58 Batch 1300/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0033
    Epoch  58 Batch 1400/2536 - Train Accuracy: 0.9896, Validation Accuracy: 0.6339, Loss: 0.0761
    Epoch  58 Batch 1500/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6339, Loss: 0.0178
    Epoch  58 Batch 1600/2536 - Train Accuracy: 0.9818, Validation Accuracy: 0.6384, Loss: 0.0153
    Epoch  58 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0180
    Epoch  58 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0211
    Epoch  58 Batch 1900/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6362, Loss: 0.0630
    Epoch  58 Batch 2000/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6362, Loss: 0.0438
    Epoch  58 Batch 2100/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6362, Loss: 0.0056
    Epoch  58 Batch 2200/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6362, Loss: 0.0401
    Epoch  58 Batch 2300/2536 - Train Accuracy: 0.9471, Validation Accuracy: 0.6339, Loss: 0.0660
    Epoch  58 Batch 2400/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0501
    Epoch  58 Batch 2500/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6339, Loss: 0.0540
    Epoch  59 Batch  100/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6362, Loss: 0.0211
    Epoch  59 Batch  200/2536 - Train Accuracy: 0.9978, Validation Accuracy: 0.6339, Loss: 0.0233
    Epoch  59 Batch  300/2536 - Train Accuracy: 0.9933, Validation Accuracy: 0.6362, Loss: 0.0406
    Epoch  59 Batch  400/2536 - Train Accuracy: 0.9980, Validation Accuracy: 0.6362, Loss: 0.0441
    Epoch  59 Batch  500/2536 - Train Accuracy: 0.9437, Validation Accuracy: 0.6339, Loss: 0.0345
    Epoch  59 Batch  600/2536 - Train Accuracy: 0.9542, Validation Accuracy: 0.6339, Loss: 0.0808
    Epoch  59 Batch  700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0466
    Epoch  59 Batch  800/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6339, Loss: 0.0442
    Epoch  59 Batch  900/2536 - Train Accuracy: 0.9665, Validation Accuracy: 0.6339, Loss: 0.0638
    Epoch  59 Batch 1000/2536 - Train Accuracy: 0.8896, Validation Accuracy: 0.6362, Loss: 0.1193
    Epoch  59 Batch 1100/2536 - Train Accuracy: 0.9420, Validation Accuracy: 0.6384, Loss: 0.1095
    Epoch  59 Batch 1200/2536 - Train Accuracy: 0.9590, Validation Accuracy: 0.6384, Loss: 0.0829
    Epoch  59 Batch 1300/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6339, Loss: 0.0054
    Epoch  59 Batch 1400/2536 - Train Accuracy: 0.9542, Validation Accuracy: 0.6339, Loss: 0.0758
    Epoch  59 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0221
    Epoch  59 Batch 1600/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6339, Loss: 0.0329
    Epoch  59 Batch 1700/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6384, Loss: 0.0366
    Epoch  59 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0164
    Epoch  59 Batch 1900/2536 - Train Accuracy: 0.9896, Validation Accuracy: 0.6339, Loss: 0.0408
    Epoch  59 Batch 2000/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0263
    Epoch  59 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0033
    Epoch  59 Batch 2200/2536 - Train Accuracy: 0.9854, Validation Accuracy: 0.6362, Loss: 0.0255
    Epoch  59 Batch 2300/2536 - Train Accuracy: 0.9567, Validation Accuracy: 0.6339, Loss: 0.0652
    Epoch  59 Batch 2400/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0370
    Epoch  59 Batch 2500/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6339, Loss: 0.0558
    Epoch  60 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0169
    Epoch  60 Batch  200/2536 - Train Accuracy: 0.9933, Validation Accuracy: 0.6362, Loss: 0.0369
    Epoch  60 Batch  300/2536 - Train Accuracy: 0.9911, Validation Accuracy: 0.6362, Loss: 0.0315
    Epoch  60 Batch  400/2536 - Train Accuracy: 0.9746, Validation Accuracy: 0.6339, Loss: 0.0363
    Epoch  60 Batch  500/2536 - Train Accuracy: 0.9646, Validation Accuracy: 0.6339, Loss: 0.0607
    Epoch  60 Batch  600/2536 - Train Accuracy: 0.9750, Validation Accuracy: 0.6362, Loss: 0.0681
    Epoch  60 Batch  700/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6339, Loss: 0.0435
    Epoch  60 Batch  800/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6339, Loss: 0.0532
    Epoch  60 Batch  900/2536 - Train Accuracy: 0.9955, Validation Accuracy: 0.6339, Loss: 0.0741
    Epoch  60 Batch 1000/2536 - Train Accuracy: 0.9333, Validation Accuracy: 0.6362, Loss: 0.1015
    Epoch  60 Batch 1100/2536 - Train Accuracy: 0.9665, Validation Accuracy: 0.6362, Loss: 0.1168
    Epoch  60 Batch 1200/2536 - Train Accuracy: 0.9453, Validation Accuracy: 0.6362, Loss: 0.0978
    Epoch  60 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0041
    Epoch  60 Batch 1400/2536 - Train Accuracy: 0.9604, Validation Accuracy: 0.6339, Loss: 0.0950
    Epoch  60 Batch 1500/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0194
    Epoch  60 Batch 1600/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6362, Loss: 0.0206
    Epoch  60 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0179
    Epoch  60 Batch 1800/2536 - Train Accuracy: 0.9801, Validation Accuracy: 0.6339, Loss: 0.0138
    Epoch  60 Batch 1900/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0434
    Epoch  60 Batch 2000/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6362, Loss: 0.0221
    Epoch  60 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0067
    Epoch  60 Batch 2200/2536 - Train Accuracy: 0.9938, Validation Accuracy: 0.6384, Loss: 0.0374
    Epoch  60 Batch 2300/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6339, Loss: 0.0655
    Epoch  60 Batch 2400/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0511
    Epoch  60 Batch 2500/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6339, Loss: 0.0558
    Epoch  61 Batch  100/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6362, Loss: 0.0114
    Epoch  61 Batch  200/2536 - Train Accuracy: 0.9911, Validation Accuracy: 0.6339, Loss: 0.0240
    Epoch  61 Batch  300/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6384, Loss: 0.0257
    Epoch  61 Batch  400/2536 - Train Accuracy: 0.9609, Validation Accuracy: 0.6362, Loss: 0.0753
    Epoch  61 Batch  500/2536 - Train Accuracy: 0.9750, Validation Accuracy: 0.6362, Loss: 0.0461
    Epoch  61 Batch  600/2536 - Train Accuracy: 0.9750, Validation Accuracy: 0.6339, Loss: 0.0605
    Epoch  61 Batch  700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0334
    Epoch  61 Batch  800/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6339, Loss: 0.0409
    Epoch  61 Batch  900/2536 - Train Accuracy: 0.9554, Validation Accuracy: 0.6362, Loss: 0.0515
    Epoch  61 Batch 1000/2536 - Train Accuracy: 0.9396, Validation Accuracy: 0.6362, Loss: 0.0857
    Epoch  61 Batch 1100/2536 - Train Accuracy: 0.9330, Validation Accuracy: 0.6406, Loss: 0.1085
    Epoch  61 Batch 1200/2536 - Train Accuracy: 0.9512, Validation Accuracy: 0.6362, Loss: 0.0929
    Epoch  61 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0029
    Epoch  61 Batch 1400/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6362, Loss: 0.0852
    Epoch  61 Batch 1500/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0181
    Epoch  61 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0253
    Epoch  61 Batch 1700/2536 - Train Accuracy: 0.9886, Validation Accuracy: 0.6339, Loss: 0.0157
    Epoch  61 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0109
    Epoch  61 Batch 1900/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6339, Loss: 0.0262
    Epoch  61 Batch 2000/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6362, Loss: 0.0271
    Epoch  61 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0080
    Epoch  61 Batch 2200/2536 - Train Accuracy: 0.9854, Validation Accuracy: 0.6339, Loss: 0.0380
    Epoch  61 Batch 2300/2536 - Train Accuracy: 0.9567, Validation Accuracy: 0.6362, Loss: 0.0655
    Epoch  61 Batch 2400/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6339, Loss: 0.0315
    Epoch  61 Batch 2500/2536 - Train Accuracy: 0.9639, Validation Accuracy: 0.6339, Loss: 0.0497
    Epoch  62 Batch  100/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6362, Loss: 0.0078
    Epoch  62 Batch  200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0358
    Epoch  62 Batch  300/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0325
    Epoch  62 Batch  400/2536 - Train Accuracy: 0.9355, Validation Accuracy: 0.6339, Loss: 0.0650
    Epoch  62 Batch  500/2536 - Train Accuracy: 0.9729, Validation Accuracy: 0.6362, Loss: 0.0410
    Epoch  62 Batch  600/2536 - Train Accuracy: 0.9708, Validation Accuracy: 0.6339, Loss: 0.0774
    Epoch  62 Batch  700/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6362, Loss: 0.0410
    Epoch  62 Batch  800/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6339, Loss: 0.0364
    Epoch  62 Batch  900/2536 - Train Accuracy: 0.9509, Validation Accuracy: 0.6362, Loss: 0.0521
    Epoch  62 Batch 1000/2536 - Train Accuracy: 0.9812, Validation Accuracy: 0.6406, Loss: 0.1085
    Epoch  62 Batch 1100/2536 - Train Accuracy: 0.9710, Validation Accuracy: 0.6406, Loss: 0.1004
    Epoch  62 Batch 1200/2536 - Train Accuracy: 0.9785, Validation Accuracy: 0.6384, Loss: 0.0957
    Epoch  62 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0016
    Epoch  62 Batch 1400/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6362, Loss: 0.0746
    Epoch  62 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0223
    Epoch  62 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0189
    Epoch  62 Batch 1700/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6362, Loss: 0.0097
    Epoch  62 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0111
    Epoch  62 Batch 1900/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6362, Loss: 0.0257
    Epoch  62 Batch 2000/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6362, Loss: 0.0297
    Epoch  62 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0028
    Epoch  62 Batch 2200/2536 - Train Accuracy: 0.9958, Validation Accuracy: 0.6362, Loss: 0.0555
    Epoch  62 Batch 2300/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6362, Loss: 0.0710
    Epoch  62 Batch 2400/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6384, Loss: 0.0476
    Epoch  62 Batch 2500/2536 - Train Accuracy: 0.9639, Validation Accuracy: 0.6362, Loss: 0.0485
    Epoch  63 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0297
    Epoch  63 Batch  200/2536 - Train Accuracy: 0.9955, Validation Accuracy: 0.6362, Loss: 0.0254
    Epoch  63 Batch  300/2536 - Train Accuracy: 0.9821, Validation Accuracy: 0.6384, Loss: 0.0452
    Epoch  63 Batch  400/2536 - Train Accuracy: 0.9668, Validation Accuracy: 0.6362, Loss: 0.0509
    Epoch  63 Batch  500/2536 - Train Accuracy: 0.9750, Validation Accuracy: 0.6339, Loss: 0.0404
    Epoch  63 Batch  600/2536 - Train Accuracy: 0.9771, Validation Accuracy: 0.6362, Loss: 0.0792
    Epoch  63 Batch  700/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6362, Loss: 0.0480
    Epoch  63 Batch  800/2536 - Train Accuracy: 0.9712, Validation Accuracy: 0.6339, Loss: 0.0400
    Epoch  63 Batch  900/2536 - Train Accuracy: 0.9442, Validation Accuracy: 0.6384, Loss: 0.0666
    Epoch  63 Batch 1000/2536 - Train Accuracy: 0.9042, Validation Accuracy: 0.6429, Loss: 0.0948
    Epoch  63 Batch 1100/2536 - Train Accuracy: 0.9821, Validation Accuracy: 0.6384, Loss: 0.0968
    Epoch  63 Batch 1200/2536 - Train Accuracy: 0.9355, Validation Accuracy: 0.6362, Loss: 0.0769
    Epoch  63 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0045
    Epoch  63 Batch 1400/2536 - Train Accuracy: 0.9646, Validation Accuracy: 0.6339, Loss: 0.0713
    Epoch  63 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0168
    Epoch  63 Batch 1600/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6384, Loss: 0.0372
    Epoch  63 Batch 1700/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6384, Loss: 0.0170
    Epoch  63 Batch 1800/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6362, Loss: 0.0161
    Epoch  63 Batch 1900/2536 - Train Accuracy: 0.9896, Validation Accuracy: 0.6384, Loss: 0.0336
    Epoch  63 Batch 2000/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0314
    Epoch  63 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0050
    Epoch  63 Batch 2200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0297
    Epoch  63 Batch 2300/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6339, Loss: 0.0635
    Epoch  63 Batch 2400/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0354
    Epoch  63 Batch 2500/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6339, Loss: 0.0797
    Epoch  64 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0156
    Epoch  64 Batch  200/2536 - Train Accuracy: 0.9888, Validation Accuracy: 0.6339, Loss: 0.0259
    Epoch  64 Batch  300/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6362, Loss: 0.0378
    Epoch  64 Batch  400/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0441
    Epoch  64 Batch  500/2536 - Train Accuracy: 0.9417, Validation Accuracy: 0.6362, Loss: 0.0467
    Epoch  73 Batch  400/2536 - Train Accuracy: 0.9805, Validation Accuracy: 0.6362, Loss: 0.0254
    Epoch  73 Batch  500/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6362, Loss: 0.0359
    Epoch  73 Batch  600/2536 - Train Accuracy: 0.9667, Validation Accuracy: 0.6339, Loss: 0.0562
    Epoch  73 Batch  700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0395
    Epoch  73 Batch  800/2536 - Train Accuracy: 0.9712, Validation Accuracy: 0.6362, Loss: 0.0613
    Epoch  73 Batch  900/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6362, Loss: 0.0318
    Epoch  73 Batch 1000/2536 - Train Accuracy: 0.9313, Validation Accuracy: 0.6362, Loss: 0.0638
    Epoch  73 Batch 1100/2536 - Train Accuracy: 0.9554, Validation Accuracy: 0.6362, Loss: 0.0990
    Epoch  73 Batch 1200/2536 - Train Accuracy: 0.9805, Validation Accuracy: 0.6384, Loss: 0.0593
    Epoch  73 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0023
    Epoch  73 Batch 1400/2536 - Train Accuracy: 0.9583, Validation Accuracy: 0.6362, Loss: 0.0575
    Epoch  73 Batch 1500/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0087
    Epoch  73 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0123
    Epoch  73 Batch 1700/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6339, Loss: 0.0167
    Epoch  73 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0049
    Epoch  73 Batch 1900/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6362, Loss: 0.0229
    Epoch  73 Batch 2000/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0204
    Epoch  73 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0023
    Epoch  73 Batch 2200/2536 - Train Accuracy: 0.9833, Validation Accuracy: 0.6384, Loss: 0.0329
    Epoch  73 Batch 2300/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6362, Loss: 0.0350
    Epoch  73 Batch 2400/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0164
    Epoch  73 Batch 2500/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6384, Loss: 0.0374
    Epoch  74 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0130
    Epoch  74 Batch  200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0192
    Epoch  74 Batch  300/2536 - Train Accuracy: 0.9911, Validation Accuracy: 0.6339, Loss: 0.0344
    Epoch  74 Batch  400/2536 - Train Accuracy: 0.9863, Validation Accuracy: 0.6339, Loss: 0.0399
    Epoch  74 Batch  500/2536 - Train Accuracy: 0.9958, Validation Accuracy: 0.6339, Loss: 0.0322
    Epoch  74 Batch  600/2536 - Train Accuracy: 0.9854, Validation Accuracy: 0.6339, Loss: 0.0364
    Epoch  74 Batch  700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0237
    Epoch  74 Batch  800/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6362, Loss: 0.0285
    Epoch  74 Batch  900/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6339, Loss: 0.0557
    Epoch  74 Batch 1000/2536 - Train Accuracy: 0.9479, Validation Accuracy: 0.6362, Loss: 0.0896
    Epoch  74 Batch 1100/2536 - Train Accuracy: 0.9710, Validation Accuracy: 0.6384, Loss: 0.0919
    Epoch  74 Batch 1200/2536 - Train Accuracy: 0.9746, Validation Accuracy: 0.6362, Loss: 0.0776
    Epoch  74 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0025
    Epoch  74 Batch 1400/2536 - Train Accuracy: 0.9771, Validation Accuracy: 0.6339, Loss: 0.0353
    Epoch  74 Batch 1500/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0123
    Epoch  74 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0109
    Epoch  74 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0084
    Epoch  74 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0089
    Epoch  74 Batch 1900/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6362, Loss: 0.0179
    Epoch  74 Batch 2000/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6406, Loss: 0.0203
    Epoch  74 Batch 2100/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6384, Loss: 0.0033
    Epoch  74 Batch 2200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0259
    Epoch  74 Batch 2300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0405
    Epoch  74 Batch 2400/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0257
    Epoch  74 Batch 2500/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6362, Loss: 0.0266
    Epoch  75 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0146
    Epoch  75 Batch  200/2536 - Train Accuracy: 0.9955, Validation Accuracy: 0.6362, Loss: 0.0181
    Epoch  75 Batch  300/2536 - Train Accuracy: 0.9955, Validation Accuracy: 0.6362, Loss: 0.0173
    Epoch  75 Batch  400/2536 - Train Accuracy: 0.9434, Validation Accuracy: 0.6362, Loss: 0.0337
    Epoch  75 Batch  500/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6362, Loss: 0.0492
    Epoch  75 Batch  600/2536 - Train Accuracy: 0.9854, Validation Accuracy: 0.6339, Loss: 0.0421
    Epoch  75 Batch  700/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6362, Loss: 0.0320
    Epoch  75 Batch  800/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6339, Loss: 0.0234
    Epoch  75 Batch  900/2536 - Train Accuracy: 0.9710, Validation Accuracy: 0.6384, Loss: 0.0399
    Epoch  75 Batch 1000/2536 - Train Accuracy: 0.9417, Validation Accuracy: 0.6384, Loss: 0.0791
    Epoch  75 Batch 1100/2536 - Train Accuracy: 0.9754, Validation Accuracy: 0.6384, Loss: 0.0963
    Epoch  75 Batch 1200/2536 - Train Accuracy: 0.9824, Validation Accuracy: 0.6384, Loss: 0.0702
    Epoch  75 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0040
    Epoch  75 Batch 1400/2536 - Train Accuracy: 0.9750, Validation Accuracy: 0.6362, Loss: 0.0315
    Epoch  75 Batch 1500/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6339, Loss: 0.0140
    Epoch  75 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0309
    Epoch  75 Batch 1700/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6384, Loss: 0.0125
    Epoch  75 Batch 1800/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6384, Loss: 0.0123
    Epoch  75 Batch 1900/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0163
    Epoch  75 Batch 2000/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6362, Loss: 0.0180
    Epoch  75 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0046
    Epoch  75 Batch 2200/2536 - Train Accuracy: 0.9938, Validation Accuracy: 0.6384, Loss: 0.0165
    Epoch  75 Batch 2300/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6362, Loss: 0.0434
    Epoch  75 Batch 2400/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0256
    Epoch  75 Batch 2500/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6362, Loss: 0.0453
    Epoch  76 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0082
    Epoch  76 Batch  200/2536 - Train Accuracy: 0.9955, Validation Accuracy: 0.6384, Loss: 0.0091
    Epoch  76 Batch  300/2536 - Train Accuracy: 0.9621, Validation Accuracy: 0.6362, Loss: 0.0370
    Epoch  76 Batch  400/2536 - Train Accuracy: 0.9648, Validation Accuracy: 0.6362, Loss: 0.0355
    Epoch  76 Batch  500/2536 - Train Accuracy: 0.9979, Validation Accuracy: 0.6339, Loss: 0.0331
    Epoch  76 Batch  600/2536 - Train Accuracy: 0.9771, Validation Accuracy: 0.6339, Loss: 0.0291
    Epoch  76 Batch  700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0251
    Epoch  76 Batch  800/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6339, Loss: 0.0279
    Epoch  76 Batch  900/2536 - Train Accuracy: 0.9554, Validation Accuracy: 0.6339, Loss: 0.0514
    Epoch  76 Batch 1000/2536 - Train Accuracy: 0.9542, Validation Accuracy: 0.6362, Loss: 0.0563
    Epoch  76 Batch 1100/2536 - Train Accuracy: 0.9554, Validation Accuracy: 0.6362, Loss: 0.0603
    Epoch  76 Batch 1200/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6384, Loss: 0.0814
    Epoch  76 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0035
    Epoch  76 Batch 1400/2536 - Train Accuracy: 0.9729, Validation Accuracy: 0.6339, Loss: 0.0530
    Epoch  76 Batch 1500/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0143
    Epoch  76 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0112
    Epoch  76 Batch 1700/2536 - Train Accuracy: 0.9886, Validation Accuracy: 0.6339, Loss: 0.0243
    Epoch  76 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0103
    Epoch  76 Batch 1900/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0270
    Epoch  76 Batch 2000/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0152
    Epoch  76 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0026
    Epoch  76 Batch 2200/2536 - Train Accuracy: 0.9833, Validation Accuracy: 0.6362, Loss: 0.0156
    Epoch  76 Batch 2300/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6362, Loss: 0.0494
    Epoch  76 Batch 2400/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0179
    Epoch  76 Batch 2500/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6339, Loss: 0.0287
    Epoch  77 Batch  100/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6339, Loss: 0.0067
    Epoch  77 Batch  200/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6339, Loss: 0.0233
    Epoch  77 Batch  300/2536 - Train Accuracy: 0.9710, Validation Accuracy: 0.6339, Loss: 0.0414
    Epoch  77 Batch  400/2536 - Train Accuracy: 0.9727, Validation Accuracy: 0.6362, Loss: 0.0317
    Epoch  77 Batch  500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0285
    Epoch  77 Batch  600/2536 - Train Accuracy: 0.9708, Validation Accuracy: 0.6362, Loss: 0.0715
    Epoch  77 Batch  700/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6339, Loss: 0.0251
    Epoch  77 Batch  800/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6339, Loss: 0.0418
    Epoch  77 Batch  900/2536 - Train Accuracy: 0.9888, Validation Accuracy: 0.6362, Loss: 0.0332
    Epoch  77 Batch 1000/2536 - Train Accuracy: 0.9583, Validation Accuracy: 0.6384, Loss: 0.0583
    Epoch  77 Batch 1100/2536 - Train Accuracy: 0.9621, Validation Accuracy: 0.6384, Loss: 0.0813
    Epoch  77 Batch 1200/2536 - Train Accuracy: 0.9746, Validation Accuracy: 0.6406, Loss: 0.0510
    Epoch  77 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0023
    Epoch  77 Batch 1400/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6384, Loss: 0.0414
    Epoch  77 Batch 1500/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6339, Loss: 0.0097
    Epoch  77 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0137
    Epoch  77 Batch 1700/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6362, Loss: 0.0060
    Epoch  77 Batch 1800/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6384, Loss: 0.0096
    Epoch  77 Batch 1900/2536 - Train Accuracy: 0.9870, Validation Accuracy: 0.6362, Loss: 0.0280
    Epoch  77 Batch 2000/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6406, Loss: 0.0124
    Epoch  77 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0039
    Epoch  77 Batch 2200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0151
    Epoch  77 Batch 2300/2536 - Train Accuracy: 0.9567, Validation Accuracy: 0.6406, Loss: 0.0326
    Epoch  77 Batch 2400/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6339, Loss: 0.0255
    Epoch  77 Batch 2500/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6339, Loss: 0.0350
    Epoch  78 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6339, Loss: 0.0077
    Epoch  78 Batch  200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0208
    Epoch  78 Batch  300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0259
    Epoch  78 Batch  400/2536 - Train Accuracy: 0.9766, Validation Accuracy: 0.6406, Loss: 0.0384
    Epoch  78 Batch  500/2536 - Train Accuracy: 0.9854, Validation Accuracy: 0.6339, Loss: 0.0302
    Epoch  78 Batch  600/2536 - Train Accuracy: 0.9896, Validation Accuracy: 0.6362, Loss: 0.0336
    Epoch  78 Batch  700/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0162
    Epoch  78 Batch  800/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6362, Loss: 0.0335
    Epoch  78 Batch  900/2536 - Train Accuracy: 0.9799, Validation Accuracy: 0.6339, Loss: 0.0318
    Epoch  78 Batch 1000/2536 - Train Accuracy: 0.9396, Validation Accuracy: 0.6339, Loss: 0.0683
    Epoch  78 Batch 1100/2536 - Train Accuracy: 0.9888, Validation Accuracy: 0.6362, Loss: 0.0631
    Epoch  78 Batch 1200/2536 - Train Accuracy: 0.9668, Validation Accuracy: 0.6384, Loss: 0.0672
    Epoch  78 Batch 1300/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6362, Loss: 0.0021
    Epoch  78 Batch 1400/2536 - Train Accuracy: 0.9604, Validation Accuracy: 0.6362, Loss: 0.0577
    Epoch  78 Batch 1500/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6406, Loss: 0.0079
    Epoch  78 Batch 1600/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6384, Loss: 0.0360
    Epoch  78 Batch 1700/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6429, Loss: 0.0143
    Epoch  78 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0058
    Epoch  78 Batch 1900/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6429, Loss: 0.0236
    Epoch  78 Batch 2000/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6384, Loss: 0.0140
    Epoch  78 Batch 2100/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6429, Loss: 0.0022
    Epoch  78 Batch 2200/2536 - Train Accuracy: 0.9917, Validation Accuracy: 0.6384, Loss: 0.0250
    Epoch  78 Batch 2300/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0624
    Epoch  78 Batch 2400/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6384, Loss: 0.0155
    Epoch  78 Batch 2500/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6384, Loss: 0.0374
    Epoch  79 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0143
    Epoch  79 Batch  200/2536 - Train Accuracy: 0.9933, Validation Accuracy: 0.6384, Loss: 0.0147
    Epoch  79 Batch  300/2536 - Train Accuracy: 0.9754, Validation Accuracy: 0.6384, Loss: 0.0225
    Epoch  79 Batch  400/2536 - Train Accuracy: 0.9980, Validation Accuracy: 0.6339, Loss: 0.0224
    Epoch  79 Batch  500/2536 - Train Accuracy: 0.9833, Validation Accuracy: 0.6339, Loss: 0.0198
    Epoch  79 Batch  600/2536 - Train Accuracy: 0.9812, Validation Accuracy: 0.6362, Loss: 0.0534
    Epoch  79 Batch  700/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0231
    Epoch  79 Batch  800/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6339, Loss: 0.0330
    Epoch  79 Batch  900/2536 - Train Accuracy: 0.9978, Validation Accuracy: 0.6339, Loss: 0.0456
    Epoch  79 Batch 1000/2536 - Train Accuracy: 0.9437, Validation Accuracy: 0.6339, Loss: 0.0517
    Epoch  79 Batch 1100/2536 - Train Accuracy: 0.9665, Validation Accuracy: 0.6339, Loss: 0.0661
    Epoch  79 Batch 1200/2536 - Train Accuracy: 0.9727, Validation Accuracy: 0.6339, Loss: 0.0552
    Epoch  79 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0025
    Epoch  79 Batch 1400/2536 - Train Accuracy: 0.9667, Validation Accuracy: 0.6362, Loss: 0.0501
    Epoch  79 Batch 1500/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6339, Loss: 0.0084
    Epoch  79 Batch 1600/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6339, Loss: 0.0075
    Epoch  79 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0065
    Epoch  79 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0093
    Epoch  79 Batch 1900/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0211
    Epoch  79 Batch 2000/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0217
    Epoch  79 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6429, Loss: 0.0017
    Epoch  79 Batch 2200/2536 - Train Accuracy: 0.9938, Validation Accuracy: 0.6406, Loss: 0.0293
    Epoch  79 Batch 2300/2536 - Train Accuracy: 0.9639, Validation Accuracy: 0.6362, Loss: 0.0446
    Epoch  79 Batch 2400/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6339, Loss: 0.0207
    Epoch  79 Batch 2500/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6362, Loss: 0.0554
    Epoch  80 Batch  100/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6362, Loss: 0.0153
    Epoch  80 Batch  200/2536 - Train Accuracy: 0.9955, Validation Accuracy: 0.6339, Loss: 0.0187
    Epoch  80 Batch  300/2536 - Train Accuracy: 0.9777, Validation Accuracy: 0.6362, Loss: 0.0224
    Epoch  80 Batch  400/2536 - Train Accuracy: 0.9707, Validation Accuracy: 0.6339, Loss: 0.0284
    Epoch  80 Batch  500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0224
    Epoch  80 Batch  600/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0412
    Epoch  80 Batch  700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0225
    Epoch  80 Batch  800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0334
    Epoch  80 Batch  900/2536 - Train Accuracy: 0.9821, Validation Accuracy: 0.6339, Loss: 0.0315
    Epoch  80 Batch 1000/2536 - Train Accuracy: 0.9667, Validation Accuracy: 0.6339, Loss: 0.0534
    Epoch  80 Batch 1100/2536 - Train Accuracy: 0.9777, Validation Accuracy: 0.6362, Loss: 0.0566
    Epoch  80 Batch 1200/2536 - Train Accuracy: 0.9629, Validation Accuracy: 0.6384, Loss: 0.0522
    Epoch  80 Batch 1300/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0022
    Epoch  80 Batch 1400/2536 - Train Accuracy: 0.9854, Validation Accuracy: 0.6339, Loss: 0.0527
    Epoch  80 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0138
    Epoch  80 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0095
    Epoch  80 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0194
    Epoch  80 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0099
    Epoch  80 Batch 1900/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0265
    Epoch  80 Batch 2000/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0152
    Epoch  80 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6429, Loss: 0.0050
    Epoch  80 Batch 2200/2536 - Train Accuracy: 0.9646, Validation Accuracy: 0.6384, Loss: 0.0220
    Epoch  80 Batch 2300/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6362, Loss: 0.0532
    Epoch  80 Batch 2400/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0261
    Epoch  80 Batch 2500/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6339, Loss: 0.0294
    Epoch  81 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0120
    Epoch  81 Batch  200/2536 - Train Accuracy: 0.9888, Validation Accuracy: 0.6384, Loss: 0.0279
    Epoch  81 Batch  300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0237
    Epoch  81 Batch  400/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6339, Loss: 0.0238
    Epoch  81 Batch  500/2536 - Train Accuracy: 0.9854, Validation Accuracy: 0.6362, Loss: 0.0368
    Epoch  81 Batch  600/2536 - Train Accuracy: 0.9479, Validation Accuracy: 0.6362, Loss: 0.0408
    Epoch  81 Batch  700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0253
    Epoch  81 Batch  800/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6339, Loss: 0.0383
    Epoch  81 Batch  900/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6362, Loss: 0.0411
    Epoch  81 Batch 1000/2536 - Train Accuracy: 0.9771, Validation Accuracy: 0.6362, Loss: 0.0694
    Epoch  81 Batch 1100/2536 - Train Accuracy: 0.9598, Validation Accuracy: 0.6384, Loss: 0.0604
    Epoch  81 Batch 1200/2536 - Train Accuracy: 0.9766, Validation Accuracy: 0.6384, Loss: 0.0695
    Epoch  81 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0080
    Epoch  81 Batch 1400/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6339, Loss: 0.0397
    Epoch  81 Batch 1500/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6362, Loss: 0.0153
    Epoch  81 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0155
    Epoch  81 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0089
    Epoch  81 Batch 1800/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6362, Loss: 0.0164
    Epoch  81 Batch 1900/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0204
    Epoch  81 Batch 2000/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6339, Loss: 0.0176
    Epoch  81 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0051
    Epoch  81 Batch 2200/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6384, Loss: 0.0154
    Epoch  81 Batch 2300/2536 - Train Accuracy: 0.9471, Validation Accuracy: 0.6362, Loss: 0.0313
    Epoch  81 Batch 2400/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6362, Loss: 0.0186
    Epoch  81 Batch 2500/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0425
    Epoch  82 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0128
    Epoch  82 Batch  200/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6339, Loss: 0.0112
    Epoch  82 Batch  300/2536 - Train Accuracy: 0.9777, Validation Accuracy: 0.6362, Loss: 0.0204
    Epoch  82 Batch  400/2536 - Train Accuracy: 0.9531, Validation Accuracy: 0.6339, Loss: 0.0212
    Epoch  82 Batch  500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0309
    Epoch  82 Batch  600/2536 - Train Accuracy: 0.9854, Validation Accuracy: 0.6339, Loss: 0.0342
    Epoch  82 Batch  700/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6362, Loss: 0.0228
    Epoch  82 Batch  800/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6339, Loss: 0.0281
    Epoch  82 Batch  900/2536 - Train Accuracy: 0.9933, Validation Accuracy: 0.6339, Loss: 0.0254
    Epoch  82 Batch 1000/2536 - Train Accuracy: 0.9646, Validation Accuracy: 0.6362, Loss: 0.0915
    Epoch  82 Batch 1100/2536 - Train Accuracy: 0.9844, Validation Accuracy: 0.6384, Loss: 0.0917
    Epoch  82 Batch 1200/2536 - Train Accuracy: 0.9805, Validation Accuracy: 0.6384, Loss: 0.0578
    Epoch  82 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0009
    Epoch  82 Batch 1400/2536 - Train Accuracy: 0.9917, Validation Accuracy: 0.6339, Loss: 0.0446
    Epoch  82 Batch 1500/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6339, Loss: 0.0104
    Epoch  82 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0092
    Epoch  82 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0078
    Epoch  82 Batch 1800/2536 - Train Accuracy: 0.9858, Validation Accuracy: 0.6362, Loss: 0.0056
    Epoch  82 Batch 1900/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0087
    Epoch  82 Batch 2000/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0175
    Epoch  82 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0019
    Epoch  82 Batch 2200/2536 - Train Accuracy: 0.9896, Validation Accuracy: 0.6384, Loss: 0.0209
    Epoch  82 Batch 2300/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6362, Loss: 0.0459
    Epoch  82 Batch 2400/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0352
    Epoch  82 Batch 2500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0391
    Epoch  83 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0027
    Epoch  83 Batch  200/2536 - Train Accuracy: 0.9978, Validation Accuracy: 0.6384, Loss: 0.0077
    Epoch  83 Batch  300/2536 - Train Accuracy: 0.9888, Validation Accuracy: 0.6384, Loss: 0.0459
    Epoch  83 Batch  400/2536 - Train Accuracy: 0.9980, Validation Accuracy: 0.6406, Loss: 0.0274
    Epoch  83 Batch  500/2536 - Train Accuracy: 0.9979, Validation Accuracy: 0.6384, Loss: 0.0243
    Epoch  83 Batch  600/2536 - Train Accuracy: 0.9646, Validation Accuracy: 0.6384, Loss: 0.0280
    Epoch  83 Batch  700/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0249
    Epoch  83 Batch  800/2536 - Train Accuracy: 0.9760, Validation Accuracy: 0.6384, Loss: 0.0209
    Epoch  83 Batch  900/2536 - Train Accuracy: 0.9888, Validation Accuracy: 0.6362, Loss: 0.0252
    Epoch  83 Batch 1000/2536 - Train Accuracy: 0.9521, Validation Accuracy: 0.6384, Loss: 0.0604
    Epoch  83 Batch 1100/2536 - Train Accuracy: 0.9621, Validation Accuracy: 0.6362, Loss: 0.0512
    Epoch  83 Batch 1200/2536 - Train Accuracy: 0.9707, Validation Accuracy: 0.6362, Loss: 0.0459
    Epoch  83 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0022
    Epoch  83 Batch 1400/2536 - Train Accuracy: 0.9708, Validation Accuracy: 0.6362, Loss: 0.0355
    Epoch  83 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0166
    Epoch  83 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0082
    Epoch  83 Batch 1700/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6339, Loss: 0.0150
    Epoch  83 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0081
    Epoch  83 Batch 1900/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6384, Loss: 0.0217
    Epoch  83 Batch 2000/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6384, Loss: 0.0237
    Epoch  83 Batch 2100/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6406, Loss: 0.0048
    Epoch  83 Batch 2200/2536 - Train Accuracy: 0.9979, Validation Accuracy: 0.6406, Loss: 0.0200
    Epoch  83 Batch 2300/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0525
    Epoch  83 Batch 2400/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0284
    Epoch  83 Batch 2500/2536 - Train Accuracy: 0.9736, Validation Accuracy: 0.6384, Loss: 0.0401
    Epoch  84 Batch  100/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6406, Loss: 0.0067
    Epoch  84 Batch  200/2536 - Train Accuracy: 0.9777, Validation Accuracy: 0.6384, Loss: 0.0066
    Epoch  84 Batch  300/2536 - Train Accuracy: 0.9978, Validation Accuracy: 0.6362, Loss: 0.0219
    Epoch  84 Batch  400/2536 - Train Accuracy: 0.9727, Validation Accuracy: 0.6384, Loss: 0.0367
    Epoch  84 Batch  500/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6339, Loss: 0.0591
    Epoch  84 Batch  600/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6339, Loss: 0.0425
    Epoch  84 Batch  700/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6339, Loss: 0.0121
    Epoch  84 Batch  800/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0170
    Epoch  84 Batch  900/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6339, Loss: 0.0210
    Epoch  84 Batch 1000/2536 - Train Accuracy: 0.9375, Validation Accuracy: 0.6362, Loss: 0.0708
    Epoch  84 Batch 1100/2536 - Train Accuracy: 0.9732, Validation Accuracy: 0.6384, Loss: 0.0641
    Epoch  84 Batch 1200/2536 - Train Accuracy: 0.9570, Validation Accuracy: 0.6362, Loss: 0.0590
    Epoch  84 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0015
    Epoch  84 Batch 1400/2536 - Train Accuracy: 0.9917, Validation Accuracy: 0.6339, Loss: 0.0302
    Epoch  84 Batch 1500/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6339, Loss: 0.0134
    Epoch  84 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0209
    Epoch  84 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0281
    Epoch  84 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0092
    Epoch  84 Batch 1900/2536 - Train Accuracy: 0.9870, Validation Accuracy: 0.6384, Loss: 0.0181
    Epoch  84 Batch 2000/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6384, Loss: 0.0084
    Epoch  84 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0019
    Epoch  84 Batch 2200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6429, Loss: 0.0277
    Epoch  84 Batch 2300/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6362, Loss: 0.0370
    Epoch  84 Batch 2400/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0080
    Epoch  84 Batch 2500/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0294
    Epoch  85 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0086
    Epoch  85 Batch  200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0181
    Epoch  85 Batch  300/2536 - Train Accuracy: 0.9821, Validation Accuracy: 0.6384, Loss: 0.0218
    Epoch  85 Batch  400/2536 - Train Accuracy: 0.9980, Validation Accuracy: 0.6384, Loss: 0.0233
    Epoch  85 Batch  500/2536 - Train Accuracy: 0.9979, Validation Accuracy: 0.6384, Loss: 0.0265
    Epoch  85 Batch  600/2536 - Train Accuracy: 0.9938, Validation Accuracy: 0.6362, Loss: 0.0341
    Epoch  85 Batch  700/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0150
    Epoch  85 Batch  800/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6339, Loss: 0.0169
    Epoch  85 Batch  900/2536 - Train Accuracy: 0.9710, Validation Accuracy: 0.6339, Loss: 0.0379
    Epoch  85 Batch 1000/2536 - Train Accuracy: 0.9646, Validation Accuracy: 0.6362, Loss: 0.0630
    Epoch  85 Batch 1100/2536 - Train Accuracy: 0.9487, Validation Accuracy: 0.6362, Loss: 0.0683
    Epoch  85 Batch 1200/2536 - Train Accuracy: 0.9805, Validation Accuracy: 0.6362, Loss: 0.0593
    Epoch  85 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0032
    Epoch  85 Batch 1400/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6339, Loss: 0.0526
    Epoch  85 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0129
    Epoch  85 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0161
    Epoch  85 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0153
    Epoch  85 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0082
    Epoch  85 Batch 1900/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6406, Loss: 0.0117
    Epoch  85 Batch 2000/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6384, Loss: 0.0263
    Epoch  85 Batch 2100/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6406, Loss: 0.0011
    Epoch  85 Batch 2200/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6406, Loss: 0.0228
    Epoch  85 Batch 2300/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0450
    Epoch  85 Batch 2400/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0195
    Epoch  85 Batch 2500/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6362, Loss: 0.0285
    Epoch  86 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0050
    Epoch  86 Batch  200/2536 - Train Accuracy: 0.9955, Validation Accuracy: 0.6339, Loss: 0.0165
    Epoch  86 Batch  300/2536 - Train Accuracy: 0.9911, Validation Accuracy: 0.6339, Loss: 0.0267
    Epoch  86 Batch  400/2536 - Train Accuracy: 0.9863, Validation Accuracy: 0.6339, Loss: 0.0302
    Epoch  86 Batch  500/2536 - Train Accuracy: 0.9812, Validation Accuracy: 0.6339, Loss: 0.0292
    Epoch  86 Batch  600/2536 - Train Accuracy: 0.9896, Validation Accuracy: 0.6362, Loss: 0.0287
    Epoch  86 Batch  700/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6362, Loss: 0.0272
    Epoch  86 Batch  800/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6339, Loss: 0.0220
    Epoch  86 Batch  900/2536 - Train Accuracy: 0.9911, Validation Accuracy: 0.6362, Loss: 0.0281
    Epoch  86 Batch 1000/2536 - Train Accuracy: 0.9917, Validation Accuracy: 0.6339, Loss: 0.0857
    Epoch  86 Batch 1100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0527
    Epoch  86 Batch 1200/2536 - Train Accuracy: 0.9648, Validation Accuracy: 0.6384, Loss: 0.0529
    Epoch  86 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0014
    Epoch  86 Batch 1400/2536 - Train Accuracy: 0.9854, Validation Accuracy: 0.6362, Loss: 0.0522
    Epoch  86 Batch 1500/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6339, Loss: 0.0127
    Epoch  86 Batch 1600/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6362, Loss: 0.0086
    Epoch  86 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0116
    Epoch  86 Batch 1800/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6339, Loss: 0.0137
    Epoch  86 Batch 1900/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6339, Loss: 0.0176
    Epoch  86 Batch 2000/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0149
    Epoch  86 Batch 2100/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6406, Loss: 0.0007
    Epoch  86 Batch 2200/2536 - Train Accuracy: 0.9938, Validation Accuracy: 0.6406, Loss: 0.0170
    Epoch  86 Batch 2300/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6362, Loss: 0.0360
    Epoch  86 Batch 2400/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6362, Loss: 0.0206
    Epoch  86 Batch 2500/2536 - Train Accuracy: 0.9663, Validation Accuracy: 0.6362, Loss: 0.0448
    Epoch  87 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0135
    Epoch  87 Batch  200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0122
    Epoch  87 Batch  300/2536 - Train Accuracy: 0.9777, Validation Accuracy: 0.6362, Loss: 0.0249
    Epoch  87 Batch  400/2536 - Train Accuracy: 0.9785, Validation Accuracy: 0.6362, Loss: 0.0190
    Epoch  87 Batch  500/2536 - Train Accuracy: 0.9958, Validation Accuracy: 0.6339, Loss: 0.0238
    Epoch  87 Batch  600/2536 - Train Accuracy: 0.9771, Validation Accuracy: 0.6339, Loss: 0.0375
    Epoch  87 Batch  700/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0441
    Epoch  87 Batch  800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0185
    Epoch  87 Batch  900/2536 - Train Accuracy: 0.9911, Validation Accuracy: 0.6339, Loss: 0.0392
    Epoch  87 Batch 1000/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0406
    Epoch  87 Batch 1100/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6362, Loss: 0.0549
    Epoch  87 Batch 1200/2536 - Train Accuracy: 0.9531, Validation Accuracy: 0.6362, Loss: 0.0654
    Epoch  87 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0025
    Epoch  87 Batch 1400/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6362, Loss: 0.0454
    Epoch  87 Batch 1500/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0098
    Epoch  87 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0104
    Epoch  87 Batch 1700/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6339, Loss: 0.0118
    Epoch  87 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0060
    Epoch  87 Batch 1900/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6362, Loss: 0.0232
    Epoch  87 Batch 2000/2536 - Train Accuracy: 0.9808, Validation Accuracy: 0.6362, Loss: 0.0184
    Epoch  87 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0027
    Epoch  87 Batch 2200/2536 - Train Accuracy: 0.9958, Validation Accuracy: 0.6384, Loss: 0.0218
    Epoch  87 Batch 2300/2536 - Train Accuracy: 0.9615, Validation Accuracy: 0.6362, Loss: 0.0383
    Epoch  87 Batch 2400/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0201
    Epoch  87 Batch 2500/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6362, Loss: 0.0318
    Epoch  88 Batch  100/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6362, Loss: 0.0089
    Epoch  88 Batch  200/2536 - Train Accuracy: 0.9955, Validation Accuracy: 0.6406, Loss: 0.0109
    Epoch  88 Batch  300/2536 - Train Accuracy: 0.9911, Validation Accuracy: 0.6362, Loss: 0.0297
    Epoch  88 Batch  400/2536 - Train Accuracy: 0.9883, Validation Accuracy: 0.6362, Loss: 0.0331
    Epoch  88 Batch  500/2536 - Train Accuracy: 0.9938, Validation Accuracy: 0.6339, Loss: 0.0265
    Epoch  88 Batch  600/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6362, Loss: 0.0302
    Epoch  88 Batch  700/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0141
    Epoch  88 Batch  800/2536 - Train Accuracy: 0.9784, Validation Accuracy: 0.6362, Loss: 0.0139
    Epoch  88 Batch  900/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6384, Loss: 0.0302
    Epoch  88 Batch 1000/2536 - Train Accuracy: 0.9667, Validation Accuracy: 0.6384, Loss: 0.0528
    Epoch  88 Batch 1100/2536 - Train Accuracy: 0.9531, Validation Accuracy: 0.6406, Loss: 0.0517
    Epoch  88 Batch 1200/2536 - Train Accuracy: 0.9609, Validation Accuracy: 0.6384, Loss: 0.0488
    Epoch  88 Batch 1300/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6384, Loss: 0.0005
    Epoch  88 Batch 1400/2536 - Train Accuracy: 0.9750, Validation Accuracy: 0.6362, Loss: 0.0551
    Epoch  88 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0089
    Epoch  88 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0068
    Epoch  88 Batch 1700/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6406, Loss: 0.0150
    Epoch  88 Batch 1800/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6362, Loss: 0.0121
    Epoch  88 Batch 1900/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6384, Loss: 0.0183
    Epoch  88 Batch 2000/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6406, Loss: 0.0200
    Epoch  88 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0056
    Epoch  88 Batch 2200/2536 - Train Accuracy: 0.9958, Validation Accuracy: 0.6406, Loss: 0.0262
    Epoch  88 Batch 2300/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6384, Loss: 0.0334
    Epoch  88 Batch 2400/2536 - Train Accuracy: 0.9880, Validation Accuracy: 0.6362, Loss: 0.0118
    Epoch  88 Batch 2500/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0307
    Epoch  89 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0039
    Epoch  89 Batch  200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0080
    Epoch  89 Batch  300/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6384, Loss: 0.0231
    Epoch  89 Batch  400/2536 - Train Accuracy: 0.9902, Validation Accuracy: 0.6384, Loss: 0.0433
    Epoch  89 Batch  500/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0361
    Epoch  89 Batch  600/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0317
    Epoch  89 Batch  700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0236
    Epoch  89 Batch  800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0265
    Epoch  89 Batch  900/2536 - Train Accuracy: 0.9821, Validation Accuracy: 0.6384, Loss: 0.0262
    Epoch  89 Batch 1000/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0527
    Epoch  89 Batch 1100/2536 - Train Accuracy: 0.9978, Validation Accuracy: 0.6384, Loss: 0.0476
    Epoch  89 Batch 1200/2536 - Train Accuracy: 0.9590, Validation Accuracy: 0.6362, Loss: 0.0642
    Epoch  89 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0013
    Epoch  89 Batch 1400/2536 - Train Accuracy: 0.9646, Validation Accuracy: 0.6362, Loss: 0.0576
    Epoch  89 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0101
    Epoch  89 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0176
    Epoch  89 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0076
    Epoch  89 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0097
    Epoch  89 Batch 1900/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0272
    Epoch  89 Batch 2000/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0233
    Epoch  89 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0034
    Epoch  89 Batch 2200/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6384, Loss: 0.0159
    Epoch  89 Batch 2300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0268
    Epoch  89 Batch 2400/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0138
    Epoch  89 Batch 2500/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6362, Loss: 0.0331
    Epoch  90 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0096
    Epoch  90 Batch  200/2536 - Train Accuracy: 0.9933, Validation Accuracy: 0.6384, Loss: 0.0296
    Epoch  90 Batch  300/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6362, Loss: 0.0208
    Epoch  90 Batch  400/2536 - Train Accuracy: 0.9805, Validation Accuracy: 0.6384, Loss: 0.0443
    Epoch  90 Batch  500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0208
    Epoch  90 Batch  600/2536 - Train Accuracy: 0.9875, Validation Accuracy: 0.6406, Loss: 0.0256
    Epoch  90 Batch  700/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0309
    Epoch  90 Batch  800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0181
    Epoch  90 Batch  900/2536 - Train Accuracy: 0.9933, Validation Accuracy: 0.6384, Loss: 0.0283
    Epoch  90 Batch 1000/2536 - Train Accuracy: 0.9812, Validation Accuracy: 0.6384, Loss: 0.0465
    Epoch  90 Batch 1100/2536 - Train Accuracy: 0.9777, Validation Accuracy: 0.6384, Loss: 0.0634
    Epoch  90 Batch 1200/2536 - Train Accuracy: 0.9980, Validation Accuracy: 0.6362, Loss: 0.0375
    Epoch  90 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0052
    Epoch  90 Batch 1400/2536 - Train Accuracy: 0.9646, Validation Accuracy: 0.6384, Loss: 0.0462
    Epoch  90 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0105
    Epoch  90 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0084
    Epoch  90 Batch 1700/2536 - Train Accuracy: 0.9886, Validation Accuracy: 0.6406, Loss: 0.0117
    Epoch  90 Batch 1800/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6429, Loss: 0.0047
    Epoch  90 Batch 1900/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0323
    Epoch  90 Batch 2000/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0143
    Epoch  90 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0022
    Epoch  90 Batch 2200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0098
    Epoch  90 Batch 2300/2536 - Train Accuracy: 0.9639, Validation Accuracy: 0.6384, Loss: 0.0271
    Epoch  90 Batch 2400/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6384, Loss: 0.0174
    Epoch  90 Batch 2500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0344
    Epoch  91 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6384, Loss: 0.0049
    Epoch  91 Batch  200/2536 - Train Accuracy: 0.9933, Validation Accuracy: 0.6384, Loss: 0.0174
    Epoch  91 Batch  300/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6384, Loss: 0.0375
    Epoch  91 Batch  400/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0375
    Epoch  91 Batch  500/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6362, Loss: 0.0272
    Epoch  91 Batch  600/2536 - Train Accuracy: 0.9604, Validation Accuracy: 0.6362, Loss: 0.0294
    Epoch  91 Batch  700/2536 - Train Accuracy: 0.9928, Validation Accuracy: 0.6362, Loss: 0.0324
    Epoch  91 Batch  800/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6362, Loss: 0.0196
    Epoch  91 Batch  900/2536 - Train Accuracy: 0.9754, Validation Accuracy: 0.6362, Loss: 0.0215
    Epoch  91 Batch 1000/2536 - Train Accuracy: 0.9938, Validation Accuracy: 0.6384, Loss: 0.0430
    Epoch  91 Batch 1100/2536 - Train Accuracy: 0.9621, Validation Accuracy: 0.6384, Loss: 0.0507
    Epoch  91 Batch 1200/2536 - Train Accuracy: 0.9785, Validation Accuracy: 0.6406, Loss: 0.0430
    Epoch  91 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0019
    Epoch  91 Batch 1400/2536 - Train Accuracy: 0.9708, Validation Accuracy: 0.6362, Loss: 0.0349
    Epoch  91 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0095
    Epoch  91 Batch 1600/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6384, Loss: 0.0077
    Epoch  91 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0091
    Epoch  91 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0089
    Epoch  91 Batch 1900/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6362, Loss: 0.0152
    Epoch  91 Batch 2000/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6429, Loss: 0.0091
    Epoch  91 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0144
    Epoch  91 Batch 2200/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0183
    Epoch  91 Batch 2300/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6384, Loss: 0.0308
    Epoch  91 Batch 2400/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0140
    Epoch  91 Batch 2500/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6384, Loss: 0.0232
    Epoch  92 Batch  100/2536 - Train Accuracy: 0.9915, Validation Accuracy: 0.6384, Loss: 0.0182
    Epoch  92 Batch  200/2536 - Train Accuracy: 0.9978, Validation Accuracy: 0.6384, Loss: 0.0152
    Epoch  92 Batch  300/2536 - Train Accuracy: 0.9911, Validation Accuracy: 0.6384, Loss: 0.0200
    Epoch  92 Batch  400/2536 - Train Accuracy: 0.9922, Validation Accuracy: 0.6384, Loss: 0.0206
    Epoch  92 Batch  500/2536 - Train Accuracy: 0.9812, Validation Accuracy: 0.6384, Loss: 0.0149
    Epoch  92 Batch  600/2536 - Train Accuracy: 0.9708, Validation Accuracy: 0.6406, Loss: 0.0410
    Epoch  92 Batch  700/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6384, Loss: 0.0248
    Epoch  92 Batch  800/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0165
    Epoch  92 Batch  900/2536 - Train Accuracy: 0.9933, Validation Accuracy: 0.6384, Loss: 0.0334
    Epoch  92 Batch 1000/2536 - Train Accuracy: 0.9896, Validation Accuracy: 0.6406, Loss: 0.0433
    Epoch  92 Batch 1100/2536 - Train Accuracy: 0.9643, Validation Accuracy: 0.6406, Loss: 0.0642
    Epoch  92 Batch 1200/2536 - Train Accuracy: 0.9688, Validation Accuracy: 0.6406, Loss: 0.0545
    Epoch  92 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0047
    Epoch  92 Batch 1400/2536 - Train Accuracy: 0.9625, Validation Accuracy: 0.6362, Loss: 0.0648
    Epoch  92 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0159
    Epoch  92 Batch 1600/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6362, Loss: 0.0079
    Epoch  92 Batch 1700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0081
    Epoch  92 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0087
    Epoch  92 Batch 1900/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6362, Loss: 0.0131
    Epoch  92 Batch 2000/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0212
    Epoch  92 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0012
    Epoch  92 Batch 2200/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0303
    Epoch  92 Batch 2300/2536 - Train Accuracy: 0.9832, Validation Accuracy: 0.6362, Loss: 0.0305
    Epoch  92 Batch 2400/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0256
    Epoch  92 Batch 2500/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0491
    Epoch  93 Batch  100/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6406, Loss: 0.0090
    Epoch  93 Batch  200/2536 - Train Accuracy: 0.9866, Validation Accuracy: 0.6406, Loss: 0.0206
    Epoch  93 Batch  300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0192
    Epoch  93 Batch  400/2536 - Train Accuracy: 0.9883, Validation Accuracy: 0.6384, Loss: 0.0221
    Epoch  93 Batch  500/2536 - Train Accuracy: 0.9979, Validation Accuracy: 0.6384, Loss: 0.0286
    Epoch  93 Batch  600/2536 - Train Accuracy: 0.9917, Validation Accuracy: 0.6384, Loss: 0.0428
    Epoch  93 Batch  700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0097
    Epoch  93 Batch  800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0215
    Epoch  93 Batch  900/2536 - Train Accuracy: 0.9911, Validation Accuracy: 0.6384, Loss: 0.0246
    Epoch  93 Batch 1000/2536 - Train Accuracy: 0.9604, Validation Accuracy: 0.6406, Loss: 0.0483
    Epoch  93 Batch 1100/2536 - Train Accuracy: 0.9911, Validation Accuracy: 0.6384, Loss: 0.0513
    Epoch  93 Batch 1200/2536 - Train Accuracy: 0.9590, Validation Accuracy: 0.6384, Loss: 0.0424
    Epoch  93 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0014
    Epoch  93 Batch 1400/2536 - Train Accuracy: 0.9979, Validation Accuracy: 0.6406, Loss: 0.0312
    Epoch  93 Batch 1500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0073
    Epoch  93 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0102
    Epoch  93 Batch 1700/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6384, Loss: 0.0124
    Epoch  93 Batch 1800/2536 - Train Accuracy: 0.9943, Validation Accuracy: 0.6362, Loss: 0.0112
    Epoch  93 Batch 1900/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0161
    Epoch  93 Batch 2000/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6362, Loss: 0.0089
    Epoch  93 Batch 2100/2536 - Train Accuracy: 0.9974, Validation Accuracy: 0.6384, Loss: 0.0018
    Epoch  93 Batch 2200/2536 - Train Accuracy: 0.9938, Validation Accuracy: 0.6406, Loss: 0.0146
    Epoch  93 Batch 2300/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6384, Loss: 0.0250
    Epoch  93 Batch 2400/2536 - Train Accuracy: 0.9952, Validation Accuracy: 0.6362, Loss: 0.0338
    Epoch  93 Batch 2500/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6362, Loss: 0.0461
    Epoch  94 Batch  100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0050
    Epoch  94 Batch  200/2536 - Train Accuracy: 0.9888, Validation Accuracy: 0.6384, Loss: 0.0153
    Epoch  94 Batch  300/2536 - Train Accuracy: 0.9821, Validation Accuracy: 0.6384, Loss: 0.0288
    Epoch  94 Batch  400/2536 - Train Accuracy: 0.9668, Validation Accuracy: 0.6384, Loss: 0.0459
    Epoch  94 Batch  500/2536 - Train Accuracy: 0.9958, Validation Accuracy: 0.6384, Loss: 0.0217
    Epoch  94 Batch  600/2536 - Train Accuracy: 0.9958, Validation Accuracy: 0.6384, Loss: 0.0242
    Epoch  94 Batch  700/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0260
    Epoch  94 Batch  800/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6384, Loss: 0.0196
    Epoch  94 Batch  900/2536 - Train Accuracy: 0.9732, Validation Accuracy: 0.6406, Loss: 0.0368
    Epoch  94 Batch 1000/2536 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0361
    Epoch  94 Batch 1100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6406, Loss: 0.0482
    Epoch  94 Batch 1200/2536 - Train Accuracy: 0.9766, Validation Accuracy: 0.6384, Loss: 0.0509
    Epoch  94 Batch 1300/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0012
    Epoch  94 Batch 1400/2536 - Train Accuracy: 0.9729, Validation Accuracy: 0.6384, Loss: 0.0335
    Epoch  94 Batch 1500/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6384, Loss: 0.0161
    Epoch  94 Batch 1600/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0089
    Epoch  94 Batch 1700/2536 - Train Accuracy: 0.9972, Validation Accuracy: 0.6362, Loss: 0.0166
    Epoch  94 Batch 1800/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0077
    Epoch  94 Batch 1900/2536 - Train Accuracy: 0.9948, Validation Accuracy: 0.6362, Loss: 0.0086
    Epoch  94 Batch 2000/2536 - Train Accuracy: 0.9976, Validation Accuracy: 0.6362, Loss: 0.0154
    Epoch  94 Batch 2100/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0055
    Epoch  94 Batch 2200/2536 - Train Accuracy: 0.9979, Validation Accuracy: 0.6384, Loss: 0.0126
    Epoch  94 Batch 2300/2536 - Train Accuracy: 0.9856, Validation Accuracy: 0.6384, Loss: 0.0289
    Epoch  94 Batch 2400/2536 - Train Accuracy: 0.9904, Validation Accuracy: 0.6384, Loss: 0.0330
    Epoch  94 Batch 2500/2536 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0121


### Evaluate LSTM Net Only


```python
speaker_id, lexicon = list(lexicons.items())[0]
print("List of Speeches:", len(lexicon.speeches))
lexicon.evaluate_testset()
```

    List of Speeches: 240
    Speech Results:
    Average Candidate Transcript Accuracy: nan
    Average Seq2Seq Model Accuracy: nan
    


    /root/miniconda3/envs/tf-gpu/lib/python3.5/site-packages/numpy/core/_methods.py:59: RuntimeWarning: Mean of empty slice.
      warnings.warn("Mean of empty slice.", RuntimeWarning)
    /root/miniconda3/envs/tf-gpu/lib/python3.5/site-packages/numpy/core/_methods.py:70: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)



```python
import helper 
# Save parameters for checkpoint
speaker_id, lexicon = list(lexicons.items())[0]
helper.save_params(lexicon.cache_dir)
```


```python
import tensorflow as tf
import numpy as np
import helper
speaker_id, lexicon = list(lexicons.items())[0]
_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params(lexicon.cache_dir)
```
