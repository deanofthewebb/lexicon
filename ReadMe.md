
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
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=os.path.join(os.getcwd(),'Lexicon-e94eff39fad7.json')
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
    Roughly the number of unique words: 57965
    Number of sentences: 27540
    Average number of words in a sentence: 24.005228758169935
    
    Transcript sentences 0 to 10:
     Merrick
     Would it not be better for me to send these
    papers by a messenger to your house?"
    
    "No; I'll take them myself
     No one will rob me
    " And then the door
    swung open and, chuckling in his usual whimsical fashion, Uncle John
    came out, wearing his salt-and-pepper suit and stuffing; a bundle of
    papers into his inside pocket
    
    
    The Major stared at him haughtily, but made no attempt to openly
    recognize the man
     Uncle John gave a start, laughed, and then walked
    away briskly, throwing a hasty "good-bye" to the obsequious banker,
    who followed him out, bowing low
    
    
    The Major returned to his office with a grave face, and sat for the
    best part of three hours in a brown study
     Then he took his hat and
    went home
    
    
    Patsy asked anxiously if anything had happened, when she saw his face;
    but the Major shook his head
    
    
    Uncle John arrived just in time for dinner, in a very genial mood,
    and he and Patsy kept up a lively conversation at the table while the
    Major looked stern every time he caught the little man's eye
    
    Ground Truth sentences 0 to 10:
     Merrick
     Would it not be better for me to send these
    papers by a messenger to your house?"
    
    "No; I'll take them myself
     No one will rob me
    " And then the door
    swung open and, chuckling in his usual whimsical fashion, Uncle John
    came out, wearing his salt-and-pepper suit and stuffing; a bundle of
    papers into his inside pocket
    
    
    The Major stared at him haughtily, but made no attempt to openly
    recognize the man
     Uncle John gave a start, laughed, and then walked
    away briskly, throwing a hasty "good-bye" to the obsequious banker,
    who followed him out, bowing low
    
    
    The Major returned to his office with a grave face, and sat for the
    best part of three hours in a brown study
     Then he took his hat and
    went home
    
    
    Patsy asked anxiously if anything had happened, when she saw his face;
    but the Major shook his head
    
    
    Uncle John arrived just in time for dinner, in a very genial mood,
    and he and Patsy kept up a lively conversation at the table while the
    Major looked stern every time he caught the little man's eye
    
    Dataset Stats
    Roughly the number of unique words: 58051
    Number of sentences: 27600
    Average number of words in a sentence: 24.12873188405797
    
    Transcript sentences 0 to 10:
     Merrick
     Would it not be better for me to send these
    papers by a messenger to your house?"
    
    "No; I'll take them myself
     No one will rob me
    " And then the door
    swung open and, chuckling in his usual whimsical fashion, Uncle John
    came out, wearing his salt-and-pepper suit and stuffing; a bundle of
    papers into his inside pocket
    
    
    The Major stared at him haughtily, but made no attempt to openly
    recognize the man
     Uncle John gave a start, laughed, and then walked
    away briskly, throwing a hasty "good-bye" to the obsequious banker,
    who followed him out, bowing low
    
    
    The Major returned to his office with a grave face, and sat for the
    best part of three hours in a brown study
     Then he took his hat and
    went home
    
    
    Patsy asked anxiously if anything had happened, when she saw his face;
    but the Major shook his head
    
    
    Uncle John arrived just in time for dinner, in a very genial mood,
    and he and Patsy kept up a lively conversation at the table while the
    Major looked stern every time he caught the little man's eye
    
    Ground Truth sentences 0 to 10:
     Merrick
     Would it not be better for me to send these
    papers by a messenger to your house?"
    
    "No; I'll take them myself
     No one will rob me
    " And then the door
    swung open and, chuckling in his usual whimsical fashion, Uncle John
    came out, wearing his salt-and-pepper suit and stuffing; a bundle of
    papers into his inside pocket
    
    
    The Major stared at him haughtily, but made no attempt to openly
    recognize the man
     Uncle John gave a start, laughed, and then walked
    away briskly, throwing a hasty "good-bye" to the obsequious banker,
    who followed him out, bowing low
    
    
    The Major returned to his office with a grave face, and sat for the
    best part of three hours in a brown study
     Then he took his hat and
    went home
    
    
    Patsy asked anxiously if anything had happened, when she saw his face;
    but the Major shook his head
    
    
    Uncle John arrived just in time for dinner, in a very genial mood,
    and he and Patsy kept up a lively conversation at the table while the
    Major looked stern every time he caught the little man's eye
    


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

    Word|Freq:
    ('project', 'gutenbergtm')|1095
    ('project', 'gutenberg')|1014
    ('greater', 'part')|532
    ('captain', 'nemo')|452
    ('united', 'states')|407
    ('great', 'britain')|385
    ('uncle', 'john')|364
    ('gold', 'silver')|337
    ('let', 'us')|331
    ('of', 'course')|328
    ('new', 'york')|310
    ('old', 'man')|306
    ('gutenbergtm', 'electronic')|306
    ('mr', 'bounderby')|294
    ('public', 'domain')|293
    ('every', 'one')|291
    ('young', 'man')|284
    ('mrs', 'sparsit')|282
    ('one', 'day')|281
    ('one', 'another')|280
    ('archive', 'foundation')|279
    ('gutenberg', 'literary')|279
    ('literary', 'archive')|279
    ('dont', 'know')|275
    ('electronic', 'works')|272
    ('per', 'cent')|263
    ('could', 'see')|262
    ('ned', 'land')|254
    ('good', 'deal')|247
    ('two', 'three')|240
    ('set', 'forth')|225
    ('years', 'ago')|220
    ('old', 'woman')|219
    ('you', 'may')|218
    ('it', 'would')|207
    ('the', 'first')|206
    ('next', 'day')|201
    ('long', 'time')|200
    ('said', 'mrs')|199
    ('said', 'mr')|198
    ('of', 'the')|198
    ('first', 'time')|196
    ('every', 'day')|193
    ('one', 'thing')|193
    ('small', 'print')|189
    ('men', 'women')|187
    ('electronic', 'work')|187
    ('every', 'man')|182
    ('mr', 'gradgrind')|173
    ('it', 'may')|172



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

    
    Listing the words that can follow after 'greater':
     dict_keys(['valuable', 'smaller', 'gehenna', 'desolation', 'dexterity', 'opportunities', 'woe', 'returns', 'perithous', 'tenant', 'honor', 'indeed', 'agony', 'glorious', 'latter', 'account', 'divinity', 'sum', 'indignation', 'sin', 'advantage', 'transgression', 'flourish', 'necessary', 'teacher', 'activity', 'fortune', 'whole', 'rank', 'beauty', 'abundance', 'cost', 'disorders', 'sums', 'told', 'opening', 'action', 'parsimony', 'claim', 'worlds', 'beginning', 'convenience', 'labourers', 'change', 'supply', 'require', 'equal', 'found', 'weal', 'confidence', 'expected', 'knave', 'scarcity', 'quantity', 'gift', 'thoughts', 'trade', 'insult', 'deviation', 'stock', 'second', 'prince', 'extent', 'great', 'semblance', 'america', 'zeal', 'solidarity', 'clerk', 'want', 'among', 'rum', 'sun', 'riches', 'lesser', 'wealth', 'proportion', 'importation', 'slaves', 'liberty', 'grew', 'sorrow', 'intelligence', 'it', 'moment', 'need', 'inferiority', 'difference', 'vessels', 'as', 'writings', 'number', 'sometimes', 'men', 'sadness', 'little', 'clearness', 'distinctness', 'cheap', 'ones', 'rapidity', 'capital', 'countries', 'boldness', 'thirst', 'balance', 'crop', 'danger', 'shall', 'eloquence', 'never', 'whatever', 'peace', 'power', 'general', 'cheapness', 'usual', 'fast', 'warmth', 'titan', 'tartness', 'relevance', 'come', 'enthusiasm', 'body', 'sensation', 'and', 'dole', 'prospective', 'length', 'injustice', 'perpendicular', 'worldly', 'mans', 'depth', 'glory', 'otherwise', 'calamity', 'he', 'splendour', 'costage', 'heartiness', 'difficulties', 'salaries', 'frequently', 'gladness', 'influence', 'gain', 'gold', 'professed', 'melodies', 'strength', 'weight', 'desire', 'every', 'nails', 'present', 'incorporation', 'simplicity', 'environment', 'contained', 'wisdom', 'former', 'exportation', 'understanding', 'still', 'return', 'made', 'fortitude', 'therefore', 'produce', 'confusion', 'like', 'actually', 'force', 'guilds', 'remoteness', 'favour', 'france', 'well', 'alterations', 'use', 'delectation', 'already', 'if', 'than', 'peril', 'strain', 'life', 'at', 'range', 'sanctity', 'freedom', 'service', 'wrong', 'depths', 'herein', 'importance', 'goods', 'mountains', 'the', 'mass', 'rapture', 'content', 'business', 'sovereign', 'wonder', 'comfort', 'fame', 'art', 'things', 'mystery', 'far', 'subconscious', 'fiercer', 'either', 'without', 'continuing', 'restoration', 'rent', 'love', 'annual', 'portion', 'seems', 'land', 'effort', 'chance', 'impersonal', 'malversation', 'diligence', 'london', 'gifts', 'mastery', 'talents', 'to', 'waxen', 'augmentation', 'pasture', 'money', 'superabundance', 'upon', 'parts', 'quantities', 'vitality', 'such', 'ever', 'brewery', 'ii', 'capacities', 'believed', 'demand', 'name', 'pleasure', 'corn', 'strides', 'personal', 'steps', 'fire', 'economy', 'dilatation', 'no', 'rice', 'practicality', 'yet', 'place', 'would', 'view', 'death', 'variation', 'difficulty', 'shame', 'original', 'renown', 'amount', 'energy', 'revenue', 'antipathy', 'fund', 'expense', 'might', 'poets', 'triumph', 'circulation', 'universal', 'vanquished', 'field', 'fury', 'reduction', 'haste', 'favours', 'pressure', 'latitude', 'taint', 'less', 'circumstances', 'dangers', 'left', 'extensive', 'lawe', 'brightness', 'haytime', 'though', 'offence', 'lights', 'authority', 'crown', 'anyone', 'real', 'consequence', 'windbag', 'value', 'mind', 'saving', 'expression', 'wiser', 'share', 'facility', 'contemporary', 'enduring', 'point', 'degree', 'honourable', 'sufficient', 'foregoing', 'care', 'its', 'surplus', 'horn', 'pomp', 'attractions', 'favours\x94', 'group', 'numbers', 'higher', 'first', 'satisfaction', 'advantages', 'imprudence', 'dawn', 'africa', 'fear', 'lasting', 'tax', 'ruritania', 'crime', 'could', 'variety', 'english', 'ease', 'rich', 'but', 'age', 'employs', 'thing', 'annoyance', 'inconveniency', 'time', 'kindness', 'evil', 'greater', 'perhaps', 'must', 'degrees', 'producing', 'leader', 'many', 'hope', 'admiration', 'interest', 'taxes', 'individual', 'none', 'frequency', 'success', 'harm', 'dignity', 'injudicious', 'heat', 'one', 'went', 'common', 'in', 'abroad', 'formerly', 'evils', 'done', 'modern', 'pain', 'past', 'height', 'honour', 'frequent', 'ordinary', 'security', 'approximation', 'levying', 'trespass', 'fault', 'profusion', 'price', 'slumbers', 'jefferies', 'togetherness', 'pride', 'pieces', 'speed', 'loss', 'reprobate', 'ned', 'stocks', 'rights', 'violence', 'silence', 'benefit', 'distress', 'fixed', 'effect', 'almost', 'end', 'grief', 'singleness', 'acorn', 'suited', 'competition', 'velocity', 'obstacles', 'rise', 'profit', 'seignorage', 'goodness', 'grace', 'encountering', 'cultivation', 'events', 'distance', 'part'])
    
    Listing 20 most frequent words to come after 'greater':
     [('part', 532), ('quantity', 105), ('number', 50), ('proportion', 43), ('value', 24), ('smaller', 16), ('share', 16), ('greater', 16), ('less', 12), ('profit', 11), ('capital', 9), ('importance', 9), ('the', 9), ('revenue', 9), ('surplus', 8), ('degree', 7), ('variety', 7), ('distance', 7), ('sum', 6), ('stock', 6)]



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

    Waiting for operation to complete...
    
    
    RE-RANKED Results: 
     {'go go do you hear': 0.85315002696588638, 'go do you here': 0.86520871338434524, 'I go do you hear': 0.81552877281792457, 'go do here': 0.85310941742492696, 'do you here': 0.75866525587625799, 'go do you hear': 0.77847528909333052, 'goat do you hear': 0.85976193998940287, 'do you hear': 0.75895958994515234, 'goat do you here': 0.85946760592050853, 'I go do you here': 0.81523443874903023}
    
    
    ORIGINAL Transcript: 
    'goat do you here' 
    with a confidence_score of: 0.9545454978942871
    
    
    RE-RANKED Transcript: 
    'go do you here' 
    with a confidence_score of: 0.8652087133843452
    
    
    GROUND TRUTH TRANSCRIPT: 
    GO DO YOU HEAR
    
    The original transcript was RE-RANKED. The transcripts do not match!
    Differences between original and re-ranked:  ['goat']
    
    
    The original transcript DOES NOT MATCH ground truth.
    Differences between original and ground truth:  ['here', 'goat']
    
    
    The RE_RANKED transcript DOES NOT MATCH ground truth.
    Differences between Reranked and ground truth:  ['here']
    
    
    
    
    ORIGINAL Edit Distance: 
    4
    RE-RANKED Edit Distance: 
    2
    
    
    Waiting for operation to complete...
    
    
    RE-RANKED Results: 
     {'at this moment of the whole soul of the old man seem centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.87474450767040257, 'at this moment of the whole soul of the Old Man scene centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.86880395710468294, 'at this moment the whole soul of the old man seems centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.88751497417688374, 'at this moment of the whole soul of the old man seemed centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.87742958217859279, 'at this moment the whole soul of the old man seem centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.88437801450490949, 'at this moment to the whole soul of the old man seemed centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.88942871093750009, 'at this moment to the whole soul of the Old Man scene centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.87938940227031703, 'at this moment the whole soul of the Old Man scene centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.87849078625440591, 'at this moment to the whole soul of the old man seem centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.88532995283603666, 'at this moment the whole soul of the old man seemed centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.88711463958024983}
    
    
    ORIGINAL Transcript: 
    'at this moment of the whole soul of the old man seemed centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry' 
    with a confidence_score of: 0.9498937726020813
    
    
    RE-RANKED Transcript: 
    'at this moment to the whole soul of the old man seemed centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry' 
    with a confidence_score of: 0.8894287109375001
    
    
    GROUND TRUTH TRANSCRIPT: 
    AT THIS MOMENT THE WHOLE SOUL OF THE OLD MAN SEEMED CENTRED IN HIS EYES WHICH BECAME BLOODSHOT THE VEINS OF THE THROAT SWELLED HIS CHEEKS AND TEMPLES BECAME PURPLE AS THOUGH HE WAS STRUCK WITH EPILEPSY NOTHING WAS WANTING TO COMPLETE THIS BUT THE UTTERANCE OF A CRY
    
    No reranking was performed. The transcripts match!
    
    
    The original transcript DOES NOT MATCH ground truth.
    Differences between original and ground truth:  ['centered']
    
    
    The RE_RANKED transcript DOES NOT MATCH ground truth.
    Differences between Reranked and ground truth:  ['centered']
    
    
    
    
    ORIGINAL Edit Distance: 
    4
    RE-RANKED Edit Distance: 
    4
    
    
    Waiting for operation to complete...
    
    
    RE-RANKED Results: 
     {'Devon he rushed towards the old man and made him inhaler powerful restorative': 0.72487533390522008, 'deveny rushed towards the old man and made him and Halo powerful restorative': 0.74758408963680267, 'deveney rushed towards the old man and made him in Halo powerful restorative': 0.73207941949367528, 'Devon he rushed towards the old man and made him and Halo powerful restorative': 0.7115083783864975, 'Devin he rushed towards the old man and made him in Halo powerful restorative': 0.74116749465465548, 'Devin he rushed towards the old man and made him and Halo powerful restorative': 0.76070690453052525, 'Devon he rushed towards the old man and made him in Halo powerful restorative': 0.69701785147190098, 'deveney Rush towards the old man and made him and Halo powerful restorative': 0.69918780922889712, 'deveny Rush towards the old man and made him and Halo powerful restorative': 0.69918780922889712, 'deveney rushed towards the old man and made him and Halo powerful restorative': 0.74758408963680267}
    
    
    ORIGINAL Transcript: 
    'Devon he rushed towards the old man and made him inhaler powerful restorative' 
    with a confidence_score of: 0.7925808429718018
    
    
    RE-RANKED Transcript: 
    'Devin he rushed towards the old man and made him and Halo powerful restorative' 
    with a confidence_score of: 0.7607069045305253
    
    
    GROUND TRUTH TRANSCRIPT: 
    D'AVRIGNY RUSHED TOWARDS THE OLD MAN AND MADE HIM INHALE A POWERFUL RESTORATIVE
    
    The original transcript was RE-RANKED. The transcripts do not match!
    Differences between original and re-ranked:  ['devon', 'inhaler']
    
    
    The original transcript DOES NOT MATCH ground truth.
    Differences between original and ground truth:  ['devon', 'he', 'inhaler']
    
    
    The RE_RANKED transcript DOES NOT MATCH ground truth.
    Differences between Reranked and ground truth:  ['devin', 'he', 'halo']
    
    
    
    
    ORIGINAL Edit Distance: 
    9
    RE-RANKED Edit Distance: 
    13
    
    
    Waiting for operation to complete...
    
    
    RE-RANKED Results: 
     {"but in less than five minutes the staircase groaned when he's an extraordinary way.": 0.75442036092281339, "but I'm less than 5 minutes the staircase groaned when he's an extraordinary wait": 0.7674720663577318, "but in less than five minutes the staircase groaned when he's an extraordinary weight": 0.80769927799701691, "but in less than five minutes the staircase groaned when he's an extraordinary way": 0.80769927799701691, "but I'm less than 5 minutes the staircase groaned when he's an extraordinary way": 0.7674720663577318, "but in less than 5 minutes the staircase groaned when he's an extraordinary weight": 0.7811811616644263, "but in less than five minutes the staircase groaned when he's an extraordinary wait": 0.80769927799701691, "but in less than 5 minutes the staircase groaned when he's an extraordinary way": 0.81482807900756593, "but in less than 5 minutes the staircase groaned when he's an extraordinary wait": 0.81482807900756593, "but I'm less than 5 minutes the staircase groaned when he's an extraordinary weight": 0.7674720663577318}
    
    
    ORIGINAL Transcript: 
    'but I'm less than 5 minutes the staircase groaned when he's an extraordinary way' 
    with a confidence_score of: 0.8512189984321594
    
    
    RE-RANKED Transcript: 
    'but in less than 5 minutes the staircase groaned when he's an extraordinary wait' 
    with a confidence_score of: 0.8148280790075659
    
    
    GROUND TRUTH TRANSCRIPT: 
    BUT IN LESS THAN FIVE MINUTES THE STAIRCASE GROANED BENEATH AN EXTRAORDINARY WEIGHT
    
    The original transcript was RE-RANKED. The transcripts do not match!
    Differences between original and re-ranked:  ['i', "'m", 'way']
    
    
    The original transcript DOES NOT MATCH ground truth.
    Differences between original and ground truth:  ['5', 'when', "'m", 'he', "'s", 'i', 'way']
    
    
    The RE_RANKED transcript DOES NOT MATCH ground truth.
    Differences between Reranked and ground truth:  ['wait', '5', "'s", 'when', 'he']
    
    
    
    
    ORIGINAL Edit Distance: 
    18
    RE-RANKED Edit Distance: 
    14
    
    
    Waiting for operation to complete...
    
    
    RE-RANKED Results: 
     {'and the cry issued from his pores if we made us speak a cry frightful and its silence': 0.8229485416784883, 'and the cry issued from his pores if we made the speak a cry frightful and its silence': 0.81755691375583417, 'and the cry issued from his pores if we may the speak a cry frightful and its silence': 0.80107996519655, 'and the cry issued from his pores if we made the speak a cry frightful and it silence': 0.77812602724879987, 'and the cry issued from his pores if we made the speak a cry frightful in it silence': 0.77820050343871117, 'and the cry issued from his pores if we made us speak a cry frightful in its silence': 0.8230056628584862, "and the cry issued from his pores if we made the speak a cry frightful and it's silence": 0.82358051147311928, 'and the cry issued from his pores if we may the speak a cry frightful in it silence': 0.76172350123524668, 'and the cry issued from his pores if we made the speak a cry frightful in its silence': 0.81761403474956751, 'and the cry issued from his pores if we may the speak a cry frightful in its silence': 0.80190430544316771}
    
    
    ORIGINAL Transcript: 
    'and the cry issued from his pores if we made us speak a cry frightful in its silence' 
    with a confidence_score of: 0.9122896194458008
    
    
    RE-RANKED Transcript: 
    'and the cry issued from his pores if we made the speak a cry frightful and it's silence' 
    with a confidence_score of: 0.8235805114731193
    
    
    GROUND TRUTH TRANSCRIPT: 
    AND THE CRY ISSUED FROM HIS PORES IF WE MAY THUS SPEAK A CRY FRIGHTFUL IN ITS SILENCE
    
    The original transcript was RE-RANKED. The transcripts do not match!
    Differences between original and re-ranked:  ['us', 'in', 'its']
    
    
    The original transcript DOES NOT MATCH ground truth.
    Differences between original and ground truth:  ['us', 'made']
    
    
    The RE_RANKED transcript DOES NOT MATCH ground truth.
    Differences between Reranked and ground truth:  ['it', "'s", 'made']
    
    
    
    
    ORIGINAL Edit Distance: 
    4
    RE-RANKED Edit Distance: 
    7
    
    


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

    Found 97 text files in the directory: /src/lexicon/LibriSpeech/dev-clean/**/*.txt



    ---------------------------------------------------------------------------

    _DeadlineExceededError                    Traceback (most recent call last)

    ~/miniconda3/envs/tf-gpu/lib/python3.5/site-packages/google/gax/retry.py in inner(*args)
        120                 to_call = add_timeout_arg(a_func, timeout, **kwargs)
    --> 121                 return to_call(*args)
        122             except Exception as exception:  # pylint: disable=broad-except


    ~/miniconda3/envs/tf-gpu/lib/python3.5/site-packages/google/gax/retry.py in inner(*args)
         67         updated_args = args + (timeout,)
    ---> 68         return a_func(*updated_args, **kwargs)
         69 


    ~/miniconda3/envs/tf-gpu/lib/python3.5/site-packages/google/gax/__init__.py in _done_check(_)
        669             if not self.done():
    --> 670                 raise _DeadlineExceededError()
        671 


    _DeadlineExceededError: Deadline Exceeded

    
    During handling of the above exception, another exception occurred:


    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-13-ae178e66cb33> in <module>()
         55     # Detects speech and words in the audio file
         56     operation = client.long_running_recognize(config, audio)
    ---> 57     result = operation.result(timeout=90)
         58     alternatives = result.results[0].alternatives
         59 


    ~/miniconda3/envs/tf-gpu/lib/python3.5/site-packages/google/gax/__init__.py in result(self, timeout)
        593         """
        594         # Check exceptional case: raise if no response
    --> 595         if not self._poll(timeout).HasField('response'):
        596             raise GaxError(self._operation.error.message)
        597 


    ~/miniconda3/envs/tf-gpu/lib/python3.5/site-packages/google/gax/__init__.py in _poll(self, timeout)
        703 
        704         # Start polling, and return the final result from `_done_check`.
    --> 705         return retryable_done_check()
        706 
        707     def _execute_tasks(self):


    ~/miniconda3/envs/tf-gpu/lib/python3.5/site-packages/google/gax/retry.py in inner(*args)
        133                 # expected delay.
        134                 to_sleep = random.uniform(0, delay * 2)
    --> 135                 time.sleep(to_sleep / _MILLIS_PER_SECOND)
        136                 delay = min(delay * delay_mult, max_delay_millis)
        137 


    KeyboardInterrupt: 



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
from gcs_api_wrapper import GCSWrapper

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

    Epoch   0 Batch  500/2400 - Train Accuracy: 0.5570, Validation Accuracy: 0.6339, Loss: 3.0404
    Epoch   0 Batch 1000/2400 - Train Accuracy: 0.3839, Validation Accuracy: 0.6339, Loss: 4.5766
    Epoch   0 Batch 1500/2400 - Train Accuracy: 0.6307, Validation Accuracy: 0.6339, Loss: 2.7718
    Epoch   0 Batch 2000/2400 - Train Accuracy: 0.3582, Validation Accuracy: 0.6339, Loss: 4.4786
    Epoch   1 Batch  500/2400 - Train Accuracy: 0.6232, Validation Accuracy: 0.6339, Loss: 2.5335
    Epoch   1 Batch 1000/2400 - Train Accuracy: 0.4174, Validation Accuracy: 0.6339, Loss: 3.7214
    Epoch   1 Batch 1500/2400 - Train Accuracy: 0.6705, Validation Accuracy: 0.6339, Loss: 2.1842
    Epoch   1 Batch 2000/2400 - Train Accuracy: 0.4183, Validation Accuracy: 0.6339, Loss: 3.9006
    Epoch   2 Batch  500/2400 - Train Accuracy: 0.6415, Validation Accuracy: 0.6339, Loss: 2.2166
    Epoch   2 Batch 1000/2400 - Train Accuracy: 0.4576, Validation Accuracy: 0.6339, Loss: 3.2150
    Epoch   2 Batch 1500/2400 - Train Accuracy: 0.6903, Validation Accuracy: 0.6339, Loss: 1.9266
    Epoch   2 Batch 2000/2400 - Train Accuracy: 0.4495, Validation Accuracy: 0.6339, Loss: 3.4746
    Epoch   3 Batch  500/2400 - Train Accuracy: 0.6728, Validation Accuracy: 0.6339, Loss: 2.0024
    Epoch   3 Batch 1000/2400 - Train Accuracy: 0.4777, Validation Accuracy: 0.6339, Loss: 2.8544
    Epoch   3 Batch 1500/2400 - Train Accuracy: 0.7216, Validation Accuracy: 0.6339, Loss: 1.6540
    Epoch   3 Batch 2000/2400 - Train Accuracy: 0.4567, Validation Accuracy: 0.6339, Loss: 3.1523
    Epoch   4 Batch  500/2400 - Train Accuracy: 0.6783, Validation Accuracy: 0.6339, Loss: 1.8692
    Epoch   4 Batch 1000/2400 - Train Accuracy: 0.4777, Validation Accuracy: 0.6339, Loss: 2.5700
    Epoch   4 Batch 1500/2400 - Train Accuracy: 0.7301, Validation Accuracy: 0.6339, Loss: 1.4682
    Epoch   4 Batch 2000/2400 - Train Accuracy: 0.4736, Validation Accuracy: 0.6339, Loss: 2.8515
    Epoch   5 Batch  500/2400 - Train Accuracy: 0.6893, Validation Accuracy: 0.6339, Loss: 1.7067
    Epoch   5 Batch 1000/2400 - Train Accuracy: 0.4933, Validation Accuracy: 0.6339, Loss: 2.3715
    Epoch   5 Batch 1500/2400 - Train Accuracy: 0.7500, Validation Accuracy: 0.6339, Loss: 1.2849
    Epoch   5 Batch 2000/2400 - Train Accuracy: 0.4808, Validation Accuracy: 0.6339, Loss: 2.6013
    Epoch   6 Batch  500/2400 - Train Accuracy: 0.6985, Validation Accuracy: 0.6339, Loss: 1.5452
    Epoch   6 Batch 1000/2400 - Train Accuracy: 0.5379, Validation Accuracy: 0.6339, Loss: 2.1082
    Epoch   6 Batch 1500/2400 - Train Accuracy: 0.7585, Validation Accuracy: 0.6339, Loss: 1.1517
    Epoch   6 Batch 2000/2400 - Train Accuracy: 0.4928, Validation Accuracy: 0.6339, Loss: 2.3818
    Epoch   7 Batch  500/2400 - Train Accuracy: 0.6912, Validation Accuracy: 0.6339, Loss: 1.4616
    Epoch   7 Batch 1000/2400 - Train Accuracy: 0.5312, Validation Accuracy: 0.6339, Loss: 1.9536
    Epoch   7 Batch 1500/2400 - Train Accuracy: 0.7699, Validation Accuracy: 0.6339, Loss: 1.0182
    Epoch   7 Batch 2000/2400 - Train Accuracy: 0.5192, Validation Accuracy: 0.6339, Loss: 2.1623
    Epoch   8 Batch  500/2400 - Train Accuracy: 0.6967, Validation Accuracy: 0.6362, Loss: 1.3247
    Epoch   8 Batch 1000/2400 - Train Accuracy: 0.5781, Validation Accuracy: 0.6339, Loss: 1.7735
    Epoch   8 Batch 1500/2400 - Train Accuracy: 0.7926, Validation Accuracy: 0.6339, Loss: 0.8861
    Epoch   8 Batch 2000/2400 - Train Accuracy: 0.5433, Validation Accuracy: 0.6339, Loss: 1.9498
    Epoch   9 Batch  500/2400 - Train Accuracy: 0.7096, Validation Accuracy: 0.6339, Loss: 1.2163
    Epoch   9 Batch 1000/2400 - Train Accuracy: 0.6049, Validation Accuracy: 0.6339, Loss: 1.5981
    Epoch   9 Batch 1500/2400 - Train Accuracy: 0.8125, Validation Accuracy: 0.6339, Loss: 0.7869
    Epoch   9 Batch 2000/2400 - Train Accuracy: 0.5553, Validation Accuracy: 0.6339, Loss: 1.7822
    Epoch  10 Batch  500/2400 - Train Accuracy: 0.7243, Validation Accuracy: 0.6339, Loss: 1.1307
    Epoch  10 Batch 1000/2400 - Train Accuracy: 0.6406, Validation Accuracy: 0.6339, Loss: 1.4611
    Epoch  10 Batch 1500/2400 - Train Accuracy: 0.8295, Validation Accuracy: 0.6339, Loss: 0.6797
    Epoch  10 Batch 2000/2400 - Train Accuracy: 0.5938, Validation Accuracy: 0.6339, Loss: 1.6302
    Epoch  11 Batch  500/2400 - Train Accuracy: 0.7463, Validation Accuracy: 0.6339, Loss: 1.0599
    Epoch  11 Batch 1000/2400 - Train Accuracy: 0.6629, Validation Accuracy: 0.6339, Loss: 1.3507
    Epoch  11 Batch 1500/2400 - Train Accuracy: 0.8580, Validation Accuracy: 0.6339, Loss: 0.5859
    Epoch  11 Batch 2000/2400 - Train Accuracy: 0.6034, Validation Accuracy: 0.6339, Loss: 1.5344
    Epoch  12 Batch  500/2400 - Train Accuracy: 0.7610, Validation Accuracy: 0.6339, Loss: 0.9842
    Epoch  12 Batch 1000/2400 - Train Accuracy: 0.6473, Validation Accuracy: 0.6339, Loss: 1.2971
    Epoch  12 Batch 1500/2400 - Train Accuracy: 0.8580, Validation Accuracy: 0.6339, Loss: 0.5255
    Epoch  12 Batch 2000/2400 - Train Accuracy: 0.6010, Validation Accuracy: 0.6339, Loss: 1.3935
    Epoch  13 Batch  500/2400 - Train Accuracy: 0.7721, Validation Accuracy: 0.6339, Loss: 0.8974
    Epoch  13 Batch 1000/2400 - Train Accuracy: 0.6629, Validation Accuracy: 0.6339, Loss: 1.2005
    Epoch  13 Batch 1500/2400 - Train Accuracy: 0.8949, Validation Accuracy: 0.6339, Loss: 0.4541
    Epoch  13 Batch 2000/2400 - Train Accuracy: 0.6298, Validation Accuracy: 0.6339, Loss: 1.2525
    Epoch  14 Batch  500/2400 - Train Accuracy: 0.7923, Validation Accuracy: 0.6339, Loss: 0.8303
    Epoch  14 Batch 1000/2400 - Train Accuracy: 0.6719, Validation Accuracy: 0.6339, Loss: 1.0979
    Epoch  14 Batch 1500/2400 - Train Accuracy: 0.8778, Validation Accuracy: 0.6339, Loss: 0.3918
    Epoch  14 Batch 2000/2400 - Train Accuracy: 0.6635, Validation Accuracy: 0.6339, Loss: 1.1165
    Epoch  15 Batch  500/2400 - Train Accuracy: 0.8015, Validation Accuracy: 0.6339, Loss: 0.8186
    Epoch  15 Batch 1000/2400 - Train Accuracy: 0.7411, Validation Accuracy: 0.6339, Loss: 0.9573
    Epoch  15 Batch 1500/2400 - Train Accuracy: 0.9034, Validation Accuracy: 0.6339, Loss: 0.3815
    Epoch  15 Batch 2000/2400 - Train Accuracy: 0.6322, Validation Accuracy: 0.6339, Loss: 1.0469
    Epoch  16 Batch  500/2400 - Train Accuracy: 0.8088, Validation Accuracy: 0.6339, Loss: 0.7384
    Epoch  16 Batch 1000/2400 - Train Accuracy: 0.6853, Validation Accuracy: 0.6339, Loss: 0.8702
    Epoch  16 Batch 1500/2400 - Train Accuracy: 0.9062, Validation Accuracy: 0.6339, Loss: 0.3387
    Epoch  16 Batch 2000/2400 - Train Accuracy: 0.6779, Validation Accuracy: 0.6339, Loss: 0.9736
    Epoch  17 Batch  500/2400 - Train Accuracy: 0.8180, Validation Accuracy: 0.6339, Loss: 0.7337
    Epoch  17 Batch 1000/2400 - Train Accuracy: 0.7634, Validation Accuracy: 0.6339, Loss: 0.8291
    Epoch  17 Batch 1500/2400 - Train Accuracy: 0.9318, Validation Accuracy: 0.6339, Loss: 0.2843
    Epoch  17 Batch 2000/2400 - Train Accuracy: 0.6971, Validation Accuracy: 0.6339, Loss: 0.9246
    Epoch  18 Batch  500/2400 - Train Accuracy: 0.7941, Validation Accuracy: 0.6339, Loss: 0.6937
    Epoch  18 Batch 1000/2400 - Train Accuracy: 0.7812, Validation Accuracy: 0.6339, Loss: 0.7610
    Epoch  18 Batch 1500/2400 - Train Accuracy: 0.9347, Validation Accuracy: 0.6339, Loss: 0.2580
    Epoch  18 Batch 2000/2400 - Train Accuracy: 0.7019, Validation Accuracy: 0.6339, Loss: 0.8523
    Epoch  19 Batch  500/2400 - Train Accuracy: 0.8327, Validation Accuracy: 0.6339, Loss: 0.6111
    Epoch  19 Batch 1000/2400 - Train Accuracy: 0.7746, Validation Accuracy: 0.6339, Loss: 0.7092
    Epoch  19 Batch 1500/2400 - Train Accuracy: 0.9432, Validation Accuracy: 0.6339, Loss: 0.2428
    Epoch  19 Batch 2000/2400 - Train Accuracy: 0.7620, Validation Accuracy: 0.6339, Loss: 0.7962
    Epoch  20 Batch  500/2400 - Train Accuracy: 0.8438, Validation Accuracy: 0.6339, Loss: 0.5840
    Epoch  20 Batch 1000/2400 - Train Accuracy: 0.7522, Validation Accuracy: 0.6339, Loss: 0.6448
    Epoch  20 Batch 1500/2400 - Train Accuracy: 0.9460, Validation Accuracy: 0.6362, Loss: 0.2140
    Epoch  20 Batch 2000/2400 - Train Accuracy: 0.7500, Validation Accuracy: 0.6362, Loss: 0.7080
    Epoch  21 Batch  500/2400 - Train Accuracy: 0.8640, Validation Accuracy: 0.6339, Loss: 0.5800
    Epoch  21 Batch 1000/2400 - Train Accuracy: 0.7589, Validation Accuracy: 0.6339, Loss: 0.6473
    Epoch  21 Batch 1500/2400 - Train Accuracy: 0.9602, Validation Accuracy: 0.6339, Loss: 0.1847
    Epoch  21 Batch 2000/2400 - Train Accuracy: 0.7139, Validation Accuracy: 0.6384, Loss: 0.6637
    Epoch  22 Batch  500/2400 - Train Accuracy: 0.8585, Validation Accuracy: 0.6339, Loss: 0.5283
    Epoch  22 Batch 1000/2400 - Train Accuracy: 0.7723, Validation Accuracy: 0.6339, Loss: 0.5823
    Epoch  22 Batch 1500/2400 - Train Accuracy: 0.9489, Validation Accuracy: 0.6339, Loss: 0.1528
    Epoch  22 Batch 2000/2400 - Train Accuracy: 0.7788, Validation Accuracy: 0.6384, Loss: 0.6134
    Epoch  23 Batch  500/2400 - Train Accuracy: 0.8493, Validation Accuracy: 0.6339, Loss: 0.4685
    Epoch  23 Batch 1000/2400 - Train Accuracy: 0.7969, Validation Accuracy: 0.6339, Loss: 0.5515
    Epoch  23 Batch 1500/2400 - Train Accuracy: 0.9716, Validation Accuracy: 0.6339, Loss: 0.1543
    Epoch  23 Batch 2000/2400 - Train Accuracy: 0.7620, Validation Accuracy: 0.6362, Loss: 0.5848
    Epoch  24 Batch  500/2400 - Train Accuracy: 0.8676, Validation Accuracy: 0.6339, Loss: 0.4613
    Epoch  24 Batch 1000/2400 - Train Accuracy: 0.8013, Validation Accuracy: 0.6339, Loss: 0.5133
    Epoch  24 Batch 1500/2400 - Train Accuracy: 0.9631, Validation Accuracy: 0.6339, Loss: 0.1362
    Epoch  24 Batch 2000/2400 - Train Accuracy: 0.8125, Validation Accuracy: 0.6362, Loss: 0.5467
    Epoch  25 Batch  500/2400 - Train Accuracy: 0.8566, Validation Accuracy: 0.6339, Loss: 0.4405
    Epoch  25 Batch 1000/2400 - Train Accuracy: 0.8371, Validation Accuracy: 0.6339, Loss: 0.4634
    Epoch  25 Batch 1500/2400 - Train Accuracy: 0.9716, Validation Accuracy: 0.6339, Loss: 0.1353
    Epoch  25 Batch 2000/2400 - Train Accuracy: 0.7812, Validation Accuracy: 0.6384, Loss: 0.5157
    Epoch  26 Batch  500/2400 - Train Accuracy: 0.8511, Validation Accuracy: 0.6339, Loss: 0.4270
    Epoch  26 Batch 1000/2400 - Train Accuracy: 0.8460, Validation Accuracy: 0.6339, Loss: 0.4068
    Epoch  26 Batch 1500/2400 - Train Accuracy: 0.9659, Validation Accuracy: 0.6339, Loss: 0.1237
    Epoch  26 Batch 2000/2400 - Train Accuracy: 0.8149, Validation Accuracy: 0.6384, Loss: 0.4771
    Epoch  27 Batch  500/2400 - Train Accuracy: 0.8511, Validation Accuracy: 0.6339, Loss: 0.4089
    Epoch  27 Batch 1000/2400 - Train Accuracy: 0.8504, Validation Accuracy: 0.6339, Loss: 0.3670
    Epoch  27 Batch 1500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0948
    Epoch  27 Batch 2000/2400 - Train Accuracy: 0.8269, Validation Accuracy: 0.6384, Loss: 0.4333
    Epoch  28 Batch  500/2400 - Train Accuracy: 0.8658, Validation Accuracy: 0.6362, Loss: 0.3985
    Epoch  28 Batch 1000/2400 - Train Accuracy: 0.8348, Validation Accuracy: 0.6339, Loss: 0.3839
    Epoch  28 Batch 1500/2400 - Train Accuracy: 0.9773, Validation Accuracy: 0.6339, Loss: 0.1016
    Epoch  28 Batch 2000/2400 - Train Accuracy: 0.8029, Validation Accuracy: 0.6362, Loss: 0.4031
    Epoch  29 Batch  500/2400 - Train Accuracy: 0.8787, Validation Accuracy: 0.6339, Loss: 0.3768
    Epoch  29 Batch 1000/2400 - Train Accuracy: 0.8415, Validation Accuracy: 0.6362, Loss: 0.3458
    Epoch  29 Batch 1500/2400 - Train Accuracy: 0.9830, Validation Accuracy: 0.6339, Loss: 0.0966
    Epoch  29 Batch 2000/2400 - Train Accuracy: 0.8438, Validation Accuracy: 0.6362, Loss: 0.3798
    Epoch  30 Batch  500/2400 - Train Accuracy: 0.8676, Validation Accuracy: 0.6339, Loss: 0.3389
    Epoch  30 Batch 1000/2400 - Train Accuracy: 0.8571, Validation Accuracy: 0.6362, Loss: 0.3212
    Epoch  30 Batch 1500/2400 - Train Accuracy: 0.9744, Validation Accuracy: 0.6362, Loss: 0.0786
    Epoch  30 Batch 2000/2400 - Train Accuracy: 0.8822, Validation Accuracy: 0.6362, Loss: 0.3821
    Epoch  31 Batch  500/2400 - Train Accuracy: 0.8732, Validation Accuracy: 0.6362, Loss: 0.3289
    Epoch  31 Batch 1000/2400 - Train Accuracy: 0.8415, Validation Accuracy: 0.6339, Loss: 0.3226
    Epoch  31 Batch 1500/2400 - Train Accuracy: 0.9886, Validation Accuracy: 0.6339, Loss: 0.0878
    Epoch  31 Batch 2000/2400 - Train Accuracy: 0.8606, Validation Accuracy: 0.6384, Loss: 0.3134
    Epoch  32 Batch  500/2400 - Train Accuracy: 0.8603, Validation Accuracy: 0.6339, Loss: 0.3268
    Epoch  32 Batch 1000/2400 - Train Accuracy: 0.9040, Validation Accuracy: 0.6339, Loss: 0.2952
    Epoch  32 Batch 1500/2400 - Train Accuracy: 0.9773, Validation Accuracy: 0.6339, Loss: 0.0615
    Epoch  32 Batch 2000/2400 - Train Accuracy: 0.8365, Validation Accuracy: 0.6384, Loss: 0.3058
    Epoch  33 Batch  500/2400 - Train Accuracy: 0.8879, Validation Accuracy: 0.6339, Loss: 0.3062
    Epoch  33 Batch 1000/2400 - Train Accuracy: 0.9152, Validation Accuracy: 0.6339, Loss: 0.2599
    Epoch  33 Batch 1500/2400 - Train Accuracy: 0.9773, Validation Accuracy: 0.6339, Loss: 0.0742
    Epoch  33 Batch 2000/2400 - Train Accuracy: 0.8702, Validation Accuracy: 0.6339, Loss: 0.3450
    Epoch  34 Batch  500/2400 - Train Accuracy: 0.8713, Validation Accuracy: 0.6339, Loss: 0.2665
    Epoch  34 Batch 1000/2400 - Train Accuracy: 0.8996, Validation Accuracy: 0.6339, Loss: 0.2396
    Epoch  34 Batch 1500/2400 - Train Accuracy: 0.9915, Validation Accuracy: 0.6362, Loss: 0.0561
    Epoch  34 Batch 2000/2400 - Train Accuracy: 0.8990, Validation Accuracy: 0.6362, Loss: 0.2825
    Epoch  35 Batch  500/2400 - Train Accuracy: 0.8879, Validation Accuracy: 0.6339, Loss: 0.2788
    Epoch  35 Batch 1000/2400 - Train Accuracy: 0.9062, Validation Accuracy: 0.6362, Loss: 0.2616
    Epoch  35 Batch 1500/2400 - Train Accuracy: 0.9830, Validation Accuracy: 0.6339, Loss: 0.0416
    Epoch  35 Batch 2000/2400 - Train Accuracy: 0.8942, Validation Accuracy: 0.6362, Loss: 0.2956
    Epoch  36 Batch  500/2400 - Train Accuracy: 0.8952, Validation Accuracy: 0.6339, Loss: 0.2750
    Epoch  36 Batch 1000/2400 - Train Accuracy: 0.9241, Validation Accuracy: 0.6339, Loss: 0.2404
    Epoch  36 Batch 1500/2400 - Train Accuracy: 0.9744, Validation Accuracy: 0.6339, Loss: 0.0535
    Epoch  36 Batch 2000/2400 - Train Accuracy: 0.8678, Validation Accuracy: 0.6362, Loss: 0.2449
    Epoch  37 Batch  500/2400 - Train Accuracy: 0.8860, Validation Accuracy: 0.6362, Loss: 0.2426
    Epoch  37 Batch 1000/2400 - Train Accuracy: 0.8973, Validation Accuracy: 0.6339, Loss: 0.2280
    Epoch  37 Batch 1500/2400 - Train Accuracy: 0.9830, Validation Accuracy: 0.6362, Loss: 0.0763
    Epoch  37 Batch 2000/2400 - Train Accuracy: 0.8966, Validation Accuracy: 0.6362, Loss: 0.2536
    Epoch  38 Batch  500/2400 - Train Accuracy: 0.9044, Validation Accuracy: 0.6384, Loss: 0.2438
    Epoch  38 Batch 1000/2400 - Train Accuracy: 0.8973, Validation Accuracy: 0.6339, Loss: 0.1792
    Epoch  38 Batch 1500/2400 - Train Accuracy: 0.9716, Validation Accuracy: 0.6362, Loss: 0.0512
    Epoch  38 Batch 2000/2400 - Train Accuracy: 0.9135, Validation Accuracy: 0.6362, Loss: 0.2638
    Epoch  39 Batch  500/2400 - Train Accuracy: 0.8952, Validation Accuracy: 0.6339, Loss: 0.2388
    Epoch  39 Batch 1000/2400 - Train Accuracy: 0.8951, Validation Accuracy: 0.6339, Loss: 0.1637
    Epoch  39 Batch 1500/2400 - Train Accuracy: 0.9915, Validation Accuracy: 0.6339, Loss: 0.0374
    Epoch  39 Batch 2000/2400 - Train Accuracy: 0.8918, Validation Accuracy: 0.6362, Loss: 0.2389
    Epoch  40 Batch  500/2400 - Train Accuracy: 0.9210, Validation Accuracy: 0.6339, Loss: 0.2273
    Epoch  40 Batch 1000/2400 - Train Accuracy: 0.9196, Validation Accuracy: 0.6339, Loss: 0.1760
    Epoch  40 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0464
    Epoch  40 Batch 2000/2400 - Train Accuracy: 0.8726, Validation Accuracy: 0.6362, Loss: 0.2212
    Epoch  41 Batch  500/2400 - Train Accuracy: 0.8787, Validation Accuracy: 0.6384, Loss: 0.2444
    Epoch  41 Batch 1000/2400 - Train Accuracy: 0.9353, Validation Accuracy: 0.6339, Loss: 0.1539
    Epoch  41 Batch 1500/2400 - Train Accuracy: 0.9773, Validation Accuracy: 0.6384, Loss: 0.0501
    Epoch  41 Batch 2000/2400 - Train Accuracy: 0.8606, Validation Accuracy: 0.6362, Loss: 0.1887
    Epoch  42 Batch  500/2400 - Train Accuracy: 0.9136, Validation Accuracy: 0.6339, Loss: 0.2276
    Epoch  42 Batch 1000/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6339, Loss: 0.1541
    Epoch  42 Batch 1500/2400 - Train Accuracy: 0.9915, Validation Accuracy: 0.6362, Loss: 0.0365
    Epoch  42 Batch 2000/2400 - Train Accuracy: 0.8942, Validation Accuracy: 0.6362, Loss: 0.2104
    Epoch  43 Batch  500/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6339, Loss: 0.1884
    Epoch  43 Batch 1000/2400 - Train Accuracy: 0.9107, Validation Accuracy: 0.6339, Loss: 0.1835
    Epoch  43 Batch 1500/2400 - Train Accuracy: 0.9972, Validation Accuracy: 0.6362, Loss: 0.0341
    Epoch  43 Batch 2000/2400 - Train Accuracy: 0.9159, Validation Accuracy: 0.6339, Loss: 0.1875
    Epoch  44 Batch  500/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6362, Loss: 0.1641
    Epoch  44 Batch 1000/2400 - Train Accuracy: 0.9442, Validation Accuracy: 0.6339, Loss: 0.1380
    Epoch  44 Batch 1500/2400 - Train Accuracy: 0.9943, Validation Accuracy: 0.6339, Loss: 0.0358
    Epoch  44 Batch 2000/2400 - Train Accuracy: 0.8654, Validation Accuracy: 0.6362, Loss: 0.2172
    Epoch  45 Batch  500/2400 - Train Accuracy: 0.9191, Validation Accuracy: 0.6384, Loss: 0.1719
    Epoch  45 Batch 1000/2400 - Train Accuracy: 0.9509, Validation Accuracy: 0.6339, Loss: 0.1356
    Epoch  45 Batch 1500/2400 - Train Accuracy: 0.9972, Validation Accuracy: 0.6362, Loss: 0.0339
    Epoch  45 Batch 2000/2400 - Train Accuracy: 0.8942, Validation Accuracy: 0.6339, Loss: 0.1802
    Epoch  46 Batch  500/2400 - Train Accuracy: 0.9283, Validation Accuracy: 0.6339, Loss: 0.1879
    Epoch  46 Batch 1000/2400 - Train Accuracy: 0.9129, Validation Accuracy: 0.6339, Loss: 0.1220
    Epoch  46 Batch 1500/2400 - Train Accuracy: 0.9716, Validation Accuracy: 0.6362, Loss: 0.0290
    Epoch  46 Batch 2000/2400 - Train Accuracy: 0.9423, Validation Accuracy: 0.6406, Loss: 0.1382
    Epoch  47 Batch  500/2400 - Train Accuracy: 0.9540, Validation Accuracy: 0.6339, Loss: 0.1799
    Epoch  47 Batch 1000/2400 - Train Accuracy: 0.9598, Validation Accuracy: 0.6339, Loss: 0.1299
    Epoch  47 Batch 1500/2400 - Train Accuracy: 0.9972, Validation Accuracy: 0.6339, Loss: 0.0227
    Epoch  47 Batch 2000/2400 - Train Accuracy: 0.9447, Validation Accuracy: 0.6362, Loss: 0.1260
    Epoch  48 Batch  500/2400 - Train Accuracy: 0.9338, Validation Accuracy: 0.6339, Loss: 0.1546
    Epoch  48 Batch 1000/2400 - Train Accuracy: 0.9643, Validation Accuracy: 0.6339, Loss: 0.1172
    Epoch  48 Batch 1500/2400 - Train Accuracy: 0.9915, Validation Accuracy: 0.6362, Loss: 0.0264
    Epoch  48 Batch 2000/2400 - Train Accuracy: 0.9303, Validation Accuracy: 0.6339, Loss: 0.1500
    Epoch  49 Batch  500/2400 - Train Accuracy: 0.9283, Validation Accuracy: 0.6339, Loss: 0.1290
    Epoch  49 Batch 1000/2400 - Train Accuracy: 0.9286, Validation Accuracy: 0.6339, Loss: 0.0936
    Epoch  49 Batch 1500/2400 - Train Accuracy: 0.9886, Validation Accuracy: 0.6362, Loss: 0.0314
    Epoch  49 Batch 2000/2400 - Train Accuracy: 0.9784, Validation Accuracy: 0.6339, Loss: 0.1293
    Epoch  50 Batch  500/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6339, Loss: 0.1285
    Epoch  50 Batch 1000/2400 - Train Accuracy: 0.9576, Validation Accuracy: 0.6362, Loss: 0.1085
    Epoch  50 Batch 1500/2400 - Train Accuracy: 0.9830, Validation Accuracy: 0.6339, Loss: 0.0186
    Epoch  50 Batch 2000/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6339, Loss: 0.1443
    Epoch  51 Batch  500/2400 - Train Accuracy: 0.9577, Validation Accuracy: 0.6362, Loss: 0.1446
    Epoch  51 Batch 1000/2400 - Train Accuracy: 0.9531, Validation Accuracy: 0.6339, Loss: 0.1003
    Epoch  51 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0229
    Epoch  51 Batch 2000/2400 - Train Accuracy: 0.9231, Validation Accuracy: 0.6339, Loss: 0.1405
    Epoch  52 Batch  500/2400 - Train Accuracy: 0.9210, Validation Accuracy: 0.6362, Loss: 0.1247
    Epoch  52 Batch 1000/2400 - Train Accuracy: 0.9866, Validation Accuracy: 0.6362, Loss: 0.0688
    Epoch  52 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0247
    Epoch  52 Batch 2000/2400 - Train Accuracy: 0.9423, Validation Accuracy: 0.6362, Loss: 0.1144
    Epoch  53 Batch  500/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6362, Loss: 0.1256
    Epoch  53 Batch 1000/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0971
    Epoch  53 Batch 1500/2400 - Train Accuracy: 0.9773, Validation Accuracy: 0.6339, Loss: 0.0272
    Epoch  53 Batch 2000/2400 - Train Accuracy: 0.9279, Validation Accuracy: 0.6339, Loss: 0.1503
    Epoch  54 Batch  500/2400 - Train Accuracy: 0.9614, Validation Accuracy: 0.6339, Loss: 0.1076
    Epoch  54 Batch 1000/2400 - Train Accuracy: 0.9509, Validation Accuracy: 0.6362, Loss: 0.0895
    Epoch  54 Batch 1500/2400 - Train Accuracy: 0.9972, Validation Accuracy: 0.6339, Loss: 0.0214
    Epoch  54 Batch 2000/2400 - Train Accuracy: 0.9663, Validation Accuracy: 0.6362, Loss: 0.1065
    Epoch  55 Batch  500/2400 - Train Accuracy: 0.9651, Validation Accuracy: 0.6362, Loss: 0.1130
    Epoch  55 Batch 1000/2400 - Train Accuracy: 0.9844, Validation Accuracy: 0.6339, Loss: 0.0643
    Epoch  55 Batch 1500/2400 - Train Accuracy: 0.9915, Validation Accuracy: 0.6339, Loss: 0.0216
    Epoch  55 Batch 2000/2400 - Train Accuracy: 0.9567, Validation Accuracy: 0.6339, Loss: 0.0888
    Epoch  56 Batch  500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6339, Loss: 0.1098
    Epoch  56 Batch 1000/2400 - Train Accuracy: 0.9621, Validation Accuracy: 0.6362, Loss: 0.0730
    Epoch  56 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0171
    Epoch  56 Batch 2000/2400 - Train Accuracy: 0.9471, Validation Accuracy: 0.6384, Loss: 0.1046
    Epoch  57 Batch  500/2400 - Train Accuracy: 0.9430, Validation Accuracy: 0.6339, Loss: 0.1291
    Epoch  57 Batch 1000/2400 - Train Accuracy: 0.9799, Validation Accuracy: 0.6339, Loss: 0.0745
    Epoch  57 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0191
    Epoch  57 Batch 2000/2400 - Train Accuracy: 0.9399, Validation Accuracy: 0.6362, Loss: 0.1102
    Epoch  58 Batch  500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6339, Loss: 0.1312
    Epoch  58 Batch 1000/2400 - Train Accuracy: 0.9732, Validation Accuracy: 0.6339, Loss: 0.0695
    Epoch  58 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0175
    Epoch  58 Batch 2000/2400 - Train Accuracy: 0.9615, Validation Accuracy: 0.6362, Loss: 0.1031
    Epoch  59 Batch  500/2400 - Train Accuracy: 0.9430, Validation Accuracy: 0.6339, Loss: 0.1139
    Epoch  59 Batch 1000/2400 - Train Accuracy: 0.9487, Validation Accuracy: 0.6362, Loss: 0.0846
    Epoch  59 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0217
    Epoch  59 Batch 2000/2400 - Train Accuracy: 0.9736, Validation Accuracy: 0.6339, Loss: 0.1029
    Epoch  60 Batch  500/2400 - Train Accuracy: 0.9412, Validation Accuracy: 0.6339, Loss: 0.1038
    Epoch  60 Batch 1000/2400 - Train Accuracy: 0.9442, Validation Accuracy: 0.6339, Loss: 0.0980
    Epoch  60 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0162
    Epoch  60 Batch 2000/2400 - Train Accuracy: 0.9519, Validation Accuracy: 0.6362, Loss: 0.1130
    Epoch  61 Batch  500/2400 - Train Accuracy: 0.9449, Validation Accuracy: 0.6339, Loss: 0.0936
    Epoch  61 Batch 1000/2400 - Train Accuracy: 0.9710, Validation Accuracy: 0.6339, Loss: 0.0903
    Epoch  61 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0223
    Epoch  61 Batch 2000/2400 - Train Accuracy: 0.9760, Validation Accuracy: 0.6339, Loss: 0.0850
    Epoch  62 Batch  500/2400 - Train Accuracy: 0.9504, Validation Accuracy: 0.6339, Loss: 0.1195
    Epoch  62 Batch 1000/2400 - Train Accuracy: 0.9844, Validation Accuracy: 0.6339, Loss: 0.0836
    Epoch  62 Batch 1500/2400 - Train Accuracy: 0.9943, Validation Accuracy: 0.6339, Loss: 0.0175
    Epoch  62 Batch 2000/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6339, Loss: 0.0639
    Epoch  63 Batch  500/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6339, Loss: 0.0960
    Epoch  63 Batch 1000/2400 - Train Accuracy: 0.9710, Validation Accuracy: 0.6362, Loss: 0.0803
    Epoch  63 Batch 1500/2400 - Train Accuracy: 0.9943, Validation Accuracy: 0.6339, Loss: 0.0220
    Epoch  63 Batch 2000/2400 - Train Accuracy: 0.9760, Validation Accuracy: 0.6339, Loss: 0.1181
    Epoch  64 Batch  500/2400 - Train Accuracy: 0.9412, Validation Accuracy: 0.6339, Loss: 0.0972
    Epoch  64 Batch 1000/2400 - Train Accuracy: 0.9821, Validation Accuracy: 0.6339, Loss: 0.0682
    Epoch  64 Batch 1500/2400 - Train Accuracy: 0.9943, Validation Accuracy: 0.6339, Loss: 0.0232
    Epoch  64 Batch 2000/2400 - Train Accuracy: 0.9712, Validation Accuracy: 0.6339, Loss: 0.0637
    Epoch  65 Batch  500/2400 - Train Accuracy: 0.9393, Validation Accuracy: 0.6362, Loss: 0.0818
    Epoch  65 Batch 1000/2400 - Train Accuracy: 0.9554, Validation Accuracy: 0.6339, Loss: 0.0633
    Epoch  65 Batch 1500/2400 - Train Accuracy: 0.9886, Validation Accuracy: 0.6339, Loss: 0.0194
    Epoch  65 Batch 2000/2400 - Train Accuracy: 0.9591, Validation Accuracy: 0.6339, Loss: 0.0715
    Epoch  66 Batch  500/2400 - Train Accuracy: 0.9393, Validation Accuracy: 0.6339, Loss: 0.0905
    Epoch  66 Batch 1000/2400 - Train Accuracy: 0.9866, Validation Accuracy: 0.6339, Loss: 0.0568
    Epoch  66 Batch 1500/2400 - Train Accuracy: 0.9744, Validation Accuracy: 0.6362, Loss: 0.0192
    Epoch  66 Batch 2000/2400 - Train Accuracy: 0.9760, Validation Accuracy: 0.6339, Loss: 0.0857
    Epoch  67 Batch  500/2400 - Train Accuracy: 0.9540, Validation Accuracy: 0.6339, Loss: 0.0758
    Epoch  67 Batch 1000/2400 - Train Accuracy: 0.9777, Validation Accuracy: 0.6339, Loss: 0.0533
    Epoch  67 Batch 1500/2400 - Train Accuracy: 0.9858, Validation Accuracy: 0.6362, Loss: 0.0218
    Epoch  67 Batch 2000/2400 - Train Accuracy: 0.9832, Validation Accuracy: 0.6339, Loss: 0.0672
    Epoch  68 Batch  500/2400 - Train Accuracy: 0.9467, Validation Accuracy: 0.6339, Loss: 0.0947
    Epoch  68 Batch 1000/2400 - Train Accuracy: 0.9777, Validation Accuracy: 0.6339, Loss: 0.0587
    Epoch  68 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0113
    Epoch  68 Batch 2000/2400 - Train Accuracy: 0.9760, Validation Accuracy: 0.6339, Loss: 0.0460
    Epoch  69 Batch  500/2400 - Train Accuracy: 0.9651, Validation Accuracy: 0.6339, Loss: 0.1071
    Epoch  69 Batch 1000/2400 - Train Accuracy: 0.9799, Validation Accuracy: 0.6339, Loss: 0.0662
    Epoch  69 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0197
    Epoch  69 Batch 2000/2400 - Train Accuracy: 0.9760, Validation Accuracy: 0.6339, Loss: 0.0713
    Epoch  70 Batch  500/2400 - Train Accuracy: 0.9706, Validation Accuracy: 0.6339, Loss: 0.0865
    Epoch  70 Batch 1000/2400 - Train Accuracy: 0.9955, Validation Accuracy: 0.6339, Loss: 0.0444
    Epoch  70 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0130
    Epoch  70 Batch 2000/2400 - Train Accuracy: 0.9712, Validation Accuracy: 0.6362, Loss: 0.0825
    Epoch  71 Batch  500/2400 - Train Accuracy: 0.9449, Validation Accuracy: 0.6339, Loss: 0.0898
    Epoch  71 Batch 1000/2400 - Train Accuracy: 0.9464, Validation Accuracy: 0.6339, Loss: 0.0690
    Epoch  71 Batch 1500/2400 - Train Accuracy: 0.9972, Validation Accuracy: 0.6339, Loss: 0.0092
    Epoch  71 Batch 2000/2400 - Train Accuracy: 0.9856, Validation Accuracy: 0.6339, Loss: 0.0593
    Epoch  72 Batch  500/2400 - Train Accuracy: 0.9504, Validation Accuracy: 0.6339, Loss: 0.0731
    Epoch  72 Batch 1000/2400 - Train Accuracy: 0.9844, Validation Accuracy: 0.6339, Loss: 0.0629
    Epoch  72 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0086
    Epoch  72 Batch 2000/2400 - Train Accuracy: 0.9639, Validation Accuracy: 0.6339, Loss: 0.0517
    Epoch  73 Batch  500/2400 - Train Accuracy: 0.9577, Validation Accuracy: 0.6339, Loss: 0.0638
    Epoch  73 Batch 1000/2400 - Train Accuracy: 0.9598, Validation Accuracy: 0.6339, Loss: 0.0658
    Epoch  73 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0153
    Epoch  73 Batch 2000/2400 - Train Accuracy: 0.9832, Validation Accuracy: 0.6339, Loss: 0.0552
    Epoch  74 Batch  500/2400 - Train Accuracy: 0.9724, Validation Accuracy: 0.6339, Loss: 0.0785
    Epoch  74 Batch 1000/2400 - Train Accuracy: 0.9799, Validation Accuracy: 0.6339, Loss: 0.0437
    Epoch  74 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0095
    Epoch  74 Batch 2000/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6339, Loss: 0.0825
    Epoch  75 Batch  500/2400 - Train Accuracy: 0.9614, Validation Accuracy: 0.6339, Loss: 0.0679
    Epoch  75 Batch 1000/2400 - Train Accuracy: 0.9598, Validation Accuracy: 0.6339, Loss: 0.0646
    Epoch  75 Batch 1500/2400 - Train Accuracy: 0.9972, Validation Accuracy: 0.6339, Loss: 0.0091
    Epoch  75 Batch 2000/2400 - Train Accuracy: 0.9784, Validation Accuracy: 0.6339, Loss: 0.0625
    Epoch  76 Batch  500/2400 - Train Accuracy: 0.9614, Validation Accuracy: 0.6339, Loss: 0.0786
    Epoch  76 Batch 1000/2400 - Train Accuracy: 0.9911, Validation Accuracy: 0.6339, Loss: 0.0419
    Epoch  76 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0108
    Epoch  76 Batch 2000/2400 - Train Accuracy: 0.9736, Validation Accuracy: 0.6362, Loss: 0.0555
    Epoch  77 Batch  500/2400 - Train Accuracy: 0.9632, Validation Accuracy: 0.6362, Loss: 0.0579
    Epoch  77 Batch 1000/2400 - Train Accuracy: 0.9866, Validation Accuracy: 0.6339, Loss: 0.0388
    Epoch  77 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0063
    Epoch  77 Batch 2000/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6339, Loss: 0.0422
    Epoch  78 Batch  500/2400 - Train Accuracy: 0.9706, Validation Accuracy: 0.6362, Loss: 0.0727
    Epoch  78 Batch 1000/2400 - Train Accuracy: 0.9821, Validation Accuracy: 0.6339, Loss: 0.0257
    Epoch  78 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0131
    Epoch  78 Batch 2000/2400 - Train Accuracy: 0.9736, Validation Accuracy: 0.6362, Loss: 0.0612
    Epoch  79 Batch  500/2400 - Train Accuracy: 0.9835, Validation Accuracy: 0.6362, Loss: 0.0559
    Epoch  79 Batch 1000/2400 - Train Accuracy: 0.9888, Validation Accuracy: 0.6384, Loss: 0.0444
    Epoch  79 Batch 1500/2400 - Train Accuracy: 0.9972, Validation Accuracy: 0.6339, Loss: 0.0111
    Epoch  79 Batch 2000/2400 - Train Accuracy: 0.9591, Validation Accuracy: 0.6362, Loss: 0.0400
    Epoch  80 Batch  500/2400 - Train Accuracy: 0.9706, Validation Accuracy: 0.6339, Loss: 0.0448
    Epoch  80 Batch 1000/2400 - Train Accuracy: 0.9754, Validation Accuracy: 0.6362, Loss: 0.0429
    Epoch  80 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0081
    Epoch  80 Batch 2000/2400 - Train Accuracy: 0.9880, Validation Accuracy: 0.6339, Loss: 0.0639
    Epoch  81 Batch  500/2400 - Train Accuracy: 0.9559, Validation Accuracy: 0.6339, Loss: 0.0572
    Epoch  81 Batch 1000/2400 - Train Accuracy: 0.9665, Validation Accuracy: 0.6362, Loss: 0.0411
    Epoch  81 Batch 1500/2400 - Train Accuracy: 0.9972, Validation Accuracy: 0.6339, Loss: 0.0113
    Epoch  81 Batch 2000/2400 - Train Accuracy: 0.9760, Validation Accuracy: 0.6339, Loss: 0.0683
    Epoch  82 Batch  500/2400 - Train Accuracy: 0.9596, Validation Accuracy: 0.6339, Loss: 0.0489
    Epoch  82 Batch 1000/2400 - Train Accuracy: 0.9866, Validation Accuracy: 0.6339, Loss: 0.0312
    Epoch  82 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0150
    Epoch  82 Batch 2000/2400 - Train Accuracy: 0.9832, Validation Accuracy: 0.6339, Loss: 0.0417
    Epoch  83 Batch  500/2400 - Train Accuracy: 0.9651, Validation Accuracy: 0.6339, Loss: 0.0798
    Epoch  83 Batch 1000/2400 - Train Accuracy: 0.9754, Validation Accuracy: 0.6339, Loss: 0.0387
    Epoch  83 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0070
    Epoch  83 Batch 2000/2400 - Train Accuracy: 0.9952, Validation Accuracy: 0.6339, Loss: 0.0522
    Epoch  84 Batch  500/2400 - Train Accuracy: 0.9577, Validation Accuracy: 0.6339, Loss: 0.0522
    Epoch  84 Batch 1000/2400 - Train Accuracy: 0.9844, Validation Accuracy: 0.6339, Loss: 0.0495
    Epoch  84 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0116
    Epoch  84 Batch 2000/2400 - Train Accuracy: 0.9784, Validation Accuracy: 0.6339, Loss: 0.0565
    Epoch  85 Batch  500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6339, Loss: 0.0640
    Epoch  85 Batch 1000/2400 - Train Accuracy: 0.9911, Validation Accuracy: 0.6362, Loss: 0.0448
    Epoch  85 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0155
    Epoch  85 Batch 2000/2400 - Train Accuracy: 0.9567, Validation Accuracy: 0.6339, Loss: 0.0615
    Epoch  86 Batch  500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6339, Loss: 0.0496
    Epoch  86 Batch 1000/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6362, Loss: 0.0458
    Epoch  86 Batch 1500/2400 - Train Accuracy: 0.9943, Validation Accuracy: 0.6339, Loss: 0.0155
    Epoch  86 Batch 2000/2400 - Train Accuracy: 0.9880, Validation Accuracy: 0.6362, Loss: 0.0187
    Epoch  87 Batch  500/2400 - Train Accuracy: 0.9596, Validation Accuracy: 0.6339, Loss: 0.0506
    Epoch  87 Batch 1000/2400 - Train Accuracy: 0.9621, Validation Accuracy: 0.6339, Loss: 0.0269
    Epoch  87 Batch 1500/2400 - Train Accuracy: 0.9972, Validation Accuracy: 0.6339, Loss: 0.0077
    Epoch  87 Batch 2000/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6339, Loss: 0.0526
    Epoch  88 Batch  500/2400 - Train Accuracy: 0.9614, Validation Accuracy: 0.6339, Loss: 0.0491
    Epoch  88 Batch 1000/2400 - Train Accuracy: 0.9777, Validation Accuracy: 0.6339, Loss: 0.0569
    Epoch  88 Batch 1500/2400 - Train Accuracy: 0.9943, Validation Accuracy: 0.6339, Loss: 0.0110
    Epoch  88 Batch 2000/2400 - Train Accuracy: 0.9880, Validation Accuracy: 0.6339, Loss: 0.0480
    Epoch  89 Batch  500/2400 - Train Accuracy: 0.9651, Validation Accuracy: 0.6339, Loss: 0.0488
    Epoch  89 Batch 1000/2400 - Train Accuracy: 0.9911, Validation Accuracy: 0.6339, Loss: 0.0445
    Epoch  89 Batch 1500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6339, Loss: 0.0130
    Epoch  89 Batch 2000/2400 - Train Accuracy: 0.9856, Validation Accuracy: 0.6339, Loss: 0.0293


### Evaluate LSTM Net Only


```python
speaker_id, lexicon = list(lexicons.items())[0]
print("List of Speeches:", len(lexicon.speeches))
lexicon.evaluate_testset()
```


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
