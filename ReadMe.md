
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

for book_filename in text_paths[:15]: # 1 Book
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        lines = book_file.read()
        librispeech_corpus += lines
for stm_filename in stm_paths: # Process STM files (Tedlium)
        stm_segments.append(utils.parse_stm_file(stm_filename))
        

# Train on 3 speakers
for segments in stm_segments[15:18]: 
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
     As she took the hand, the girl blushed and
    half smiled, remembering the vaults and the baron
      Among other
    things, this means that no one owns a United States copyright
    on or for this work, so the Project (and you!) can copy and
    distribute it in the United States without permission and
    without paying copyright royalties
    
    I really did, Mr
      A tiny wall fountain
    modeled in classic pattern, for us penetrates into the world of
    the past, but for the Italian immigrant it may defy distance and
    barriers as he dimly responds to that typical beauty in which
    Italy has ever written its message, even as classic art knew no
    region of the gods which was not also sensuous, and as the art of
    Dante mysteriously blended the material and the spiritual
      Dr Gillette was at that time head of the latter
    institution; his scholarly explanation of the method of teaching,
    his concern for his charges, this sudden demonstration of the care
    the state bestowed upon its most unfortunate children, filled me
    with grave speculations in which the first, the fifth, or the
    ninth place in the oratorical contest seemed of little moment
     Under its influence they make all the progress
    compatible with the creed, and finally outgrow it; when a period follows
    of criticism and negation, in which mankind lose their old convictions
    without acquiring any new ones, of a general or authoritative character,
    except the conviction that the old are false
     In my rage I
    stamped my foot
    
    
    All speculation, however, on the possible future developments of my
    father's opinions, and on the probabilities of permanent co-operation
    between him and me in the promulgation of our thoughts, was doomed to be
    cut short
     Captain Wegg had been killed and old Thompson perhaps injured by a
    blow upon the head from which he had never recovered
      I thought of
    George Gravener confronted with such magnificence as that, and I
    asked what could have made two such persons ever suppose they
    understood each other
    
    Ground Truth sentences 0 to 10:
     As she took the hand, the girl blushed and
    half smiled, remembering the vaults and the baron
      Among other
    things, this means that no one owns a United States copyright
    on or for this work, so the Project (and you!) can copy and
    distribute it in the United States without permission and
    without paying copyright royalties
    
    I really did, Mr
      A tiny wall fountain
    modeled in classic pattern, for us penetrates into the world of
    the past, but for the Italian immigrant it may defy distance and
    barriers as he dimly responds to that typical beauty in which
    Italy has ever written its message, even as classic art knew no
    region of the gods which was not also sensuous, and as the art of
    Dante mysteriously blended the material and the spiritual
      Dr Gillette was at that time head of the latter
    institution; his scholarly explanation of the method of teaching,
    his concern for his charges, this sudden demonstration of the care
    the state bestowed upon its most unfortunate children, filled me
    with grave speculations in which the first, the fifth, or the
    ninth place in the oratorical contest seemed of little moment
     Under its influence they make all the progress
    compatible with the creed, and finally outgrow it; when a period follows
    of criticism and negation, in which mankind lose their old convictions
    without acquiring any new ones, of a general or authoritative character,
    except the conviction that the old are false
     In my rage I
    stamped my foot
    
    
    All speculation, however, on the possible future developments of my
    father's opinions, and on the probabilities of permanent co-operation
    between him and me in the promulgation of our thoughts, was doomed to be
    cut short
     Captain Wegg had been killed and old Thompson perhaps injured by a
    blow upon the head from which he had never recovered
      I thought of
    George Gravener confronted with such magnificence as that, and I
    asked what could have made two such persons ever suppose they
    understood each other
    
    Dataset Stats
    Roughly the number of unique words: 58051
    Number of sentences: 27600
    Average number of words in a sentence: 24.12873188405797
    
    Transcript sentences 0 to 10:
     The nieces did not tell her of their newly conceived
    hopes that the young couple would presently possess enough money to
    render their future comfortable, because there were so many chances that
    Bob West might win the little game being played
    
    
    The physical history of our planet shows us first an incandescent nebula
    dispersed over vast infinitudes of space; later this condenses into a
    central sun surrounded by a family of glowing planets hardly yet
    consolidated from the plastic primordial matter; then succeed untold
    millenniums of slow geological formation; an earth peopled by the lowest
    forms of life, whether vegetable or animal; from which crude beginnings a
    majestic, unceasing, unhurried, forward movement brings things stage by
    stage to the condition in which we know them now
     Don't tell uncle, but
    let us see what will come of it
     Yet we call them the Golden [-One,-] {+one,+} for they are not
    like the others
     it
    drops sharply from there on to the lake
     An idea flashed across his
    brain--perhaps evolved by the scratching
    
    
    Project Gutenberg-tm eBooks are often created from several printed
    editions, all of which are confirmed as Public Domain in the U
    ”
    
    “Those were my late master’s hours, sir
     You are also
    instructed to procure for Mr
     Saltram's want of dignity
    
    Ground Truth sentences 0 to 10:
     The nieces did not tell her of their newly conceived
    hopes that the young couple would presently possess enough money to
    render their future comfortable, because there were so many chances that
    Bob West might win the little game being played
    
    
    The physical history of our planet shows us first an incandescent nebula
    dispersed over vast infinitudes of space; later this condenses into a
    central sun surrounded by a family of glowing planets hardly yet
    consolidated from the plastic primordial matter; then succeed untold
    millenniums of slow geological formation; an earth peopled by the lowest
    forms of life, whether vegetable or animal; from which crude beginnings a
    majestic, unceasing, unhurried, forward movement brings things stage by
    stage to the condition in which we know them now
     Don't tell uncle, but
    let us see what will come of it
     Yet we call them the Golden [-One,-] {+one,+} for they are not
    like the others
     it
    drops sharply from there on to the lake
     An idea flashed across his
    brain--perhaps evolved by the scratching
    
    
    Project Gutenberg-tm eBooks are often created from several printed
    editions, all of which are confirmed as Public Domain in the U
    ”
    
    “Those were my late master’s hours, sir
     You are also
    instructed to procure for Mr
     Saltram's want of dignity
    
    Dataset Stats
    Roughly the number of unique words: 3057
    Number of sentences: 14
    Average number of words in a sentence: 627.0
    
    Transcript sentences 0 to 10:
    1272-128104-0000 MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
    1272-128104-0001 NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER
    1272-128104-0002 HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
    1272-128104-0003 HE HAS GRAVE DOUBTS WHETHER SIR FREDERICK LEIGHTON'S WORK IS REALLY GREEK AFTER ALL AND CAN DISCOVER IN IT BUT LITTLE OF ROCKY ITHACA
    1272-128104-0004 LINNELL'S PICTURES ARE A SORT OF UP GUARDS AND AT EM PAINTINGS AND MASON'S EXQUISITE IDYLLS ARE AS NATIONAL AS A JINGO POEM MISTER BIRKET FOSTER'S LANDSCAPES SMILE AT ONE MUCH IN THE SAME WAY THAT MISTER CARKER USED TO FLASH HIS TEETH AND MISTER JOHN COLLIER GIVES HIS SITTER A CHEERFUL SLAP ON THE BACK BEFORE HE SAYS LIKE A SHAMPOOER IN A TURKISH BATH NEXT MAN
    1272-128104-0005 IT IS OBVIOUSLY UNNECESSARY FOR US TO POINT OUT HOW LUMINOUS THESE CRITICISMS ARE HOW DELICATE IN EXPRESSION
    1272-128104-0006 ON THE GENERAL PRINCIPLES OF ART MISTER QUILTER WRITES WITH EQUAL LUCIDITY
    1272-128104-0007 PAINTING HE TELLS US IS OF A DIFFERENT QUALITY TO MATHEMATICS AND FINISH IN ART IS ADDING MORE FACT
    1272-128104-0008 AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN
    1272-128104-0009 HE LAMENTS MOST BITTERLY THE DIVORCE THAT HAS BEEN MADE BETWEEN DECORATIVE ART AND WHAT WE USUALLY CALL PICTURES MAKES THE CUSTOMARY APPEAL TO THE LAST JUDGMENT AND REMINDS US THAT IN THE GREAT DAYS OF ART MICHAEL ANGELO WAS THE FURNISHING UPHOLSTERER
    1272-128104-0010 NEAR THE FIRE AND THE ORNAMENTS FRED BROUGHT HOME FROM INDIA ON THE MANTEL BOARD
    1272-128104-0011 IN FACT HE IS QUITE SEVERE ON MISTER RUSKIN FOR NOT RECOGNISING THAT A PICTURE SHOULD DENOTE THE FRAILTY OF MAN AND REMARKS WITH PLEASING COURTESY AND FELICITOUS GRACE THAT MANY PHASES OF FEELING
    1272-128104-0012 ONLY UNFORTUNATELY HIS OWN WORK NEVER DOES GET GOOD
    1272-128104-0013 MISTER QUILTER HAS MISSED HIS CHANCE FOR HE HAS FAILED EVEN TO MAKE HIMSELF THE TUPPER OF PAINTING
    1272-128104-0014 BY HARRY QUILTER M A
    1272-135031-0000 BECAUSE YOU WERE SLEEPING INSTEAD OF CONQUERING THE LOVELY ROSE PRINCESS HAS BECOME A FIDDLE WITHOUT A BOW WHILE POOR SHAGGY SITS THERE A COOING DOVE
    1272-135031-0001 HE HAS GONE AND GONE FOR GOOD ANSWERED POLYCHROME WHO HAD MANAGED TO SQUEEZE INTO THE ROOM BESIDE THE DRAGON AND HAD WITNESSED THE OCCURRENCES WITH MUCH INTEREST
    1272-135031-0002 I HAVE REMAINED A PRISONER ONLY BECAUSE I WISHED TO BE ONE AND WITH THIS HE STEPPED FORWARD AND BURST THE STOUT CHAINS AS EASILY AS IF THEY HAD BEEN THREADS
    1272-135031-0003 THE LITTLE GIRL HAD BEEN ASLEEP BUT SHE HEARD THE RAPS AND OPENED THE DOOR
    1272-135031-0004 THE KING HAS FLED IN DISGRACE AND YOUR FRIENDS ARE ASKING FOR YOU
    1272-135031-0005 I BEGGED RUGGEDO LONG AGO TO SEND HIM AWAY BUT HE WOULD NOT DO SO
    1272-135031-0006 I ALSO OFFERED TO HELP YOUR BROTHER TO ESCAPE BUT HE WOULD NOT GO
    1272-135031-0007 HE EATS AND SLEEPS VERY STEADILY REPLIED THE NEW KING
    1272-135031-0008 I HOPE HE DOESN'T WORK TOO HARD SAID SHAGGY
    1272-135031-0009 HE DOESN'T WORK AT ALL
    1272-135031-0010 IN FACT THERE IS NOTHING HE CAN DO IN THESE DOMINIONS AS WELL AS OUR NOMES WHOSE NUMBERS ARE SO GREAT THAT IT WORRIES US TO KEEP THEM ALL BUSY
    1272-135031-0011 NOT EXACTLY RETURNED KALIKO
    1272-135031-0012 WHERE IS MY BROTHER NOW
    1272-135031-0013 INQUIRED SHAGGY IN THE METAL FOREST
    1272-135031-0014 WHERE IS THAT
    1272-135031-0015 THE METAL FOREST IS IN THE GREAT DOMED CAVERN THE LARGEST IN ALL OUR DOMINIONS REPLIED KALIKO
    1272-135031-0016 KALIKO HESITATED
    1272-135031-0017 HOWEVER IF WE LOOK SHARP WE MAY BE ABLE TO DISCOVER ONE OF THESE SECRET WAYS
    1272-135031-0018 OH NO I'M QUITE SURE HE DIDN'T
    1272-135031-0019 THAT'S FUNNY REMARKED BETSY THOUGHTFULLY
    1272-135031-0020 I DON'T BELIEVE ANN KNEW ANY MAGIC OR SHE'D HAVE WORKED IT BEFORE
    1272-135031-0021 I DO NOT KNOW CONFESSED SHAGGY
    1272-135031-0022 TRUE AGREED KALIKO
    1272-135031-0023 KALIKO WENT TO THE BIG GONG AND POUNDED ON IT JUST AS RUGGEDO USED TO DO BUT NO ONE ANSWERED THE SUMMONS
    1272-135031-0024 HAVING RETURNED TO THE ROYAL CAVERN KALIKO FIRST POUNDED THE GONG AND THEN SAT IN THE THRONE WEARING RUGGEDO'S DISCARDED RUBY CROWN AND HOLDING IN HIS HAND THE SCEPTRE WHICH RUGGEDO HAD SO OFTEN THROWN AT HIS HEAD
    1272-141231-0000 A MAN SAID TO THE UNIVERSE SIR I EXIST
    1272-141231-0001 SWEAT COVERED BRION'S BODY TRICKLING INTO THE TIGHT LOINCLOTH THAT WAS THE ONLY GARMENT HE WORE
    1272-141231-0002 THE CUT ON HIS CHEST STILL DRIPPING BLOOD THE ACHE OF HIS OVERSTRAINED EYES EVEN THE SOARING ARENA AROUND HIM WITH THE THOUSANDS OF SPECTATORS WERE TRIVIALITIES NOT WORTH THINKING ABOUT
    1272-141231-0003 HIS INSTANT OF PANIC WAS FOLLOWED BY A SMALL SHARP BLOW HIGH ON HIS CHEST
    1272-141231-0004 ONE MINUTE A VOICE SAID AND THE TIME BUZZER SOUNDED
    1272-141231-0005 A MINUTE IS NOT A VERY LARGE MEASURE OF TIME AND HIS BODY NEEDED EVERY FRACTION OF IT
    1272-141231-0006 THE BUZZER'S WHIRR TRIGGERED HIS MUSCLES INTO COMPLETE RELAXATION
    1272-141231-0007 ONLY HIS HEART AND LUNGS WORKED ON AT A STRONG MEASURED RATE
    1272-141231-0008 HE WAS IN REVERIE SLIDING ALONG THE BORDERS OF CONSCIOUSNESS
    1272-141231-0009 THE CONTESTANTS IN THE TWENTIES NEEDED UNDISTURBED REST THEREFORE NIGHTS IN THE DORMITORIES WERE AS QUIET AS DEATH
    1272-141231-0010 PARTICULARLY SO ON THIS LAST NIGHT WHEN ONLY TWO OF THE LITTLE CUBICLES WERE OCCUPIED THE THOUSANDS OF OTHERS STANDING WITH DARK EMPTY DOORS
    1272-141231-0011 THE OTHER VOICE SNAPPED WITH A HARSH URGENCY CLEARLY USED TO COMMAND
    1272-141231-0012 I'M HERE BECAUSE THE MATTER IS OF UTMOST IMPORTANCE AND BRANDD IS THE ONE I MUST SEE NOW STAND ASIDE
    1272-141231-0013 THE TWENTIES
    1272-141231-0014 HE MUST HAVE DRAWN HIS GUN BECAUSE THE INTRUDER SAID QUICKLY PUT THAT AWAY YOU'RE BEING A FOOL OUT
    1272-141231-0015 THERE WAS SILENCE THEN AND STILL WONDERING BRION WAS ONCE MORE ASLEEP
    1272-141231-0016 TEN SECONDS
    1272-141231-0017 HE ASKED THE HANDLER WHO WAS KNEADING HIS ACHING MUSCLES
    1272-141231-0018 A RED HAIRED MOUNTAIN OF A MAN WITH AN APPARENTLY INEXHAUSTIBLE STORE OF ENERGY
    1272-141231-0019 THERE COULD BE LITTLE ART IN THIS LAST AND FINAL ROUND OF FENCING
    1272-141231-0020 JUST THRUST AND PARRY AND VICTORY TO THE STRONGER
    1272-141231-0021 EVERY MAN WHO ENTERED THE TWENTIES HAD HIS OWN TRAINING TRICKS
    1272-141231-0022 THERE APPEARED TO BE AN IMMEDIATE ASSOCIATION WITH THE DEATH TRAUMA AS IF THE TWO WERE INEXTRICABLY LINKED INTO ONE
    1272-141231-0023 THE STRENGTH THAT ENABLES SOMEONE IN A TRANCE TO HOLD HIS BODY STIFF AND UNSUPPORTED EXCEPT AT TWO POINTS THE HEAD AND HEELS
    1272-141231-0024 THIS IS PHYSICALLY IMPOSSIBLE WHEN CONSCIOUS
    1272-141231-0025 OTHERS HAD DIED BEFORE DURING THE TWENTIES AND DEATH DURING THE LAST ROUND WAS IN SOME WAYS EASIER THAN DEFEAT
    1272-141231-0026 BREATHING DEEPLY BRION SOFTLY SPOKE THE AUTO HYPNOTIC PHRASES THAT TRIGGERED THE PROCESS
    1272-141231-0027 WHEN THE BUZZER SOUNDED HE PULLED HIS FOIL FROM HIS SECOND'S STARTLED GRASP AND RAN FORWARD
    1272-141231-0028 IROLG LOOKED AMAZED AT THE SUDDEN FURY OF THE ATTACK THEN SMILED
    1272-141231-0029 HE THOUGHT IT WAS A LAST BURST OF ENERGY HE KNEW HOW CLOSE THEY BOTH WERE TO EXHAUSTION
    1272-141231-0030 BRION SAW SOMETHING CLOSE TO PANIC ON HIS OPPONENT'S FACE WHEN THE MAN FINALLY RECOGNIZED HIS ERROR
    1272-141231-0031 A WAVE OF DESPAIR ROLLED OUT FROM IROLG BRION SENSED IT AND KNEW THE FIFTH POINT WAS HIS
    1272-141231-0032 THEN THE POWERFUL TWIST THAT THRUST IT ASIDE IN AND UNDER THE GUARD
    1462-170138-0000 HE HAD WRITTEN A NUMBER OF BOOKS HIMSELF AMONG THEM A HISTORY OF DANCING A HISTORY OF COSTUME A KEY TO SHAKESPEARE'S SONNETS A STUDY OF THE POETRY OF ERNEST DOWSON ET CETERA
    1462-170138-0001 HUGH'S WRITTEN A DELIGHTFUL PART FOR HER AND SHE'S QUITE INEXPRESSIBLE
    1462-170138-0002 I HAPPEN TO HAVE MAC CONNELL'S BOX FOR TONIGHT OR THERE'D BE NO CHANCE OF OUR GETTING PLACES
    1462-170138-0003 ALEXANDER EXCLAIMED MILDLY
    1462-170138-0004 MYSELF I ALWAYS KNEW SHE HAD IT IN HER
    1462-170138-0005 DO YOU KNOW ALEXANDER MAINHALL LOOKED WITH PERPLEXITY UP INTO THE TOP OF THE HANSOM AND RUBBED HIS PINK CHEEK WITH HIS GLOVED FINGER DO YOU KNOW I SOMETIMES THINK OF TAKING TO CRITICISM SERIOUSLY MYSELF
    1462-170138-0006 WHEN THEY ENTERED THE STAGE BOX ON THE LEFT THE FIRST ACT WAS WELL UNDER WAY THE SCENE BEING THE INTERIOR OF A CABIN IN THE SOUTH OF IRELAND
    1462-170138-0007 AS THEY SAT DOWN A BURST OF APPLAUSE DREW ALEXANDER'S ATTENTION TO THE STAGE
    1462-170138-0008 OF COURSE HILDA IS IRISH THE BURGOYNES HAVE BEEN STAGE PEOPLE FOR GENERATIONS AND SHE HAS THE IRISH VOICE
    1462-170138-0009 IT'S DELIGHTFUL TO HEAR IT IN A LONDON THEATRE
    1462-170138-0010 WHEN SHE BEGAN TO DANCE BY WAY OF SHOWING THE GOSSOONS WHAT SHE HAD SEEN IN THE FAIRY RINGS AT NIGHT THE HOUSE BROKE INTO A PROLONGED UPROAR
    1462-170138-0011 AFTER HER DANCE SHE WITHDREW FROM THE DIALOGUE AND RETREATED TO THE DITCH WALL BACK OF PHILLY'S BURROW WHERE SHE SAT SINGING THE RISING OF THE MOON AND MAKING A WREATH OF PRIMROSES FOR HER DONKEY
    1462-170138-0012 MAC CONNELL LET ME INTRODUCE MISTER BARTLEY ALEXANDER
    1462-170138-0013 THE PLAYWRIGHT GAVE MAINHALL A CURIOUS LOOK OUT OF HIS DEEP SET FADED EYES AND MADE A WRY FACE
    1462-170138-0014 HE NODDED CURTLY AND MADE FOR THE DOOR DODGING ACQUAINTANCES AS HE WENT
    1462-170138-0015 I DARE SAY IT'S QUITE TRUE THAT THERE'S NEVER BEEN ANY ONE ELSE
    1462-170138-0016 HE'S ANOTHER WHO'S AWFULLY KEEN ABOUT HER LET ME INTRODUCE YOU
    1462-170138-0017 SIR HARRY TOWNE BOWED AND SAID THAT HE HAD MET MISTER ALEXANDER AND HIS WIFE IN TOKYO
    1462-170138-0018 I SAY SIR HARRY THE LITTLE GIRL'S GOING FAMOUSLY TO NIGHT ISN'T SHE
    1462-170138-0019 THE FACT IS SHE'S FEELING RATHER SEEDY POOR CHILD
    1462-170138-0020 A LITTLE ATTACK OF NERVES POSSIBLY
    1462-170138-0021 HE BOWED AS THE WARNING BELL RANG AND MAINHALL WHISPERED YOU KNOW LORD WESTMERE OF COURSE THE STOOPED MAN WITH THE LONG GRAY MUSTACHE TALKING TO LADY DOWLE
    1462-170138-0022 IN A MOMENT PEGGY WAS ON THE STAGE AGAIN AND ALEXANDER APPLAUDED VIGOROUSLY WITH THE REST
    1462-170138-0023 IN THE HALF LIGHT HE LOOKED ABOUT AT THE STALLS AND BOXES AND SMILED A LITTLE CONSCIOUSLY RECALLING WITH AMUSEMENT SIR HARRY'S JUDICIAL FROWN
    1462-170138-0024 HE LEANED FORWARD AND BEAMED FELICITATIONS AS WARMLY AS MAINHALL HIMSELF WHEN AT THE END OF THE PLAY SHE CAME AGAIN AND AGAIN BEFORE THE CURTAIN PANTING A LITTLE AND FLUSHED HER EYES DANCING AND HER EAGER NERVOUS LITTLE MOUTH TREMULOUS WITH EXCITEMENT
    1462-170138-0025 ALL THE SAME HE LIFTED HIS GLASS HERE'S TO YOU LITTLE HILDA
    1462-170138-0026 I'M GLAD SHE'S HELD HER OWN SINCE
    1462-170138-0027 IT WAS YOUTH AND POVERTY AND PROXIMITY AND EVERYTHING WAS YOUNG AND KINDLY
    1462-170142-0000 THE LAST TWO DAYS OF THE VOYAGE BARTLEY FOUND ALMOST INTOLERABLE
    1462-170142-0001 EMERGING AT EUSTON AT HALF PAST THREE O'CLOCK IN THE AFTERNOON ALEXANDER HAD HIS LUGGAGE SENT TO THE SAVOY AND DROVE AT ONCE TO BEDFORD SQUARE
    1462-170142-0002 SHE BLUSHED AND SMILED AND FUMBLED HIS CARD IN HER CONFUSION BEFORE SHE RAN UPSTAIRS
    1462-170142-0003 THE ROOM WAS EMPTY WHEN HE ENTERED
    1462-170142-0004 A COAL FIRE WAS CRACKLING IN THE GRATE AND THE LAMPS WERE LIT FOR IT WAS ALREADY BEGINNING TO GROW DARK OUTSIDE
    1462-170142-0005 SHE CALLED HIS NAME ON THE THRESHOLD BUT IN HER SWIFT FLIGHT ACROSS THE ROOM SHE FELT A CHANGE IN HIM AND CAUGHT HERSELF UP SO DEFTLY THAT HE COULD NOT TELL JUST WHEN SHE DID IT
    1462-170142-0006 SHE MERELY BRUSHED HIS CHEEK WITH HER LIPS AND PUT A HAND LIGHTLY AND JOYOUSLY ON EITHER SHOULDER
    1462-170142-0007 I NEVER DREAMED IT WOULD BE YOU BARTLEY
    1462-170142-0008 WHEN DID YOU COME BARTLEY AND HOW DID IT HAPPEN YOU HAVEN'T SPOKEN A WORD
    1462-170142-0009 SHE LOOKED AT HIS HEAVY SHOULDERS AND BIG DETERMINED HEAD THRUST FORWARD LIKE A CATAPULT IN LEASH
    1462-170142-0010 I'LL DO ANYTHING YOU WISH ME TO BARTLEY SHE SAID TREMULOUSLY
    1462-170142-0011 HE PULLED UP A WINDOW AS IF THE AIR WERE HEAVY
    1462-170142-0012 HILDA WATCHED HIM FROM HER CORNER TREMBLING AND SCARCELY BREATHING DARK SHADOWS GROWING ABOUT HER EYES
    1462-170142-0013 IT IT HASN'T ALWAYS MADE YOU MISERABLE HAS IT
    1462-170142-0014 ALWAYS BUT IT'S WORSE NOW
    1462-170142-0015 IT'S UNBEARABLE IT TORTURES ME EVERY MINUTE
    1462-170142-0016 I AM NOT A MAN WHO CAN LIVE TWO LIVES HE WENT ON FEVERISHLY EACH LIFE SPOILS THE OTHER
    1462-170142-0017 I GET NOTHING BUT MISERY OUT OF EITHER
    1462-170142-0018 THERE IS THIS DECEPTION BETWEEN ME AND EVERYTHING
    1462-170142-0019 AT THAT WORD DECEPTION SPOKEN WITH SUCH SELF CONTEMPT THE COLOR FLASHED BACK INTO HILDA'S FACE AS SUDDENLY AS IF SHE HAD BEEN STRUCK BY A WHIPLASH
    1462-170142-0020 SHE BIT HER LIP AND LOOKED DOWN AT HER HANDS WHICH WERE CLASPED TIGHTLY IN FRONT OF HER
    1462-170142-0021 COULD YOU COULD YOU SIT DOWN AND TALK ABOUT IT QUIETLY BARTLEY AS IF I WERE A FRIEND AND NOT SOME ONE WHO HAD TO BE DEFIED
    1462-170142-0022 HE DROPPED BACK HEAVILY INTO HIS CHAIR BY THE FIRE
    1462-170142-0023 I HAVE THOUGHT ABOUT IT UNTIL I AM WORN OUT
    1462-170142-0024 AFTER THE VERY FIRST
    1462-170142-0025 HILDA'S FACE QUIVERED BUT SHE WHISPERED YES I THINK IT MUST HAVE BEEN
    1462-170142-0026 SHE PRESSED HIS HAND GENTLY IN GRATITUDE WEREN'T YOU HAPPY THEN AT ALL
    1462-170142-0027 SOMETHING OF THEIR TROUBLING SWEETNESS CAME BACK TO ALEXANDER TOO
    1462-170142-0028 PRESENTLY IT STOLE BACK TO HIS COAT SLEEVE
    1462-170142-0029 YES HILDA I KNOW THAT HE SAID SIMPLY
    1462-170142-0030 I UNDERSTAND BARTLEY I WAS WRONG
    1462-170142-0031 SHE LISTENED INTENTLY BUT SHE HEARD NOTHING BUT THE CREAKING OF HIS CHAIR
    1462-170142-0032 YOU WANT ME TO SAY IT SHE WHISPERED
    1462-170142-0033 BARTLEY LEANED HIS HEAD IN HIS HANDS AND SPOKE THROUGH HIS TEETH
    1462-170142-0034 IT'S GOT TO BE A CLEAN BREAK HILDA
    1462-170142-0035 OH BARTLEY WHAT AM I TO DO
    1462-170142-0036 YOU ASK ME TO STAY AWAY FROM YOU BECAUSE YOU WANT ME
    1462-170142-0037 I WILL ASK THE LEAST IMAGINABLE BUT I MUST HAVE SOMETHING
    1462-170142-0038 HILDA SAT ON THE ARM OF IT AND PUT HER HANDS LIGHTLY ON HIS SHOULDERS
    1462-170142-0039 YOU SEE LOVING SOME ONE AS I LOVE YOU MAKES THE WHOLE WORLD DIFFERENT
    1462-170142-0040 AND THEN YOU CAME BACK NOT CARING VERY MUCH BUT IT MADE NO DIFFERENCE
    1462-170142-0041 SHE SLID TO THE FLOOR BESIDE HIM AS IF SHE WERE TOO TIRED TO SIT UP ANY LONGER
    1462-170142-0042 DON'T CRY DON'T CRY HE WHISPERED
    1462-170145-0000 ON THE LAST SATURDAY IN APRIL THE NEW YORK TIMES PUBLISHED AN ACCOUNT OF THE STRIKE COMPLICATIONS WHICH WERE DELAYING ALEXANDER'S NEW JERSEY BRIDGE AND STATED THAT THE ENGINEER HIMSELF WAS IN TOWN AND AT HIS OFFICE ON WEST TENTH STREET
    1462-170145-0001 OVER THE FIREPLACE THERE WAS A LARGE OLD FASHIONED GILT MIRROR
    1462-170145-0002 HE ROSE AND CROSSED THE ROOM QUICKLY
    1462-170145-0003 OF COURSE I KNOW BARTLEY SHE SAID AT LAST THAT AFTER THIS YOU WON'T OWE ME THE LEAST CONSIDERATION BUT WE SAIL ON TUESDAY
    1462-170145-0004 I SAW THAT INTERVIEW IN THE PAPER YESTERDAY TELLING WHERE YOU WERE AND I THOUGHT I HAD TO SEE YOU THAT'S ALL GOOD NIGHT I'M GOING NOW
    1462-170145-0005 LET ME TAKE OFF YOUR COAT AND YOUR BOOTS THEY'RE OOZING WATER
    1462-170145-0006 IF YOU'D SENT ME A NOTE OR TELEPHONED ME OR ANYTHING
    1462-170145-0007 I TOLD MYSELF THAT IF I WERE REALLY THINKING OF YOU AND NOT OF MYSELF A LETTER WOULD BE BETTER THAN NOTHING
    1462-170145-0008 HE PAUSED THEY NEVER DID TO ME
    1462-170145-0009 OH BARTLEY DID YOU WRITE TO ME
    1462-170145-0010 ALEXANDER SLIPPED HIS ARM ABOUT HER
    1462-170145-0011 I THINK I HAVE FELT THAT YOU WERE COMING
    1462-170145-0012 HE BENT HIS FACE OVER HER HAIR
    1462-170145-0013 AND I SHE WHISPERED I FELT THAT YOU WERE FEELING THAT
    1462-170145-0014 BUT WHEN I CAME I THOUGHT I HAD BEEN MISTAKEN
    1462-170145-0015 I'VE BEEN UP IN CANADA WITH MY BRIDGE AND I ARRANGED NOT TO COME TO NEW YORK UNTIL AFTER YOU HAD GONE
    1462-170145-0016 THEN WHEN YOUR MANAGER ADDED TWO MORE WEEKS I WAS ALREADY COMMITTED
    1462-170145-0017 I'M GOING TO DO WHAT YOU ASKED ME TO DO WHEN YOU WERE IN LONDON
    1462-170145-0018 ONLY I'LL DO IT MORE COMPLETELY
    1462-170145-0019 THEN YOU DON'T KNOW WHAT YOU'RE TALKING ABOUT
    1462-170145-0020 YES I KNOW VERY WELL
    1462-170145-0021 ALEXANDER FLUSHED ANGRILY
    1462-170145-0022 I DON'T KNOW WHAT I OUGHT TO SAY BUT I DON'T BELIEVE YOU'D BE HAPPY TRULY I DON'T AREN'T YOU TRYING TO FRIGHTEN ME
    1673-143396-0000 A LAUDABLE REGARD FOR THE HONOR OF THE FIRST PROSELYTE HAS COUNTENANCED THE BELIEF THE HOPE THE WISH THAT THE EBIONITES OR AT LEAST THE NAZARENES WERE DISTINGUISHED ONLY BY THEIR OBSTINATE PERSEVERANCE IN THE PRACTICE OF THE MOSAIC RITES
    1673-143396-0001 THEIR CHURCHES HAVE DISAPPEARED THEIR BOOKS ARE OBLITERATED THEIR OBSCURE FREEDOM MIGHT ALLOW A LATITUDE OF FAITH AND THE SOFTNESS OF THEIR INFANT CREED WOULD BE VARIOUSLY MOULDED BY THE ZEAL OR PRUDENCE OF THREE HUNDRED YEARS
    1673-143396-0002 YET THE MOST CHARITABLE CRITICISM MUST REFUSE THESE SECTARIES ANY KNOWLEDGE OF THE PURE AND PROPER DIVINITY OF CHRIST
    1673-143396-0003 HIS PROGRESS FROM INFANCY TO YOUTH AND MANHOOD WAS MARKED BY A REGULAR INCREASE IN STATURE AND WISDOM AND AFTER A PAINFUL AGONY OF MIND AND BODY HE EXPIRED ON THE CROSS
    1673-143396-0004 HE LIVED AND DIED FOR THE SERVICE OF MANKIND BUT THE LIFE AND DEATH OF SOCRATES HAD LIKEWISE BEEN DEVOTED TO THE CAUSE OF RELIGION AND JUSTICE AND ALTHOUGH THE STOIC OR THE HERO MAY DISDAIN THE HUMBLE VIRTUES OF JESUS THE TEARS WHICH HE SHED OVER HIS FRIEND AND COUNTRY MAY BE ESTEEMED THE PUREST EVIDENCE OF HIS HUMANITY
    1673-143396-0005 THE SON OF A VIRGIN GENERATED BY THE INEFFABLE OPERATION OF THE HOLY SPIRIT WAS A CREATURE WITHOUT EXAMPLE OR RESEMBLANCE SUPERIOR IN EVERY ATTRIBUTE OF MIND AND BODY TO THE CHILDREN OF ADAM
    1673-143396-0006 NOR COULD IT SEEM STRANGE OR INCREDIBLE THAT THE FIRST OF THESE AEONS THE LOGOS OR WORD OF GOD OF THE SAME SUBSTANCE WITH THE FATHER SHOULD DESCEND UPON EARTH TO DELIVER THE HUMAN RACE FROM VICE AND ERROR AND TO CONDUCT THEM IN THE PATHS OF LIFE AND IMMORTALITY
    1673-143396-0007 BUT THE PREVAILING DOCTRINE OF THE ETERNITY AND INHERENT PRAVITY OF MATTER INFECTED THE PRIMITIVE CHURCHES OF THE EAST
    1673-143396-0008 MANY AMONG THE GENTILE PROSELYTES REFUSED TO BELIEVE THAT A CELESTIAL SPIRIT AN UNDIVIDED PORTION OF THE FIRST ESSENCE HAD BEEN PERSONALLY UNITED WITH A MASS OF IMPURE AND CONTAMINATED FLESH AND IN THEIR ZEAL FOR THE DIVINITY THEY PIOUSLY ABJURED THE HUMANITY OF CHRIST
    1673-143396-0009 HE FIRST APPEARED ON THE BANKS OF THE JORDAN IN THE FORM OF PERFECT MANHOOD BUT IT WAS A FORM ONLY AND NOT A SUBSTANCE A HUMAN FIGURE CREATED BY THE HAND OF OMNIPOTENCE TO IMITATE THE FACULTIES AND ACTIONS OF A MAN AND TO IMPOSE A PERPETUAL ILLUSION ON THE SENSES OF HIS FRIENDS AND ENEMIES
    1673-143396-0010 BUT THE RASHNESS OF THESE CONCESSIONS HAS ENCOURAGED A MILDER SENTIMENT OF THOSE OF THE DOCETES WHO TAUGHT NOT THAT CHRIST WAS A PHANTOM BUT THAT HE WAS CLOTHED WITH AN IMPASSIBLE AND INCORRUPTIBLE BODY
    1673-143396-0011 A FOETUS THAT COULD INCREASE FROM AN INVISIBLE POINT TO ITS FULL MATURITY A CHILD THAT COULD ATTAIN THE STATURE OF PERFECT MANHOOD WITHOUT DERIVING ANY NOURISHMENT FROM THE ORDINARY SOURCES MIGHT CONTINUE TO EXIST WITHOUT REPAIRING A DAILY WASTE BY A DAILY SUPPLY OF EXTERNAL MATTER
    1673-143396-0012 IN THEIR EYES JESUS OF NAZARETH WAS A MERE MORTAL THE LEGITIMATE SON OF JOSEPH AND MARY BUT HE WAS THE BEST AND WISEST OF THE HUMAN RACE SELECTED AS THE WORTHY INSTRUMENT TO RESTORE UPON EARTH THE WORSHIP OF THE TRUE AND SUPREME DEITY
    1673-143396-0013 WHEN THE MESSIAH WAS DELIVERED INTO THE HANDS OF THE JEWS THE CHRIST AN IMMORTAL AND IMPASSIBLE BEING FORSOOK HIS EARTHLY TABERNACLE FLEW BACK TO THE PLEROMA OR WORLD OF SPIRITS AND LEFT THE SOLITARY JESUS TO SUFFER TO COMPLAIN AND TO EXPIRE
    1673-143396-0014 BUT THE JUSTICE AND GENEROSITY OF SUCH A DESERTION ARE STRONGLY QUESTIONABLE AND THE FATE OF AN INNOCENT MARTYR AT FIRST IMPELLED AND AT LENGTH ABANDONED BY HIS DIVINE COMPANION MIGHT PROVOKE THE PITY AND INDIGNATION OF THE PROFANE
    1673-143396-0015 THEIR MURMURS WERE VARIOUSLY SILENCED BY THE SECTARIES WHO ESPOUSED AND MODIFIED THE DOUBLE SYSTEM OF CERINTHUS
    1673-143396-0016 THE WORTHY FRIEND OF ATHANASIUS THE WORTHY ANTAGONIST OF JULIAN HE BRAVELY WRESTLED WITH THE ARIANS AND POLYTHEISTS AND THOUGH HE AFFECTED THE RIGOR OF GEOMETRICAL DEMONSTRATION HIS COMMENTARIES REVEALED THE LITERAL AND ALLEGORICAL SENSE OF THE SCRIPTURES
    1673-143396-0017 YET AS THE PROFOUND DOCTOR HAD BEEN TERRIFIED AT HIS OWN RASHNESS APOLLINARIS WAS HEARD TO MUTTER SOME FAINT ACCENTS OF EXCUSE AND EXPLANATION
    1673-143396-0018 HE ACQUIESCED IN THE OLD DISTINCTION OF THE GREEK PHILOSOPHERS BETWEEN THE RATIONAL AND SENSITIVE SOUL OF MAN THAT HE MIGHT RESERVE THE LOGOS FOR INTELLECTUAL FUNCTIONS AND EMPLOY THE SUBORDINATE HUMAN PRINCIPLE IN THE MEANER ACTIONS OF ANIMAL LIFE
    1673-143396-0019 BUT INSTEAD OF A TEMPORARY AND OCCASIONAL ALLIANCE THEY ESTABLISHED AND WE STILL EMBRACE THE SUBSTANTIAL INDISSOLUBLE AND EVERLASTING UNION OF A PERFECT GOD WITH A PERFECT MAN OF THE SECOND PERSON OF THE TRINITY WITH A REASONABLE SOUL AND HUMAN FLESH
    1673-143396-0020 UNDER THE TUITION OF THE ABBOT SERAPION HE APPLIED HIMSELF TO ECCLESIASTICAL STUDIES WITH SUCH INDEFATIGABLE ARDOR THAT IN THE COURSE OF ONE SLEEPLESS NIGHT HE HAS PERUSED THE FOUR GOSPELS THE CATHOLIC EPISTLES AND THE EPISTLE TO THE ROMANS
    1673-143397-0000 ARDENT IN THE PROSECUTION OF HERESY CYRIL AUSPICIOUSLY OPENED HIS REIGN BY OPPRESSING THE NOVATIANS THE MOST INNOCENT AND HARMLESS OF THE SECTARIES
    1673-143397-0001 WITHOUT ANY LEGAL SENTENCE WITHOUT ANY ROYAL MANDATE THE PATRIARCH AT THE DAWN OF DAY LED A SEDITIOUS MULTITUDE TO THE ATTACK OF THE SYNAGOGUES
    1673-143397-0002 SUCH CRIMES WOULD HAVE DESERVED THE ANIMADVERSION OF THE MAGISTRATE BUT IN THIS PROMISCUOUS OUTRAGE THE INNOCENT WERE CONFOUNDED WITH THE GUILTY AND ALEXANDRIA WAS IMPOVERISHED BY THE LOSS OF A WEALTHY AND INDUSTRIOUS COLONY
    1673-143397-0003 THE ZEAL OF CYRIL EXPOSED HIM TO THE PENALTIES OF THE JULIAN LAW BUT IN A FEEBLE GOVERNMENT AND A SUPERSTITIOUS AGE HE WAS SECURE OF IMPUNITY AND EVEN OF PRAISE
    1673-143397-0004 ORESTES COMPLAINED BUT HIS JUST COMPLAINTS WERE TOO QUICKLY FORGOTTEN BY THE MINISTERS OF THEODOSIUS AND TOO DEEPLY REMEMBERED BY A PRIEST WHO AFFECTED TO PARDON AND CONTINUED TO HATE THE PRAEFECT OF EGYPT
    1673-143397-0005 A RUMOR WAS SPREAD AMONG THE CHRISTIANS THAT THE DAUGHTER OF THEON WAS THE ONLY OBSTACLE TO THE RECONCILIATION OF THE PRAEFECT AND THE ARCHBISHOP AND THAT OBSTACLE WAS SPEEDILY REMOVED
    1673-143397-0006 WHICH OPPRESSED THE METROPOLITANS OF EUROPE AND ASIA INVADED THE PROVINCES OF ANTIOCH AND ALEXANDRIA AND MEASURED THEIR DIOCESE BY THE LIMITS OF THE EMPIRE
    1673-143397-0007 EXTERMINATE WITH ME THE HERETICS AND WITH YOU I WILL EXTERMINATE THE PERSIANS
    1673-143397-0008 AT THESE BLASPHEMOUS SOUNDS THE PILLARS OF THE SANCTUARY WERE SHAKEN
    1673-143397-0009 BUT THE VATICAN RECEIVED WITH OPEN ARMS THE MESSENGERS OF EGYPT
    1673-143397-0010 THE VANITY OF CELESTINE WAS FLATTERED BY THE APPEAL AND THE PARTIAL VERSION OF A MONK DECIDED THE FAITH OF THE POPE WHO WITH HIS LATIN CLERGY WAS IGNORANT OF THE LANGUAGE THE ARTS AND THE THEOLOGY OF THE GREEKS
    1673-143397-0011 NESTORIUS WHO DEPENDED ON THE NEAR APPROACH OF HIS EASTERN FRIENDS PERSISTED LIKE HIS PREDECESSOR CHRYSOSTOM TO DISCLAIM THE JURISDICTION AND TO DISOBEY THE SUMMONS OF HIS ENEMIES THEY HASTENED HIS TRIAL AND HIS ACCUSER PRESIDED IN THE SEAT OF JUDGMENT
    1673-143397-0012 SIXTY EIGHT BISHOPS TWENTY TWO OF METROPOLITAN RANK DEFENDED HIS CAUSE BY A MODEST AND TEMPERATE PROTEST THEY WERE EXCLUDED FROM THE COUNCILS OF THEIR BRETHREN
    1673-143397-0013 BY THE VIGILANCE OF MEMNON THE CHURCHES WERE SHUT AGAINST THEM AND A STRONG GARRISON WAS THROWN INTO THE CATHEDRAL
    1673-143397-0014 DURING A BUSY PERIOD OF THREE MONTHS THE EMPEROR TRIED EVERY METHOD EXCEPT THE MOST EFFECTUAL MEANS OF INDIFFERENCE AND CONTEMPT TO RECONCILE THIS THEOLOGICAL QUARREL
    1673-143397-0015 RETURN TO YOUR PROVINCES AND MAY YOUR PRIVATE VIRTUES REPAIR THE MISCHIEF AND SCANDAL OF YOUR MEETING
    1673-143397-0016 THE FEEBLE SON OF ARCADIUS WAS ALTERNATELY SWAYED BY HIS WIFE AND SISTER BY THE EUNUCHS AND WOMEN OF THE PALACE SUPERSTITION AND AVARICE WERE THEIR RULING PASSIONS AND THE ORTHODOX CHIEFS WERE ASSIDUOUS IN THEIR ENDEAVORS TO ALARM THE FORMER AND TO GRATIFY THE LATTER
    1673-143397-0017 BUT IN THIS AWFUL MOMENT OF THE DANGER OF THE CHURCH THEIR VOW WAS SUPERSEDED BY A MORE SUBLIME AND INDISPENSABLE DUTY
    1673-143397-0018 AT THE SAME TIME EVERY AVENUE OF THE THRONE WAS ASSAULTED WITH GOLD
    1673-143397-0019 THE PAST HE REGRETTED HE WAS DISCONTENTED WITH THE PRESENT AND THE FUTURE HE HAD REASON TO DREAD THE ORIENTAL BISHOPS SUCCESSIVELY DISENGAGED THEIR CAUSE FROM HIS UNPOPULAR NAME AND EACH DAY DECREASED THE NUMBER OF THE SCHISMATICS WHO REVERED NESTORIUS AS THE CONFESSOR OF THE FAITH
    1673-143397-0020 A WANDERING TRIBE OF THE BLEMMYES OR NUBIANS INVADED HIS SOLITARY PRISON IN THEIR RETREAT THEY DISMISSED A CROWD OF USELESS CAPTIVES BUT NO SOONER HAD NESTORIUS REACHED THE BANKS OF THE NILE THAN HE WOULD GLADLY HAVE ESCAPED FROM A ROMAN AND ORTHODOX CITY TO THE MILDER SERVITUDE OF THE SAVAGES
    174-168635-0000 HE HAD NEVER BEEN FATHER LOVER HUSBAND FRIEND
    174-168635-0001 THE HEART OF THAT EX CONVICT WAS FULL OF VIRGINITY
    174-168635-0002 HIS SISTER AND HIS SISTER'S CHILDREN HAD LEFT HIM ONLY A VAGUE AND FAR OFF MEMORY WHICH HAD FINALLY ALMOST COMPLETELY VANISHED HE HAD MADE EVERY EFFORT TO FIND THEM AND NOT HAVING BEEN ABLE TO FIND THEM HE HAD FORGOTTEN THEM
    174-168635-0003 HE SUFFERED ALL THE PANGS OF A MOTHER AND HE KNEW NOT WHAT IT MEANT FOR THAT GREAT AND SINGULAR MOVEMENT OF A HEART WHICH BEGINS TO LOVE IS A VERY OBSCURE AND A VERY SWEET THING
    174-168635-0004 ONLY AS HE WAS FIVE AND FIFTY AND COSETTE EIGHT YEARS OF AGE ALL THAT MIGHT HAVE BEEN LOVE IN THE WHOLE COURSE OF HIS LIFE FLOWED TOGETHER INTO A SORT OF INEFFABLE LIGHT
    174-168635-0005 COSETTE ON HER SIDE HAD ALSO UNKNOWN TO HERSELF BECOME ANOTHER BEING POOR LITTLE THING
    174-168635-0006 SHE FELT THAT WHICH SHE HAD NEVER FELT BEFORE A SENSATION OF EXPANSION
    174-168635-0007 THE MAN NO LONGER PRODUCED ON HER THE EFFECT OF BEING OLD OR POOR SHE THOUGHT JEAN VALJEAN HANDSOME JUST AS SHE THOUGHT THE HOVEL PRETTY
    174-168635-0008 NATURE A DIFFERENCE OF FIFTY YEARS HAD SET A PROFOUND GULF BETWEEN JEAN VALJEAN AND COSETTE DESTINY FILLED IN THIS GULF
    174-168635-0009 TO MEET WAS TO FIND EACH OTHER
    174-168635-0010 WHEN THESE TWO SOULS PERCEIVED EACH OTHER THEY RECOGNIZED EACH OTHER AS NECESSARY TO EACH OTHER AND EMBRACED EACH OTHER CLOSELY
    174-168635-0011 MOREOVER JEAN VALJEAN HAD CHOSEN HIS REFUGE WELL
    174-168635-0012 HE HAD PAID HER SIX MONTHS IN ADVANCE AND HAD COMMISSIONED THE OLD WOMAN TO FURNISH THE CHAMBER AND DRESSING ROOM AS WE HAVE SEEN
    174-168635-0013 WEEK FOLLOWED WEEK THESE TWO BEINGS LED A HAPPY LIFE IN THAT HOVEL
    174-168635-0014 COSETTE WAS NO LONGER IN RAGS SHE WAS IN MOURNING
    174-168635-0015 AND THEN HE TALKED OF HER MOTHER AND HE MADE HER PRAY
    174-168635-0016 HE PASSED HOURS IN WATCHING HER DRESSING AND UNDRESSING HER DOLL AND IN LISTENING TO HER PRATTLE
    174-168635-0017 THE BEST OF US ARE NOT EXEMPT FROM EGOTISTICAL THOUGHTS
    174-168635-0018 HE HAD RETURNED TO PRISON THIS TIME FOR HAVING DONE RIGHT HE HAD QUAFFED FRESH BITTERNESS DISGUST AND LASSITUDE WERE OVERPOWERING HIM EVEN THE MEMORY OF THE BISHOP PROBABLY SUFFERED A TEMPORARY ECLIPSE THOUGH SURE TO REAPPEAR LATER ON LUMINOUS AND TRIUMPHANT BUT AFTER ALL THAT SACRED MEMORY WAS GROWING DIM
    174-168635-0019 WHO KNOWS WHETHER JEAN VALJEAN HAD NOT BEEN ON THE EVE OF GROWING DISCOURAGED AND OF FALLING ONCE MORE
    174-168635-0020 ALAS HE WALKED WITH NO LESS INDECISION THAN COSETTE
    174-168635-0021 HE PROTECTED HER AND SHE STRENGTHENED HIM
    174-168635-0022 HE WAS THAT CHILD'S STAY AND SHE WAS HIS PROP
    174-50561-0000 FORGOTTEN TOO THE NAME OF GILLIAN THE LOVELY CAPTIVE
    174-50561-0001 WORSE AND WORSE HE IS EVEN PRESUMED TO BE THE CAPTIVE'S SWEETHEART WHO WHEEDLES THE FLOWER THE RING AND THE PRISON KEY OUT OF THE STRICT VIRGINS FOR HIS OWN PURPOSES AND FLIES WITH HER AT LAST IN HIS SHALLOP ACROSS THE SEA TO LIVE WITH HER HAPPILY EVER AFTER
    174-50561-0002 BUT THIS IS A FALLACY
    174-50561-0003 THE WANDERING SINGER APPROACHES THEM WITH HIS LUTE
    174-50561-0004 THE EMPEROR'S DAUGHTER
    174-50561-0005 LADY LADY MY ROSE WHITE LADY BUT WILL YOU NOT HEAR A ROUNDEL LADY
    174-50561-0006 O IF YOU PLAY US A ROUNDEL SINGER HOW CAN THAT HARM THE EMPEROR'S DAUGHTER
    174-50561-0007 SHE WOULD NOT SPEAK THOUGH WE DANCED A WEEK WITH HER THOUGHTS A THOUSAND LEAGUES OVER THE WATER SINGER SINGER WANDERING SINGER O MY HONEY SWEET SINGER
    174-50561-0008 BUT IF I PLAY YOU A ROUNDEL LADY GET ME A GIFT FROM THE EMPEROR'S DAUGHTER HER FINGER RING FOR MY FINGER BRING THOUGH SHE'S PLEDGED A THOUSAND LEAGUES OVER THE WATER LADY LADY MY FAIR LADY O MY ROSE WHITE LADY
    174-50561-0009 THE WANDERING SINGER
    174-50561-0010 BUT I DID ONCE HAVE THE LUCK TO HEAR AND SEE THE LADY PLAYED IN ENTIRETY THE CHILDREN HAD BEEN GRANTED LEAVE TO PLAY JUST ONE MORE GAME BEFORE BED TIME AND OF COURSE THEY CHOSE THE LONGEST AND PLAYED IT WITHOUT MISSING A SYLLABLE
    174-50561-0011 THE LADIES IN YELLOW DRESSES STAND AGAIN IN A RING ABOUT THE EMPEROR'S DAUGHTER AND ARE FOR THE LAST TIME ACCOSTED BY THE SINGER WITH HIS LUTE
    174-50561-0012 THE WANDERING SINGER
    174-50561-0013 I'LL PLAY FOR YOU NOW NEATH THE APPLE BOUGH AND YOU SHALL DREAM ON THE LAWN SO SHADY LADY LADY MY FAIR LADY O MY APPLE GOLD LADY
    174-50561-0014 THE LADIES
    174-50561-0015 NOW YOU MAY PLAY A SERENA SINGER A DREAM OF NIGHT FOR AN APPLE GOLD LADY FOR THE FRUIT IS NOW ON THE APPLE BOUGH AND THE MOON IS UP AND THE LAWN IS SHADY SINGER SINGER WANDERING SINGER O MY HONEY SWEET SINGER
    174-50561-0016 ONCE MORE THE SINGER PLAYS AND THE LADIES DANCE BUT ONE BY ONE THEY FALL ASLEEP TO THE DROWSY MUSIC AND THEN THE SINGER STEPS INTO THE RING AND UNLOCKS THE TOWER AND KISSES THE EMPEROR'S DAUGHTER
    174-50561-0017 I DON'T KNOW WHAT BECOMES OF THE LADIES
    174-50561-0018 BED TIME CHILDREN
    174-50561-0019 YOU SEE THE TREATMENT IS A TRIFLE FANCIFUL
    174-84280-0000 HOW WE MUST SIMPLIFY
    174-84280-0001 IT SEEMS TO ME MORE AND MORE AS I LIVE LONGER THAT MOST POETRY AND MOST LITERATURE AND PARTICULARLY THE LITERATURE OF THE PAST IS DISCORDANT WITH THE VASTNESS AND VARIETY THE RESERVES AND RESOURCES AND RECUPERATIONS OF LIFE AS WE LIVE IT TO DAY
    174-84280-0002 IT IS THE EXPRESSION OF LIFE UNDER CRUDER AND MORE RIGID CONDITIONS THAN OURS LIVED BY PEOPLE WHO LOVED AND HATED MORE NAIVELY AGED SOONER AND DIED YOUNGER THAN WE DO
    174-84280-0003 WE RANGE WIDER LAST LONGER AND ESCAPE MORE AND MORE FROM INTENSITY TOWARDS UNDERSTANDING
    174-84280-0004 AND ALREADY THIS ASTOUNDING BLOW BEGINS TO TAKE ITS PLACE AMONG OTHER EVENTS AS A THING STRANGE AND TERRIBLE INDEED BUT RELATED TO ALL THE STRANGENESS AND MYSTERY OF LIFE PART OF THE UNIVERSAL MYSTERIES OF DESPAIR AND FUTILITY AND DEATH THAT HAVE TROUBLED MY CONSCIOUSNESS SINCE CHILDHOOD
    174-84280-0005 FOR A TIME THE DEATH OF MARY OBSCURED HER LIFE FOR ME BUT NOW HER LIVING PRESENCE IS MORE IN MY MIND AGAIN
    174-84280-0006 IT WAS THAT IDEA OF WASTE THAT DOMINATED MY MIND IN A STRANGE INTERVIEW I HAD WITH JUSTIN
    174-84280-0007 I BECAME GROTESQUELY ANXIOUS TO ASSURE HIM THAT INDEED SHE AND I HAD BEEN AS THEY SAY INNOCENT THROUGHOUT OUR LAST DAY TOGETHER
    174-84280-0008 YOU WERE WRONG IN ALL THAT I SAID SHE KEPT HER FAITH WITH YOU
    174-84280-0009 WE NEVER PLANNED TO MEET AND WHEN WE MET
    174-84280-0010 IF WE HAD BEEN BROTHER AND SISTER INDEED THERE WAS NOTHING
    174-84280-0011 BUT NOW IT DOESN'T SEEM TO MATTER VERY MUCH
    174-84280-0012 AND IT IS UPON THIS EFFECT OF SWEET AND BEAUTIFUL POSSIBILITIES CAUGHT IN THE NET OF ANIMAL JEALOUSIES AND THOUGHTLESS MOTIVES AND ANCIENT RIGID INSTITUTIONS THAT I WOULD END THIS WRITING
    174-84280-0013 IN MARY IT SEEMS TO ME I FOUND BOTH WOMANHOOD AND FELLOWSHIP I FOUND WHAT MANY HAVE DREAMT OF LOVE AND FRIENDSHIP FREELY GIVEN AND I COULD DO NOTHING BUT CLUTCH AT HER TO MAKE HER MY POSSESSION
    174-84280-0014 WHAT ALTERNATIVE WAS THERE FOR HER
    174-84280-0015 SHE WAS DESTROYED NOT MERELY BY THE UNCONSIDERED UNDISCIPLINED PASSIONS OF HER HUSBAND AND HER LOVER BUT BY THE VAST TRADITION THAT SUSTAINS AND ENFORCES THE SUBJUGATION OF HER SEX
    1919-142785-0000 ILLUSTRATION LONG PEPPER
    1919-142785-0001 LONG PEPPER THIS IS THE PRODUCE OF A DIFFERENT PLANT FROM THAT WHICH PRODUCES THE BLACK IT CONSISTING OF THE HALF RIPE FLOWER HEADS OF WHAT NATURALISTS CALL PIPER LONGUM AND CHABA
    1919-142785-0002 ORIGINALLY THE MOST VALUABLE OF THESE WERE FOUND IN THE SPICE ISLANDS OR MOLUCCAS OF THE INDIAN OCEAN AND WERE HIGHLY PRIZED BY THE NATIONS OF ANTIQUITY
    1919-142785-0003 THE LONG PEPPER IS LESS AROMATIC THAN THE BLACK BUT ITS OIL IS MORE PUNGENT
    1919-142785-0004 THEN ADD THE YOLKS OF THE EGGS WELL BEATEN STIR THEM TO THE SAUCE BUT DO NOT ALLOW IT TO BOIL AND SERVE VERY HOT
    1919-142785-0005 MODE PARE AND SLICE THE CUCUMBERS AS FOR THE TABLE SPRINKLE WELL WITH SALT AND LET THEM REMAIN FOR TWENTY FOUR HOURS STRAIN OFF THE LIQUOR PACK IN JARS A THICK LAYER OF CUCUMBERS AND SALT ALTERNATELY TIE DOWN CLOSELY AND WHEN WANTED FOR USE TAKE OUT THE QUANTITY REQUIRED
    1919-142785-0006 ILLUSTRATION THE CUCUMBER
    1919-142785-0007 MODE CHOOSE THE GREENEST CUCUMBERS AND THOSE THAT ARE MOST FREE FROM SEEDS PUT THEM IN STRONG SALT AND WATER WITH A CABBAGE LEAF TO KEEP THEM DOWN TIE A PAPER OVER THEM AND PUT THEM IN A WARM PLACE TILL THEY ARE YELLOW THEN WASH THEM AND SET THEM OVER THE FIRE IN FRESH WATER WITH A VERY LITTLE SALT AND ANOTHER CABBAGE LEAF OVER THEM COVER VERY CLOSELY BUT TAKE CARE THEY DO NOT BOIL
    1919-142785-0008 PUT THE SUGAR WITH ONE QUARTER PINT OF WATER IN A SAUCEPAN OVER THE FIRE REMOVE THE SCUM AS IT RISES AND ADD THE LEMON PEEL AND GINGER WITH THE OUTSIDE SCRAPED OFF WHEN THE SYRUP IS TOLERABLY THICK TAKE IT OFF THE FIRE AND WHEN COLD WIPE THE CUCUMBERS DRY AND PUT THEM IN
    1919-142785-0009 SEASONABLE THIS RECIPE SHOULD BE USED IN JUNE JULY OR AUGUST
    1919-142785-0010 SOLID ROCKS OF SALT ARE ALSO FOUND IN VARIOUS PARTS OF THE WORLD AND THE COUNTY OF CHESTER CONTAINS MANY OF THESE MINES AND IT IS FROM THERE THAT MUCH OF OUR SALT COMES
    1919-142785-0011 SOME SPRINGS ARE SO HIGHLY IMPREGNATED WITH SALT AS TO HAVE RECEIVED THE NAME OF BRINE SPRINGS AND ARE SUPPOSED TO HAVE BECOME SO BY PASSING THROUGH THE SALT ROCKS BELOW GROUND AND THUS DISSOLVING A PORTION OF THIS MINERAL SUBSTANCE
    1919-142785-0012 MODE PUT THE MILK IN A VERY CLEAN SAUCEPAN AND LET IT BOIL
    1919-142785-0013 BEAT THE EGGS STIR TO THEM THE MILK AND POUNDED SUGAR AND PUT THE MIXTURE INTO A JUG
    1919-142785-0014 PLACE THE JUG IN A SAUCEPAN OF BOILING WATER KEEP STIRRING WELL UNTIL IT THICKENS BUT DO NOT ALLOW IT TO BOIL OR IT WILL CURDLE
    1919-142785-0015 WHEN IT IS SUFFICIENTLY THICK TAKE IT OFF AS IT SHOULD NOT BOIL
    1919-142785-0016 ILLUSTRATION THE LEMON
    1919-142785-0017 THE LEMON THIS FRUIT IS A NATIVE OF ASIA AND IS MENTIONED BY VIRGIL AS AN ANTIDOTE TO POISON
    1919-142785-0018 IT IS HARDIER THAN THE ORANGE AND AS ONE OF THE CITRON TRIBE WAS BROUGHT INTO EUROPE BY THE ARABIANS
    1919-142785-0019 THE LEMON WAS FIRST CULTIVATED IN ENGLAND IN THE BEGINNING OF THE SEVENTEENTH CENTURY AND IS NOW OFTEN TO BE FOUND IN OUR GREEN HOUSES
    1919-142785-0020 THIS JUICE WHICH IS CALLED CITRIC ACID MAY BE PRESERVED IN BOTTLES FOR A CONSIDERABLE TIME BY COVERING IT WITH A THIN STRATUM OF OIL
    1919-142785-0021 TO PICKLE EGGS
    1919-142785-0022 SEASONABLE THIS SHOULD BE MADE ABOUT EASTER AS AT THIS TIME EGGS ARE PLENTIFUL AND CHEAP
    1919-142785-0023 A STORE OF PICKLED EGGS WILL BE FOUND VERY USEFUL AND ORNAMENTAL IN SERVING WITH MANY FIRST AND SECOND COURSE DISHES
    1919-142785-0024 ILLUSTRATION GINGER
    1919-142785-0025 THE GINGER PLANT KNOWN TO NATURALISTS AS ZINGIBER OFFICINALE IS A NATIVE OF THE EAST AND WEST INDIES
    1919-142785-0026 IN JAMAICA IT FLOWERS ABOUT AUGUST OR SEPTEMBER FADING ABOUT THE END OF THE YEAR
    1919-142785-0027 BEAT THE YOLKS OF THE OTHER TWO EGGS ADD THEM WITH A LITTLE FLOUR AND SALT TO THOSE POUNDED MIX ALL WELL TOGETHER AND ROLL INTO BALLS
    1919-142785-0028 BOIL THEM BEFORE THEY ARE PUT INTO THE SOUP OR OTHER DISH THEY MAY BE INTENDED FOR
    1919-142785-0029 LEMON JUICE MAY BE ADDED AT PLEASURE
    1919-142785-0030 MODE PUT THE WHOLE OF THE INGREDIENTS INTO A BOTTLE AND LET IT REMAIN FOR A FORTNIGHT IN A WARM PLACE OCCASIONALLY SHAKING UP THE CONTENTS
    1919-142785-0031 THEY OUGHT TO BE TAKEN UP IN THE AUTUMN AND WHEN DRIED IN THE HOUSE WILL KEEP TILL SPRING
    1919-142785-0032 ADD THE WINE AND IF NECESSARY A SEASONING OF CAYENNE WHEN IT WILL BE READY TO SERVE
    1919-142785-0033 NOTE THE WINE IN THIS SAUCE MAY BE OMITTED AND AN ONION SLICED AND FRIED OF A NICE BROWN SUBSTITUTED FOR IT
    1919-142785-0034 SIMMER FOR A MINUTE OR TWO AND SERVE IN A TUREEN
    1919-142785-0035 SUFFICIENT TO SERVE WITH FIVE OR SIX MACKEREL
    1919-142785-0036 VARIOUS DISHES ARE FREQUENTLY ORNAMENTED AND GARNISHED WITH ITS GRACEFUL LEAVES AND THESE ARE SOMETIMES BOILED IN SOUPS ALTHOUGH IT IS MORE USUALLY CONFINED IN ENGLISH COOKERY TO THE MACKEREL SAUCE AS HERE GIVEN
    1919-142785-0037 FORCEMEAT FOR COLD SAVOURY PIES
    1919-142785-0038 POUND WELL AND BIND WITH ONE OR TWO EGGS WHICH HAVE BEEN PREVIOUSLY BEATEN AND STRAINED
    1919-142785-0039 ILLUSTRATION MARJORAM
    1919-142785-0040 IT IS A NATIVE OF PORTUGAL AND WHEN ITS LEAVES ARE USED AS A SEASONING HERB THEY HAVE AN AGREEABLE AROMATIC FLAVOUR
    1919-142785-0041 MODE MIX ALL THE INGREDIENTS WELL TOGETHER CAREFULLY MINCING THEM VERY FINELY BEAT UP THE EGG MOISTEN WITH IT AND WORK THE WHOLE VERY SMOOTHLY TOGETHER
    1919-142785-0042 SUFFICIENT FOR A MODERATE SIZED HADDOCK OR PIKE
    1919-142785-0043 NOW BEAT AND STRAIN THE EGGS WORK THESE UP WITH THE OTHER INGREDIENTS AND THE FORCEMEAT WILL BE READY FOR USE
    1919-142785-0044 BOIL FOR FIVE MINUTES MINCE IT VERY SMALL AND MIX IT WITH THE OTHER INGREDIENTS
    1919-142785-0045 IF IT SHOULD BE IN AN UNSOUND STATE IT MUST BE ON NO ACCOUNT MADE USE OF
    1919-142785-0046 ILLUSTRATION BASIL
    1919-142785-0047 OTHER SWEET HERBS ARE CULTIVATED FOR PURPOSES OF MEDICINE AND PERFUMERY THEY ARE MOST GRATEFUL BOTH TO THE ORGANS OF TASTE AND SMELLING AND TO THE AROMA DERIVED FROM THEM IS DUE IN A GREAT MEASURE THE SWEET AND EXHILARATING FRAGRANCE OF OUR FLOWERY MEADS
    1919-142785-0048 FRENCH FORCEMEAT
    1919-142785-0049 IT WILL BE WELL TO STATE IN THE BEGINNING OF THIS RECIPE THAT FRENCH FORCEMEAT OR QUENELLES CONSIST OF THE BLENDING OF THREE SEPARATE PROCESSES NAMELY PANADA UDDER AND WHATEVER MEAT YOU INTEND USING PANADA
    1919-142785-0050 PLACE IT OVER THE FIRE KEEP CONSTANTLY STIRRING TO PREVENT ITS BURNING AND WHEN QUITE DRY PUT IN A SMALL PIECE OF BUTTER
    1919-142785-0051 PUT THE UDDER INTO A STEWPAN WITH SUFFICIENT WATER TO COVER IT LET IT STEW GENTLY TILL QUITE DONE WHEN TAKE IT OUT TO COOL
    1919-142785-0052 ILLUSTRATION PESTLE AND MORTAR
    1919-142785-0053 WHEN THE THREE INGREDIENTS ARE PROPERLY PREPARED POUND THEM ALTOGETHER IN A MORTAR FOR SOME TIME FOR THE MORE QUENELLES ARE POUNDED THE MORE DELICATE THEY ARE
    1919-142785-0054 IF THE QUENELLES ARE NOT FIRM ENOUGH ADD THE YOLK OF ANOTHER EGG BUT OMIT THE WHITE WHICH ONLY MAKES THEM HOLLOW AND PUFFY INSIDE
    1919-142785-0055 ANY ONE WITH THE SLIGHTEST PRETENSIONS TO REFINED COOKERY MUST IN THIS PARTICULAR IMPLICITLY FOLLOW THE EXAMPLE OF OUR FRIENDS ACROSS THE CHANNEL
    1919-142785-0056 FRIED BREAD CRUMBS
    1919-142785-0057 THE FAT THEY ARE FRIED IN SHOULD BE CLEAR AND THE CRUMBS SHOULD NOT HAVE THE SLIGHTEST APPEARANCE OR TASTE OF HAVING BEEN IN THE LEAST DEGREE BURNT
    1919-142785-0058 FRIED BREAD FOR BORDERS
    1919-142785-0059 WHEN QUITE CRISP DIP ONE SIDE OF THE SIPPET INTO THE BEATEN WHITE OF AN EGG MIXED WITH A LITTLE FLOUR AND PLACE IT ON THE EDGE OF THE DISH
    1919-142785-0060 CONTINUE IN THIS MANNER TILL THE BORDER IS COMPLETED ARRANGING THE SIPPETS A PALE AND A DARK ONE ALTERNATELY
    1919-142785-0061 MODE CUT UP THE ONION AND CARROT INTO SMALL RINGS AND PUT THEM INTO A STEWPAN WITH THE HERBS MUSHROOMS BAY LEAF CLOVES AND MACE ADD THE BUTTER AND SIMMER THE WHOLE VERY GENTLY OVER A SLOW FIRE UNTIL THE ONION IS QUITE TENDER
    1919-142785-0062 SUFFICIENT HALF THIS QUANTITY FOR TWO SLICES OF SALMON
    1919-142785-0063 ILLUSTRATION SAGE
    1988-147956-0000 FUCHS BROUGHT UP A SACK OF POTATOES AND A PIECE OF CURED PORK FROM THE CELLAR AND GRANDMOTHER PACKED SOME LOAVES OF SATURDAY'S BREAD A JAR OF BUTTER AND SEVERAL PUMPKIN PIES IN THE STRAW OF THE WAGON BOX
    1988-147956-0001 OCCASIONALLY ONE OF THE HORSES WOULD TEAR OFF WITH HIS TEETH A PLANT FULL OF BLOSSOMS AND WALK ALONG MUNCHING IT THE FLOWERS NODDING IN TIME TO HIS BITES AS HE ATE DOWN TOWARD THEM
    1988-147956-0002 IT'S NO BETTER THAN A BADGER HOLE NO PROPER DUGOUT AT ALL
    1988-147956-0003 NOW WHY IS THAT OTTO
    1988-147956-0004 PRESENTLY AGAINST ONE OF THOSE BANKS I SAW A SORT OF SHED THATCHED WITH THE SAME WINE COLORED GRASS THAT GREW EVERYWHERE
    1988-147956-0005 VERY GLAD VERY GLAD SHE EJACULATED
    1988-147956-0006 YOU'LL GET FIXED UP COMFORTABLE AFTER WHILE MISSUS SHIMERDA MAKE GOOD HOUSE
    1988-147956-0007 MY GRANDMOTHER ALWAYS SPOKE IN A VERY LOUD TONE TO FOREIGNERS AS IF THEY WERE DEAF
    1988-147956-0008 SHE MADE MISSUS SHIMERDA UNDERSTAND THE FRIENDLY INTENTION OF OUR VISIT AND THE BOHEMIAN WOMAN HANDLED THE LOAVES OF BREAD AND EVEN SMELLED THEM AND EXAMINED THE PIES WITH LIVELY CURIOSITY EXCLAIMING MUCH GOOD MUCH THANK
    1988-147956-0009 THE FAMILY HAD BEEN LIVING ON CORNCAKES AND SORGHUM MOLASSES FOR THREE DAYS
    1988-147956-0010 I REMEMBERED WHAT THE CONDUCTOR HAD SAID ABOUT HER EYES
    1988-147956-0011 HER SKIN WAS BROWN TOO AND IN HER CHEEKS SHE HAD A GLOW OF RICH DARK COLOR
    1988-147956-0012 EVEN FROM A DISTANCE ONE COULD SEE THAT THERE WAS SOMETHING STRANGE ABOUT THIS BOY
    1988-147956-0013 HE WAS BORN LIKE THAT THE OTHERS ARE SMART
    1988-147956-0014 AMBROSCH HE MAKE GOOD FARMER
    1988-147956-0015 HE STRUCK AMBROSCH ON THE BACK AND THE BOY SMILED KNOWINGLY
    1988-147956-0016 AT THAT MOMENT THE FATHER CAME OUT OF THE HOLE IN THE BANK
    1988-147956-0017 IT WAS SO LONG THAT IT BUSHED OUT BEHIND HIS EARS AND MADE HIM LOOK LIKE THE OLD PORTRAITS I REMEMBERED IN VIRGINIA
    1988-147956-0018 I NOTICED HOW WHITE AND WELL SHAPED HIS OWN HANDS WERE
    1988-147956-0019 WE STOOD PANTING ON THE EDGE OF THE RAVINE LOOKING DOWN AT THE TREES AND BUSHES THAT GREW BELOW US
    1988-147956-0020 THE WIND WAS SO STRONG THAT I HAD TO HOLD MY HAT ON AND THE GIRLS SKIRTS WERE BLOWN OUT BEFORE THEM
    1988-147956-0021 SHE LOOKED AT ME HER EYES FAIRLY BLAZING WITH THINGS SHE COULD NOT SAY
    1988-147956-0022 SHE POINTED INTO THE GOLD COTTONWOOD TREE BEHIND WHOSE TOP WE STOOD AND SAID AGAIN WHAT NAME
    1988-147956-0023 ANTONIA POINTED UP TO THE SKY AND QUESTIONED ME WITH HER GLANCE
    1988-147956-0024 SHE GOT UP ON HER KNEES AND WRUNG HER HANDS
    1988-147956-0025 SHE WAS QUICK AND VERY EAGER
    1988-147956-0026 WE WERE SO DEEP IN THE GRASS THAT WE COULD SEE NOTHING BUT THE BLUE SKY OVER US AND THE GOLD TREE IN FRONT OF US
    1988-147956-0027 AFTER ANTONIA HAD SAID THE NEW WORDS OVER AND OVER SHE WANTED TO GIVE ME A LITTLE CHASED SILVER RING SHE WORE ON HER MIDDLE FINGER
    1988-147956-0028 WHEN I CAME UP HE TOUCHED MY SHOULDER AND LOOKED SEARCHINGLY DOWN INTO MY FACE FOR SEVERAL SECONDS
    1988-147956-0029 I BECAME SOMEWHAT EMBARRASSED FOR I WAS USED TO BEING TAKEN FOR GRANTED BY MY ELDERS
    1988-148538-0000 IN ARISTOCRATIC COMMUNITIES THE PEOPLE READILY GIVE THEMSELVES UP TO BURSTS OF TUMULTUOUS AND BOISTEROUS GAYETY WHICH SHAKE OFF AT ONCE THE RECOLLECTION OF THEIR PRIVATIONS THE NATIVES OF DEMOCRACIES ARE NOT FOND OF BEING THUS VIOLENTLY BROKEN IN UPON AND THEY NEVER LOSE SIGHT OF THEIR OWN SELVES WITHOUT REGRET
    1988-148538-0001 AN AMERICAN INSTEAD OF GOING IN A LEISURE HOUR TO DANCE MERRILY AT SOME PLACE OF PUBLIC RESORT AS THE FELLOWS OF HIS CALLING CONTINUE TO DO THROUGHOUT THE GREATER PART OF EUROPE SHUTS HIMSELF UP AT HOME TO DRINK
    1988-148538-0002 I BELIEVE THE SERIOUSNESS OF THE AMERICANS ARISES PARTLY FROM THEIR PRIDE
    1988-148538-0003 THIS IS MORE ESPECIALLY THE CASE AMONGST THOSE FREE NATIONS WHICH FORM DEMOCRATIC COMMUNITIES
    1988-148538-0004 THEN THERE ARE IN ALL CLASSES A VERY LARGE NUMBER OF MEN CONSTANTLY OCCUPIED WITH THE SERIOUS AFFAIRS OF THE GOVERNMENT AND THOSE WHOSE THOUGHTS ARE NOT ENGAGED IN THE DIRECTION OF THE COMMONWEALTH ARE WHOLLY ENGROSSED BY THE ACQUISITION OF A PRIVATE FORTUNE
    1988-148538-0005 I DO NOT BELIEVE IN SUCH REPUBLICS ANY MORE THAN IN THAT OF PLATO OR IF THE THINGS WE READ OF REALLY HAPPENED I DO NOT HESITATE TO AFFIRM THAT THESE SUPPOSED DEMOCRACIES WERE COMPOSED OF VERY DIFFERENT ELEMENTS FROM OURS AND THAT THEY HAD NOTHING IN COMMON WITH THE LATTER EXCEPT THEIR NAME
    1988-148538-0006 IN ARISTOCRACIES EVERY MAN HAS ONE SOLE OBJECT WHICH HE UNCEASINGLY PURSUES BUT AMONGST DEMOCRATIC NATIONS THE EXISTENCE OF MAN IS MORE COMPLEX THE SAME MIND WILL ALMOST ALWAYS EMBRACE SEVERAL OBJECTS AT THE SAME TIME AND THESE OBJECTS ARE FREQUENTLY WHOLLY FOREIGN TO EACH OTHER AS IT CANNOT KNOW THEM ALL WELL THE MIND IS READILY SATISFIED WITH IMPERFECT NOTIONS OF EACH
    1988-148538-0007 CHAPTER SIXTEEN WHY THE NATIONAL VANITY OF THE AMERICANS IS MORE RESTLESS AND CAPTIOUS THAN THAT OF THE ENGLISH
    1988-148538-0008 THE AMERICANS IN THEIR INTERCOURSE WITH STRANGERS APPEAR IMPATIENT OF THE SMALLEST CENSURE AND INSATIABLE OF PRAISE
    1988-148538-0009 IF I SAY TO AN AMERICAN THAT THE COUNTRY HE LIVES IN IS A FINE ONE AY HE REPLIES THERE IS NOT ITS FELLOW IN THE WORLD
    1988-148538-0010 IF I APPLAUD THE FREEDOM WHICH ITS INHABITANTS ENJOY HE ANSWERS FREEDOM IS A FINE THING BUT FEW NATIONS ARE WORTHY TO ENJOY IT
    1988-148538-0011 IN ARISTOCRATIC COUNTRIES THE GREAT POSSESS IMMENSE PRIVILEGES UPON WHICH THEIR PRIDE RESTS WITHOUT SEEKING TO RELY UPON THE LESSER ADVANTAGES WHICH ACCRUE TO THEM
    1988-148538-0012 THEY THEREFORE ENTERTAIN A CALM SENSE OF THEIR SUPERIORITY THEY DO NOT DREAM OF VAUNTING PRIVILEGES WHICH EVERYONE PERCEIVES AND NO ONE CONTESTS AND THESE THINGS ARE NOT SUFFICIENTLY NEW TO THEM TO BE MADE TOPICS OF CONVERSATION
    1988-148538-0013 THEY STAND UNMOVED IN THEIR SOLITARY GREATNESS WELL ASSURED THAT THEY ARE SEEN OF ALL THE WORLD WITHOUT ANY EFFORT TO SHOW THEMSELVES OFF AND THAT NO ONE WILL ATTEMPT TO DRIVE THEM FROM THAT POSITION
    1988-148538-0014 WHEN AN ARISTOCRACY CARRIES ON THE PUBLIC AFFAIRS ITS NATIONAL PRIDE NATURALLY ASSUMES THIS RESERVED INDIFFERENT AND HAUGHTY FORM WHICH IS IMITATED BY ALL THE OTHER CLASSES OF THE NATION
    1988-148538-0015 THESE PERSONS THEN DISPLAYED TOWARDS EACH OTHER PRECISELY THE SAME PUERILE JEALOUSIES WHICH ANIMATE THE MEN OF DEMOCRACIES THE SAME EAGERNESS TO SNATCH THE SMALLEST ADVANTAGES WHICH THEIR EQUALS CONTESTED AND THE SAME DESIRE TO PARADE OSTENTATIOUSLY THOSE OF WHICH THEY WERE IN POSSESSION
    1988-24833-0000 THE TWO STRAY KITTENS GRADUALLY MAKE THEMSELVES AT HOME
    1988-24833-0001 SOMEHOW OR OTHER CAT HAS TAUGHT THEM THAT HE'S IN CHARGE HERE AND HE JUST CHASES THEM FOR FUN NOW AND AGAIN WHEN HE'S NOT BUSY SLEEPING
    1988-24833-0002 SHE DOESN'T PICK THEM UP BUT JUST HAVING THEM IN THE ROOM SURE DOESN'T GIVE HER ASTHMA
    1988-24833-0003 WHEN ARE YOU GETTING RID OF THESE CATS I'M NOT FIXING TO START AN ANNEX TO KATE'S CAT HOME
    1988-24833-0004 RIGHT AWAY WHEN I BRING HOME MY NEW PROGRAM HE SAYS HOW COME YOU'RE TAKING ONE LESS COURSE THIS HALF
    1988-24833-0005 I EXPLAIN THAT I'M TAKING MUSIC AND ALSO BIOLOGY ALGEBRA ENGLISH AND FRENCH MUSIC HE SNORTS
    1988-24833-0006 POP IT'S A COURSE
    1988-24833-0007 HE DOES AND FOR ONCE I WIN A ROUND I KEEP MUSIC FOR THIS SEMESTER
    1988-24833-0008 I'LL BE LUCKY IF I HAVE TIME TO BREATHE
    1988-24833-0009 SOMETIMES SCHOOLS DO LET KIDS TAKE A LOT OF SOFT COURSES AND THEN THEY'RE OUT ON A LIMB LATER HUH
    1988-24833-0010 SO HE CARES HUH
    1988-24833-0011 BESIDES SAYS TOM HALF THE REASON YOU AND YOUR FATHER ARE ALWAYS BICKERING IS THAT YOU'RE SO MUCH ALIKE ME LIKE HIM SURE
    1988-24833-0012 AS LONG AS THERE'S A BONE ON THE FLOOR THE TWO OF YOU WORRY IT
    1988-24833-0013 I GET THE PILLOWS COMFORTABLY ARRANGED ON THE FLOOR WITH A BIG BOTTLE OF SODA AND A BAG OF POPCORN WITHIN EASY REACH
    1988-24833-0014 POP GOES RIGHT ON TUNING HIS CHANNEL
    1988-24833-0015 YOU'RE GETTING ALTOGETHER TOO UPSET ABOUT THESE PROGRAMS STOP IT AND BEHAVE YOURSELF
    1988-24833-0016 IT'S YOUR FAULT MOP IT UP YOURSELF
    1988-24833-0017 I HEAR THE T V GOING FOR A FEW MINUTES THEN POP TURNS IT OFF AND GOES IN THE KITCHEN TO TALK TO MOM
    1988-24833-0018 WELL I DON'T THINK YOU SHOULD TURN A GUY'S T V PROGRAM OFF IN THE MIDDLE WITHOUT EVEN FINDING OUT ABOUT IT
    1988-24833-0019 I LOOK AT MY WATCH IT'S A QUARTER TO ELEVEN
    1988-24833-0020 I TURN OFF THE TELEVISION SET I'VE LOST TRACK OF WHAT'S HAPPENING AND IT DOESN'T SEEM TO BE THE GRANDFATHER WHO'S THE SPOOK AFTER ALL
    1988-24833-0021 IT'S THE FIRST TIME HILDA HAS BEEN TO OUR HOUSE AND TOM INTRODUCES HER AROUND
    1988-24833-0022 I TOLD TOM WE SHOULDN'T COME SO LATE SAYS HILDA
    1988-24833-0023 TOM SAYS THANKS AND LOOKS AT HILDA AND SHE BLUSHES REALLY
    1988-24833-0024 TOM DRINKS A LITTLE MORE COFFEE AND THEN HE GOES ON THE TROUBLE IS I CAN'T GET MARRIED ON THIS FLOWER SHOP JOB
    1988-24833-0025 YOU KNOW I'D GET DRAFTED IN A YEAR OR TWO ANYWAY
    1988-24833-0026 I'VE DECIDED TO ENLIST IN THE ARMY
    1988-24833-0027 I'LL HAVE TO CHECK SOME MORE SAYS TOM
    1988-24833-0028 HERE'S TO YOU A LONG HAPPY LIFE
    
    
    Ground Truth sentences 0 to 10:
    1272-128104-0000 MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
    1272-128104-0001 NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER
    1272-128104-0002 HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
    1272-128104-0003 HE HAS GRAVE DOUBTS WHETHER SIR FREDERICK LEIGHTON'S WORK IS REALLY GREEK AFTER ALL AND CAN DISCOVER IN IT BUT LITTLE OF ROCKY ITHACA
    1272-128104-0004 LINNELL'S PICTURES ARE A SORT OF UP GUARDS AND AT EM PAINTINGS AND MASON'S EXQUISITE IDYLLS ARE AS NATIONAL AS A JINGO POEM MISTER BIRKET FOSTER'S LANDSCAPES SMILE AT ONE MUCH IN THE SAME WAY THAT MISTER CARKER USED TO FLASH HIS TEETH AND MISTER JOHN COLLIER GIVES HIS SITTER A CHEERFUL SLAP ON THE BACK BEFORE HE SAYS LIKE A SHAMPOOER IN A TURKISH BATH NEXT MAN
    1272-128104-0005 IT IS OBVIOUSLY UNNECESSARY FOR US TO POINT OUT HOW LUMINOUS THESE CRITICISMS ARE HOW DELICATE IN EXPRESSION
    1272-128104-0006 ON THE GENERAL PRINCIPLES OF ART MISTER QUILTER WRITES WITH EQUAL LUCIDITY
    1272-128104-0007 PAINTING HE TELLS US IS OF A DIFFERENT QUALITY TO MATHEMATICS AND FINISH IN ART IS ADDING MORE FACT
    1272-128104-0008 AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN
    1272-128104-0009 HE LAMENTS MOST BITTERLY THE DIVORCE THAT HAS BEEN MADE BETWEEN DECORATIVE ART AND WHAT WE USUALLY CALL PICTURES MAKES THE CUSTOMARY APPEAL TO THE LAST JUDGMENT AND REMINDS US THAT IN THE GREAT DAYS OF ART MICHAEL ANGELO WAS THE FURNISHING UPHOLSTERER
    1272-128104-0010 NEAR THE FIRE AND THE ORNAMENTS FRED BROUGHT HOME FROM INDIA ON THE MANTEL BOARD
    1272-128104-0011 IN FACT HE IS QUITE SEVERE ON MISTER RUSKIN FOR NOT RECOGNISING THAT A PICTURE SHOULD DENOTE THE FRAILTY OF MAN AND REMARKS WITH PLEASING COURTESY AND FELICITOUS GRACE THAT MANY PHASES OF FEELING
    1272-128104-0012 ONLY UNFORTUNATELY HIS OWN WORK NEVER DOES GET GOOD
    1272-128104-0013 MISTER QUILTER HAS MISSED HIS CHANCE FOR HE HAS FAILED EVEN TO MAKE HIMSELF THE TUPPER OF PAINTING
    1272-128104-0014 BY HARRY QUILTER M A
    1272-135031-0000 BECAUSE YOU WERE SLEEPING INSTEAD OF CONQUERING THE LOVELY ROSE PRINCESS HAS BECOME A FIDDLE WITHOUT A BOW WHILE POOR SHAGGY SITS THERE A COOING DOVE
    1272-135031-0001 HE HAS GONE AND GONE FOR GOOD ANSWERED POLYCHROME WHO HAD MANAGED TO SQUEEZE INTO THE ROOM BESIDE THE DRAGON AND HAD WITNESSED THE OCCURRENCES WITH MUCH INTEREST
    1272-135031-0002 I HAVE REMAINED A PRISONER ONLY BECAUSE I WISHED TO BE ONE AND WITH THIS HE STEPPED FORWARD AND BURST THE STOUT CHAINS AS EASILY AS IF THEY HAD BEEN THREADS
    1272-135031-0003 THE LITTLE GIRL HAD BEEN ASLEEP BUT SHE HEARD THE RAPS AND OPENED THE DOOR
    1272-135031-0004 THE KING HAS FLED IN DISGRACE AND YOUR FRIENDS ARE ASKING FOR YOU
    1272-135031-0005 I BEGGED RUGGEDO LONG AGO TO SEND HIM AWAY BUT HE WOULD NOT DO SO
    1272-135031-0006 I ALSO OFFERED TO HELP YOUR BROTHER TO ESCAPE BUT HE WOULD NOT GO
    1272-135031-0007 HE EATS AND SLEEPS VERY STEADILY REPLIED THE NEW KING
    1272-135031-0008 I HOPE HE DOESN'T WORK TOO HARD SAID SHAGGY
    1272-135031-0009 HE DOESN'T WORK AT ALL
    1272-135031-0010 IN FACT THERE IS NOTHING HE CAN DO IN THESE DOMINIONS AS WELL AS OUR NOMES WHOSE NUMBERS ARE SO GREAT THAT IT WORRIES US TO KEEP THEM ALL BUSY
    1272-135031-0011 NOT EXACTLY RETURNED KALIKO
    1272-135031-0012 WHERE IS MY BROTHER NOW
    1272-135031-0013 INQUIRED SHAGGY IN THE METAL FOREST
    1272-135031-0014 WHERE IS THAT
    1272-135031-0015 THE METAL FOREST IS IN THE GREAT DOMED CAVERN THE LARGEST IN ALL OUR DOMINIONS REPLIED KALIKO
    1272-135031-0016 KALIKO HESITATED
    1272-135031-0017 HOWEVER IF WE LOOK SHARP WE MAY BE ABLE TO DISCOVER ONE OF THESE SECRET WAYS
    1272-135031-0018 OH NO I'M QUITE SURE HE DIDN'T
    1272-135031-0019 THAT'S FUNNY REMARKED BETSY THOUGHTFULLY
    1272-135031-0020 I DON'T BELIEVE ANN KNEW ANY MAGIC OR SHE'D HAVE WORKED IT BEFORE
    1272-135031-0021 I DO NOT KNOW CONFESSED SHAGGY
    1272-135031-0022 TRUE AGREED KALIKO
    1272-135031-0023 KALIKO WENT TO THE BIG GONG AND POUNDED ON IT JUST AS RUGGEDO USED TO DO BUT NO ONE ANSWERED THE SUMMONS
    1272-135031-0024 HAVING RETURNED TO THE ROYAL CAVERN KALIKO FIRST POUNDED THE GONG AND THEN SAT IN THE THRONE WEARING RUGGEDO'S DISCARDED RUBY CROWN AND HOLDING IN HIS HAND THE SCEPTRE WHICH RUGGEDO HAD SO OFTEN THROWN AT HIS HEAD
    1272-141231-0000 A MAN SAID TO THE UNIVERSE SIR I EXIST
    1272-141231-0001 SWEAT COVERED BRION'S BODY TRICKLING INTO THE TIGHT LOINCLOTH THAT WAS THE ONLY GARMENT HE WORE
    1272-141231-0002 THE CUT ON HIS CHEST STILL DRIPPING BLOOD THE ACHE OF HIS OVERSTRAINED EYES EVEN THE SOARING ARENA AROUND HIM WITH THE THOUSANDS OF SPECTATORS WERE TRIVIALITIES NOT WORTH THINKING ABOUT
    1272-141231-0003 HIS INSTANT OF PANIC WAS FOLLOWED BY A SMALL SHARP BLOW HIGH ON HIS CHEST
    1272-141231-0004 ONE MINUTE A VOICE SAID AND THE TIME BUZZER SOUNDED
    1272-141231-0005 A MINUTE IS NOT A VERY LARGE MEASURE OF TIME AND HIS BODY NEEDED EVERY FRACTION OF IT
    1272-141231-0006 THE BUZZER'S WHIRR TRIGGERED HIS MUSCLES INTO COMPLETE RELAXATION
    1272-141231-0007 ONLY HIS HEART AND LUNGS WORKED ON AT A STRONG MEASURED RATE
    1272-141231-0008 HE WAS IN REVERIE SLIDING ALONG THE BORDERS OF CONSCIOUSNESS
    1272-141231-0009 THE CONTESTANTS IN THE TWENTIES NEEDED UNDISTURBED REST THEREFORE NIGHTS IN THE DORMITORIES WERE AS QUIET AS DEATH
    1272-141231-0010 PARTICULARLY SO ON THIS LAST NIGHT WHEN ONLY TWO OF THE LITTLE CUBICLES WERE OCCUPIED THE THOUSANDS OF OTHERS STANDING WITH DARK EMPTY DOORS
    1272-141231-0011 THE OTHER VOICE SNAPPED WITH A HARSH URGENCY CLEARLY USED TO COMMAND
    1272-141231-0012 I'M HERE BECAUSE THE MATTER IS OF UTMOST IMPORTANCE AND BRANDD IS THE ONE I MUST SEE NOW STAND ASIDE
    1272-141231-0013 THE TWENTIES
    1272-141231-0014 HE MUST HAVE DRAWN HIS GUN BECAUSE THE INTRUDER SAID QUICKLY PUT THAT AWAY YOU'RE BEING A FOOL OUT
    1272-141231-0015 THERE WAS SILENCE THEN AND STILL WONDERING BRION WAS ONCE MORE ASLEEP
    1272-141231-0016 TEN SECONDS
    1272-141231-0017 HE ASKED THE HANDLER WHO WAS KNEADING HIS ACHING MUSCLES
    1272-141231-0018 A RED HAIRED MOUNTAIN OF A MAN WITH AN APPARENTLY INEXHAUSTIBLE STORE OF ENERGY
    1272-141231-0019 THERE COULD BE LITTLE ART IN THIS LAST AND FINAL ROUND OF FENCING
    1272-141231-0020 JUST THRUST AND PARRY AND VICTORY TO THE STRONGER
    1272-141231-0021 EVERY MAN WHO ENTERED THE TWENTIES HAD HIS OWN TRAINING TRICKS
    1272-141231-0022 THERE APPEARED TO BE AN IMMEDIATE ASSOCIATION WITH THE DEATH TRAUMA AS IF THE TWO WERE INEXTRICABLY LINKED INTO ONE
    1272-141231-0023 THE STRENGTH THAT ENABLES SOMEONE IN A TRANCE TO HOLD HIS BODY STIFF AND UNSUPPORTED EXCEPT AT TWO POINTS THE HEAD AND HEELS
    1272-141231-0024 THIS IS PHYSICALLY IMPOSSIBLE WHEN CONSCIOUS
    1272-141231-0025 OTHERS HAD DIED BEFORE DURING THE TWENTIES AND DEATH DURING THE LAST ROUND WAS IN SOME WAYS EASIER THAN DEFEAT
    1272-141231-0026 BREATHING DEEPLY BRION SOFTLY SPOKE THE AUTO HYPNOTIC PHRASES THAT TRIGGERED THE PROCESS
    1272-141231-0027 WHEN THE BUZZER SOUNDED HE PULLED HIS FOIL FROM HIS SECOND'S STARTLED GRASP AND RAN FORWARD
    1272-141231-0028 IROLG LOOKED AMAZED AT THE SUDDEN FURY OF THE ATTACK THEN SMILED
    1272-141231-0029 HE THOUGHT IT WAS A LAST BURST OF ENERGY HE KNEW HOW CLOSE THEY BOTH WERE TO EXHAUSTION
    1272-141231-0030 BRION SAW SOMETHING CLOSE TO PANIC ON HIS OPPONENT'S FACE WHEN THE MAN FINALLY RECOGNIZED HIS ERROR
    1272-141231-0031 A WAVE OF DESPAIR ROLLED OUT FROM IROLG BRION SENSED IT AND KNEW THE FIFTH POINT WAS HIS
    1272-141231-0032 THEN THE POWERFUL TWIST THAT THRUST IT ASIDE IN AND UNDER THE GUARD
    1462-170138-0000 HE HAD WRITTEN A NUMBER OF BOOKS HIMSELF AMONG THEM A HISTORY OF DANCING A HISTORY OF COSTUME A KEY TO SHAKESPEARE'S SONNETS A STUDY OF THE POETRY OF ERNEST DOWSON ET CETERA
    1462-170138-0001 HUGH'S WRITTEN A DELIGHTFUL PART FOR HER AND SHE'S QUITE INEXPRESSIBLE
    1462-170138-0002 I HAPPEN TO HAVE MAC CONNELL'S BOX FOR TONIGHT OR THERE'D BE NO CHANCE OF OUR GETTING PLACES
    1462-170138-0003 ALEXANDER EXCLAIMED MILDLY
    1462-170138-0004 MYSELF I ALWAYS KNEW SHE HAD IT IN HER
    1462-170138-0005 DO YOU KNOW ALEXANDER MAINHALL LOOKED WITH PERPLEXITY UP INTO THE TOP OF THE HANSOM AND RUBBED HIS PINK CHEEK WITH HIS GLOVED FINGER DO YOU KNOW I SOMETIMES THINK OF TAKING TO CRITICISM SERIOUSLY MYSELF
    1462-170138-0006 WHEN THEY ENTERED THE STAGE BOX ON THE LEFT THE FIRST ACT WAS WELL UNDER WAY THE SCENE BEING THE INTERIOR OF A CABIN IN THE SOUTH OF IRELAND
    1462-170138-0007 AS THEY SAT DOWN A BURST OF APPLAUSE DREW ALEXANDER'S ATTENTION TO THE STAGE
    1462-170138-0008 OF COURSE HILDA IS IRISH THE BURGOYNES HAVE BEEN STAGE PEOPLE FOR GENERATIONS AND SHE HAS THE IRISH VOICE
    1462-170138-0009 IT'S DELIGHTFUL TO HEAR IT IN A LONDON THEATRE
    1462-170138-0010 WHEN SHE BEGAN TO DANCE BY WAY OF SHOWING THE GOSSOONS WHAT SHE HAD SEEN IN THE FAIRY RINGS AT NIGHT THE HOUSE BROKE INTO A PROLONGED UPROAR
    1462-170138-0011 AFTER HER DANCE SHE WITHDREW FROM THE DIALOGUE AND RETREATED TO THE DITCH WALL BACK OF PHILLY'S BURROW WHERE SHE SAT SINGING THE RISING OF THE MOON AND MAKING A WREATH OF PRIMROSES FOR HER DONKEY
    1462-170138-0012 MAC CONNELL LET ME INTRODUCE MISTER BARTLEY ALEXANDER
    1462-170138-0013 THE PLAYWRIGHT GAVE MAINHALL A CURIOUS LOOK OUT OF HIS DEEP SET FADED EYES AND MADE A WRY FACE
    1462-170138-0014 HE NODDED CURTLY AND MADE FOR THE DOOR DODGING ACQUAINTANCES AS HE WENT
    1462-170138-0015 I DARE SAY IT'S QUITE TRUE THAT THERE'S NEVER BEEN ANY ONE ELSE
    1462-170138-0016 HE'S ANOTHER WHO'S AWFULLY KEEN ABOUT HER LET ME INTRODUCE YOU
    1462-170138-0017 SIR HARRY TOWNE BOWED AND SAID THAT HE HAD MET MISTER ALEXANDER AND HIS WIFE IN TOKYO
    1462-170138-0018 I SAY SIR HARRY THE LITTLE GIRL'S GOING FAMOUSLY TO NIGHT ISN'T SHE
    1462-170138-0019 THE FACT IS SHE'S FEELING RATHER SEEDY POOR CHILD
    1462-170138-0020 A LITTLE ATTACK OF NERVES POSSIBLY
    1462-170138-0021 HE BOWED AS THE WARNING BELL RANG AND MAINHALL WHISPERED YOU KNOW LORD WESTMERE OF COURSE THE STOOPED MAN WITH THE LONG GRAY MUSTACHE TALKING TO LADY DOWLE
    1462-170138-0022 IN A MOMENT PEGGY WAS ON THE STAGE AGAIN AND ALEXANDER APPLAUDED VIGOROUSLY WITH THE REST
    1462-170138-0023 IN THE HALF LIGHT HE LOOKED ABOUT AT THE STALLS AND BOXES AND SMILED A LITTLE CONSCIOUSLY RECALLING WITH AMUSEMENT SIR HARRY'S JUDICIAL FROWN
    1462-170138-0024 HE LEANED FORWARD AND BEAMED FELICITATIONS AS WARMLY AS MAINHALL HIMSELF WHEN AT THE END OF THE PLAY SHE CAME AGAIN AND AGAIN BEFORE THE CURTAIN PANTING A LITTLE AND FLUSHED HER EYES DANCING AND HER EAGER NERVOUS LITTLE MOUTH TREMULOUS WITH EXCITEMENT
    1462-170138-0025 ALL THE SAME HE LIFTED HIS GLASS HERE'S TO YOU LITTLE HILDA
    1462-170138-0026 I'M GLAD SHE'S HELD HER OWN SINCE
    1462-170138-0027 IT WAS YOUTH AND POVERTY AND PROXIMITY AND EVERYTHING WAS YOUNG AND KINDLY
    1462-170142-0000 THE LAST TWO DAYS OF THE VOYAGE BARTLEY FOUND ALMOST INTOLERABLE
    1462-170142-0001 EMERGING AT EUSTON AT HALF PAST THREE O'CLOCK IN THE AFTERNOON ALEXANDER HAD HIS LUGGAGE SENT TO THE SAVOY AND DROVE AT ONCE TO BEDFORD SQUARE
    1462-170142-0002 SHE BLUSHED AND SMILED AND FUMBLED HIS CARD IN HER CONFUSION BEFORE SHE RAN UPSTAIRS
    1462-170142-0003 THE ROOM WAS EMPTY WHEN HE ENTERED
    1462-170142-0004 A COAL FIRE WAS CRACKLING IN THE GRATE AND THE LAMPS WERE LIT FOR IT WAS ALREADY BEGINNING TO GROW DARK OUTSIDE
    1462-170142-0005 SHE CALLED HIS NAME ON THE THRESHOLD BUT IN HER SWIFT FLIGHT ACROSS THE ROOM SHE FELT A CHANGE IN HIM AND CAUGHT HERSELF UP SO DEFTLY THAT HE COULD NOT TELL JUST WHEN SHE DID IT
    1462-170142-0006 SHE MERELY BRUSHED HIS CHEEK WITH HER LIPS AND PUT A HAND LIGHTLY AND JOYOUSLY ON EITHER SHOULDER
    1462-170142-0007 I NEVER DREAMED IT WOULD BE YOU BARTLEY
    1462-170142-0008 WHEN DID YOU COME BARTLEY AND HOW DID IT HAPPEN YOU HAVEN'T SPOKEN A WORD
    1462-170142-0009 SHE LOOKED AT HIS HEAVY SHOULDERS AND BIG DETERMINED HEAD THRUST FORWARD LIKE A CATAPULT IN LEASH
    1462-170142-0010 I'LL DO ANYTHING YOU WISH ME TO BARTLEY SHE SAID TREMULOUSLY
    1462-170142-0011 HE PULLED UP A WINDOW AS IF THE AIR WERE HEAVY
    1462-170142-0012 HILDA WATCHED HIM FROM HER CORNER TREMBLING AND SCARCELY BREATHING DARK SHADOWS GROWING ABOUT HER EYES
    1462-170142-0013 IT IT HASN'T ALWAYS MADE YOU MISERABLE HAS IT
    1462-170142-0014 ALWAYS BUT IT'S WORSE NOW
    1462-170142-0015 IT'S UNBEARABLE IT TORTURES ME EVERY MINUTE
    1462-170142-0016 I AM NOT A MAN WHO CAN LIVE TWO LIVES HE WENT ON FEVERISHLY EACH LIFE SPOILS THE OTHER
    1462-170142-0017 I GET NOTHING BUT MISERY OUT OF EITHER
    1462-170142-0018 THERE IS THIS DECEPTION BETWEEN ME AND EVERYTHING
    1462-170142-0019 AT THAT WORD DECEPTION SPOKEN WITH SUCH SELF CONTEMPT THE COLOR FLASHED BACK INTO HILDA'S FACE AS SUDDENLY AS IF SHE HAD BEEN STRUCK BY A WHIPLASH
    1462-170142-0020 SHE BIT HER LIP AND LOOKED DOWN AT HER HANDS WHICH WERE CLASPED TIGHTLY IN FRONT OF HER
    1462-170142-0021 COULD YOU COULD YOU SIT DOWN AND TALK ABOUT IT QUIETLY BARTLEY AS IF I WERE A FRIEND AND NOT SOME ONE WHO HAD TO BE DEFIED
    1462-170142-0022 HE DROPPED BACK HEAVILY INTO HIS CHAIR BY THE FIRE
    1462-170142-0023 I HAVE THOUGHT ABOUT IT UNTIL I AM WORN OUT
    1462-170142-0024 AFTER THE VERY FIRST
    1462-170142-0025 HILDA'S FACE QUIVERED BUT SHE WHISPERED YES I THINK IT MUST HAVE BEEN
    1462-170142-0026 SHE PRESSED HIS HAND GENTLY IN GRATITUDE WEREN'T YOU HAPPY THEN AT ALL
    1462-170142-0027 SOMETHING OF THEIR TROUBLING SWEETNESS CAME BACK TO ALEXANDER TOO
    1462-170142-0028 PRESENTLY IT STOLE BACK TO HIS COAT SLEEVE
    1462-170142-0029 YES HILDA I KNOW THAT HE SAID SIMPLY
    1462-170142-0030 I UNDERSTAND BARTLEY I WAS WRONG
    1462-170142-0031 SHE LISTENED INTENTLY BUT SHE HEARD NOTHING BUT THE CREAKING OF HIS CHAIR
    1462-170142-0032 YOU WANT ME TO SAY IT SHE WHISPERED
    1462-170142-0033 BARTLEY LEANED HIS HEAD IN HIS HANDS AND SPOKE THROUGH HIS TEETH
    1462-170142-0034 IT'S GOT TO BE A CLEAN BREAK HILDA
    1462-170142-0035 OH BARTLEY WHAT AM I TO DO
    1462-170142-0036 YOU ASK ME TO STAY AWAY FROM YOU BECAUSE YOU WANT ME
    1462-170142-0037 I WILL ASK THE LEAST IMAGINABLE BUT I MUST HAVE SOMETHING
    1462-170142-0038 HILDA SAT ON THE ARM OF IT AND PUT HER HANDS LIGHTLY ON HIS SHOULDERS
    1462-170142-0039 YOU SEE LOVING SOME ONE AS I LOVE YOU MAKES THE WHOLE WORLD DIFFERENT
    1462-170142-0040 AND THEN YOU CAME BACK NOT CARING VERY MUCH BUT IT MADE NO DIFFERENCE
    1462-170142-0041 SHE SLID TO THE FLOOR BESIDE HIM AS IF SHE WERE TOO TIRED TO SIT UP ANY LONGER
    1462-170142-0042 DON'T CRY DON'T CRY HE WHISPERED
    1462-170145-0000 ON THE LAST SATURDAY IN APRIL THE NEW YORK TIMES PUBLISHED AN ACCOUNT OF THE STRIKE COMPLICATIONS WHICH WERE DELAYING ALEXANDER'S NEW JERSEY BRIDGE AND STATED THAT THE ENGINEER HIMSELF WAS IN TOWN AND AT HIS OFFICE ON WEST TENTH STREET
    1462-170145-0001 OVER THE FIREPLACE THERE WAS A LARGE OLD FASHIONED GILT MIRROR
    1462-170145-0002 HE ROSE AND CROSSED THE ROOM QUICKLY
    1462-170145-0003 OF COURSE I KNOW BARTLEY SHE SAID AT LAST THAT AFTER THIS YOU WON'T OWE ME THE LEAST CONSIDERATION BUT WE SAIL ON TUESDAY
    1462-170145-0004 I SAW THAT INTERVIEW IN THE PAPER YESTERDAY TELLING WHERE YOU WERE AND I THOUGHT I HAD TO SEE YOU THAT'S ALL GOOD NIGHT I'M GOING NOW
    1462-170145-0005 LET ME TAKE OFF YOUR COAT AND YOUR BOOTS THEY'RE OOZING WATER
    1462-170145-0006 IF YOU'D SENT ME A NOTE OR TELEPHONED ME OR ANYTHING
    1462-170145-0007 I TOLD MYSELF THAT IF I WERE REALLY THINKING OF YOU AND NOT OF MYSELF A LETTER WOULD BE BETTER THAN NOTHING
    1462-170145-0008 HE PAUSED THEY NEVER DID TO ME
    1462-170145-0009 OH BARTLEY DID YOU WRITE TO ME
    1462-170145-0010 ALEXANDER SLIPPED HIS ARM ABOUT HER
    1462-170145-0011 I THINK I HAVE FELT THAT YOU WERE COMING
    1462-170145-0012 HE BENT HIS FACE OVER HER HAIR
    1462-170145-0013 AND I SHE WHISPERED I FELT THAT YOU WERE FEELING THAT
    1462-170145-0014 BUT WHEN I CAME I THOUGHT I HAD BEEN MISTAKEN
    1462-170145-0015 I'VE BEEN UP IN CANADA WITH MY BRIDGE AND I ARRANGED NOT TO COME TO NEW YORK UNTIL AFTER YOU HAD GONE
    1462-170145-0016 THEN WHEN YOUR MANAGER ADDED TWO MORE WEEKS I WAS ALREADY COMMITTED
    1462-170145-0017 I'M GOING TO DO WHAT YOU ASKED ME TO DO WHEN YOU WERE IN LONDON
    1462-170145-0018 ONLY I'LL DO IT MORE COMPLETELY
    1462-170145-0019 THEN YOU DON'T KNOW WHAT YOU'RE TALKING ABOUT
    1462-170145-0020 YES I KNOW VERY WELL
    1462-170145-0021 ALEXANDER FLUSHED ANGRILY
    1462-170145-0022 I DON'T KNOW WHAT I OUGHT TO SAY BUT I DON'T BELIEVE YOU'D BE HAPPY TRULY I DON'T AREN'T YOU TRYING TO FRIGHTEN ME
    1673-143396-0000 A LAUDABLE REGARD FOR THE HONOR OF THE FIRST PROSELYTE HAS COUNTENANCED THE BELIEF THE HOPE THE WISH THAT THE EBIONITES OR AT LEAST THE NAZARENES WERE DISTINGUISHED ONLY BY THEIR OBSTINATE PERSEVERANCE IN THE PRACTICE OF THE MOSAIC RITES
    1673-143396-0001 THEIR CHURCHES HAVE DISAPPEARED THEIR BOOKS ARE OBLITERATED THEIR OBSCURE FREEDOM MIGHT ALLOW A LATITUDE OF FAITH AND THE SOFTNESS OF THEIR INFANT CREED WOULD BE VARIOUSLY MOULDED BY THE ZEAL OR PRUDENCE OF THREE HUNDRED YEARS
    1673-143396-0002 YET THE MOST CHARITABLE CRITICISM MUST REFUSE THESE SECTARIES ANY KNOWLEDGE OF THE PURE AND PROPER DIVINITY OF CHRIST
    1673-143396-0003 HIS PROGRESS FROM INFANCY TO YOUTH AND MANHOOD WAS MARKED BY A REGULAR INCREASE IN STATURE AND WISDOM AND AFTER A PAINFUL AGONY OF MIND AND BODY HE EXPIRED ON THE CROSS
    1673-143396-0004 HE LIVED AND DIED FOR THE SERVICE OF MANKIND BUT THE LIFE AND DEATH OF SOCRATES HAD LIKEWISE BEEN DEVOTED TO THE CAUSE OF RELIGION AND JUSTICE AND ALTHOUGH THE STOIC OR THE HERO MAY DISDAIN THE HUMBLE VIRTUES OF JESUS THE TEARS WHICH HE SHED OVER HIS FRIEND AND COUNTRY MAY BE ESTEEMED THE PUREST EVIDENCE OF HIS HUMANITY
    1673-143396-0005 THE SON OF A VIRGIN GENERATED BY THE INEFFABLE OPERATION OF THE HOLY SPIRIT WAS A CREATURE WITHOUT EXAMPLE OR RESEMBLANCE SUPERIOR IN EVERY ATTRIBUTE OF MIND AND BODY TO THE CHILDREN OF ADAM
    1673-143396-0006 NOR COULD IT SEEM STRANGE OR INCREDIBLE THAT THE FIRST OF THESE AEONS THE LOGOS OR WORD OF GOD OF THE SAME SUBSTANCE WITH THE FATHER SHOULD DESCEND UPON EARTH TO DELIVER THE HUMAN RACE FROM VICE AND ERROR AND TO CONDUCT THEM IN THE PATHS OF LIFE AND IMMORTALITY
    1673-143396-0007 BUT THE PREVAILING DOCTRINE OF THE ETERNITY AND INHERENT PRAVITY OF MATTER INFECTED THE PRIMITIVE CHURCHES OF THE EAST
    1673-143396-0008 MANY AMONG THE GENTILE PROSELYTES REFUSED TO BELIEVE THAT A CELESTIAL SPIRIT AN UNDIVIDED PORTION OF THE FIRST ESSENCE HAD BEEN PERSONALLY UNITED WITH A MASS OF IMPURE AND CONTAMINATED FLESH AND IN THEIR ZEAL FOR THE DIVINITY THEY PIOUSLY ABJURED THE HUMANITY OF CHRIST
    1673-143396-0009 HE FIRST APPEARED ON THE BANKS OF THE JORDAN IN THE FORM OF PERFECT MANHOOD BUT IT WAS A FORM ONLY AND NOT A SUBSTANCE A HUMAN FIGURE CREATED BY THE HAND OF OMNIPOTENCE TO IMITATE THE FACULTIES AND ACTIONS OF A MAN AND TO IMPOSE A PERPETUAL ILLUSION ON THE SENSES OF HIS FRIENDS AND ENEMIES
    1673-143396-0010 BUT THE RASHNESS OF THESE CONCESSIONS HAS ENCOURAGED A MILDER SENTIMENT OF THOSE OF THE DOCETES WHO TAUGHT NOT THAT CHRIST WAS A PHANTOM BUT THAT HE WAS CLOTHED WITH AN IMPASSIBLE AND INCORRUPTIBLE BODY
    1673-143396-0011 A FOETUS THAT COULD INCREASE FROM AN INVISIBLE POINT TO ITS FULL MATURITY A CHILD THAT COULD ATTAIN THE STATURE OF PERFECT MANHOOD WITHOUT DERIVING ANY NOURISHMENT FROM THE ORDINARY SOURCES MIGHT CONTINUE TO EXIST WITHOUT REPAIRING A DAILY WASTE BY A DAILY SUPPLY OF EXTERNAL MATTER
    1673-143396-0012 IN THEIR EYES JESUS OF NAZARETH WAS A MERE MORTAL THE LEGITIMATE SON OF JOSEPH AND MARY BUT HE WAS THE BEST AND WISEST OF THE HUMAN RACE SELECTED AS THE WORTHY INSTRUMENT TO RESTORE UPON EARTH THE WORSHIP OF THE TRUE AND SUPREME DEITY
    1673-143396-0013 WHEN THE MESSIAH WAS DELIVERED INTO THE HANDS OF THE JEWS THE CHRIST AN IMMORTAL AND IMPASSIBLE BEING FORSOOK HIS EARTHLY TABERNACLE FLEW BACK TO THE PLEROMA OR WORLD OF SPIRITS AND LEFT THE SOLITARY JESUS TO SUFFER TO COMPLAIN AND TO EXPIRE
    1673-143396-0014 BUT THE JUSTICE AND GENEROSITY OF SUCH A DESERTION ARE STRONGLY QUESTIONABLE AND THE FATE OF AN INNOCENT MARTYR AT FIRST IMPELLED AND AT LENGTH ABANDONED BY HIS DIVINE COMPANION MIGHT PROVOKE THE PITY AND INDIGNATION OF THE PROFANE
    1673-143396-0015 THEIR MURMURS WERE VARIOUSLY SILENCED BY THE SECTARIES WHO ESPOUSED AND MODIFIED THE DOUBLE SYSTEM OF CERINTHUS
    1673-143396-0016 THE WORTHY FRIEND OF ATHANASIUS THE WORTHY ANTAGONIST OF JULIAN HE BRAVELY WRESTLED WITH THE ARIANS AND POLYTHEISTS AND THOUGH HE AFFECTED THE RIGOR OF GEOMETRICAL DEMONSTRATION HIS COMMENTARIES REVEALED THE LITERAL AND ALLEGORICAL SENSE OF THE SCRIPTURES
    1673-143396-0017 YET AS THE PROFOUND DOCTOR HAD BEEN TERRIFIED AT HIS OWN RASHNESS APOLLINARIS WAS HEARD TO MUTTER SOME FAINT ACCENTS OF EXCUSE AND EXPLANATION
    1673-143396-0018 HE ACQUIESCED IN THE OLD DISTINCTION OF THE GREEK PHILOSOPHERS BETWEEN THE RATIONAL AND SENSITIVE SOUL OF MAN THAT HE MIGHT RESERVE THE LOGOS FOR INTELLECTUAL FUNCTIONS AND EMPLOY THE SUBORDINATE HUMAN PRINCIPLE IN THE MEANER ACTIONS OF ANIMAL LIFE
    1673-143396-0019 BUT INSTEAD OF A TEMPORARY AND OCCASIONAL ALLIANCE THEY ESTABLISHED AND WE STILL EMBRACE THE SUBSTANTIAL INDISSOLUBLE AND EVERLASTING UNION OF A PERFECT GOD WITH A PERFECT MAN OF THE SECOND PERSON OF THE TRINITY WITH A REASONABLE SOUL AND HUMAN FLESH
    1673-143396-0020 UNDER THE TUITION OF THE ABBOT SERAPION HE APPLIED HIMSELF TO ECCLESIASTICAL STUDIES WITH SUCH INDEFATIGABLE ARDOR THAT IN THE COURSE OF ONE SLEEPLESS NIGHT HE HAS PERUSED THE FOUR GOSPELS THE CATHOLIC EPISTLES AND THE EPISTLE TO THE ROMANS
    1673-143397-0000 ARDENT IN THE PROSECUTION OF HERESY CYRIL AUSPICIOUSLY OPENED HIS REIGN BY OPPRESSING THE NOVATIANS THE MOST INNOCENT AND HARMLESS OF THE SECTARIES
    1673-143397-0001 WITHOUT ANY LEGAL SENTENCE WITHOUT ANY ROYAL MANDATE THE PATRIARCH AT THE DAWN OF DAY LED A SEDITIOUS MULTITUDE TO THE ATTACK OF THE SYNAGOGUES
    1673-143397-0002 SUCH CRIMES WOULD HAVE DESERVED THE ANIMADVERSION OF THE MAGISTRATE BUT IN THIS PROMISCUOUS OUTRAGE THE INNOCENT WERE CONFOUNDED WITH THE GUILTY AND ALEXANDRIA WAS IMPOVERISHED BY THE LOSS OF A WEALTHY AND INDUSTRIOUS COLONY
    1673-143397-0003 THE ZEAL OF CYRIL EXPOSED HIM TO THE PENALTIES OF THE JULIAN LAW BUT IN A FEEBLE GOVERNMENT AND A SUPERSTITIOUS AGE HE WAS SECURE OF IMPUNITY AND EVEN OF PRAISE
    1673-143397-0004 ORESTES COMPLAINED BUT HIS JUST COMPLAINTS WERE TOO QUICKLY FORGOTTEN BY THE MINISTERS OF THEODOSIUS AND TOO DEEPLY REMEMBERED BY A PRIEST WHO AFFECTED TO PARDON AND CONTINUED TO HATE THE PRAEFECT OF EGYPT
    1673-143397-0005 A RUMOR WAS SPREAD AMONG THE CHRISTIANS THAT THE DAUGHTER OF THEON WAS THE ONLY OBSTACLE TO THE RECONCILIATION OF THE PRAEFECT AND THE ARCHBISHOP AND THAT OBSTACLE WAS SPEEDILY REMOVED
    1673-143397-0006 WHICH OPPRESSED THE METROPOLITANS OF EUROPE AND ASIA INVADED THE PROVINCES OF ANTIOCH AND ALEXANDRIA AND MEASURED THEIR DIOCESE BY THE LIMITS OF THE EMPIRE
    1673-143397-0007 EXTERMINATE WITH ME THE HERETICS AND WITH YOU I WILL EXTERMINATE THE PERSIANS
    1673-143397-0008 AT THESE BLASPHEMOUS SOUNDS THE PILLARS OF THE SANCTUARY WERE SHAKEN
    1673-143397-0009 BUT THE VATICAN RECEIVED WITH OPEN ARMS THE MESSENGERS OF EGYPT
    1673-143397-0010 THE VANITY OF CELESTINE WAS FLATTERED BY THE APPEAL AND THE PARTIAL VERSION OF A MONK DECIDED THE FAITH OF THE POPE WHO WITH HIS LATIN CLERGY WAS IGNORANT OF THE LANGUAGE THE ARTS AND THE THEOLOGY OF THE GREEKS
    1673-143397-0011 NESTORIUS WHO DEPENDED ON THE NEAR APPROACH OF HIS EASTERN FRIENDS PERSISTED LIKE HIS PREDECESSOR CHRYSOSTOM TO DISCLAIM THE JURISDICTION AND TO DISOBEY THE SUMMONS OF HIS ENEMIES THEY HASTENED HIS TRIAL AND HIS ACCUSER PRESIDED IN THE SEAT OF JUDGMENT
    1673-143397-0012 SIXTY EIGHT BISHOPS TWENTY TWO OF METROPOLITAN RANK DEFENDED HIS CAUSE BY A MODEST AND TEMPERATE PROTEST THEY WERE EXCLUDED FROM THE COUNCILS OF THEIR BRETHREN
    1673-143397-0013 BY THE VIGILANCE OF MEMNON THE CHURCHES WERE SHUT AGAINST THEM AND A STRONG GARRISON WAS THROWN INTO THE CATHEDRAL
    1673-143397-0014 DURING A BUSY PERIOD OF THREE MONTHS THE EMPEROR TRIED EVERY METHOD EXCEPT THE MOST EFFECTUAL MEANS OF INDIFFERENCE AND CONTEMPT TO RECONCILE THIS THEOLOGICAL QUARREL
    1673-143397-0015 RETURN TO YOUR PROVINCES AND MAY YOUR PRIVATE VIRTUES REPAIR THE MISCHIEF AND SCANDAL OF YOUR MEETING
    1673-143397-0016 THE FEEBLE SON OF ARCADIUS WAS ALTERNATELY SWAYED BY HIS WIFE AND SISTER BY THE EUNUCHS AND WOMEN OF THE PALACE SUPERSTITION AND AVARICE WERE THEIR RULING PASSIONS AND THE ORTHODOX CHIEFS WERE ASSIDUOUS IN THEIR ENDEAVORS TO ALARM THE FORMER AND TO GRATIFY THE LATTER
    1673-143397-0017 BUT IN THIS AWFUL MOMENT OF THE DANGER OF THE CHURCH THEIR VOW WAS SUPERSEDED BY A MORE SUBLIME AND INDISPENSABLE DUTY
    1673-143397-0018 AT THE SAME TIME EVERY AVENUE OF THE THRONE WAS ASSAULTED WITH GOLD
    1673-143397-0019 THE PAST HE REGRETTED HE WAS DISCONTENTED WITH THE PRESENT AND THE FUTURE HE HAD REASON TO DREAD THE ORIENTAL BISHOPS SUCCESSIVELY DISENGAGED THEIR CAUSE FROM HIS UNPOPULAR NAME AND EACH DAY DECREASED THE NUMBER OF THE SCHISMATICS WHO REVERED NESTORIUS AS THE CONFESSOR OF THE FAITH
    1673-143397-0020 A WANDERING TRIBE OF THE BLEMMYES OR NUBIANS INVADED HIS SOLITARY PRISON IN THEIR RETREAT THEY DISMISSED A CROWD OF USELESS CAPTIVES BUT NO SOONER HAD NESTORIUS REACHED THE BANKS OF THE NILE THAN HE WOULD GLADLY HAVE ESCAPED FROM A ROMAN AND ORTHODOX CITY TO THE MILDER SERVITUDE OF THE SAVAGES
    174-168635-0000 HE HAD NEVER BEEN FATHER LOVER HUSBAND FRIEND
    174-168635-0001 THE HEART OF THAT EX CONVICT WAS FULL OF VIRGINITY
    174-168635-0002 HIS SISTER AND HIS SISTER'S CHILDREN HAD LEFT HIM ONLY A VAGUE AND FAR OFF MEMORY WHICH HAD FINALLY ALMOST COMPLETELY VANISHED HE HAD MADE EVERY EFFORT TO FIND THEM AND NOT HAVING BEEN ABLE TO FIND THEM HE HAD FORGOTTEN THEM
    174-168635-0003 HE SUFFERED ALL THE PANGS OF A MOTHER AND HE KNEW NOT WHAT IT MEANT FOR THAT GREAT AND SINGULAR MOVEMENT OF A HEART WHICH BEGINS TO LOVE IS A VERY OBSCURE AND A VERY SWEET THING
    174-168635-0004 ONLY AS HE WAS FIVE AND FIFTY AND COSETTE EIGHT YEARS OF AGE ALL THAT MIGHT HAVE BEEN LOVE IN THE WHOLE COURSE OF HIS LIFE FLOWED TOGETHER INTO A SORT OF INEFFABLE LIGHT
    174-168635-0005 COSETTE ON HER SIDE HAD ALSO UNKNOWN TO HERSELF BECOME ANOTHER BEING POOR LITTLE THING
    174-168635-0006 SHE FELT THAT WHICH SHE HAD NEVER FELT BEFORE A SENSATION OF EXPANSION
    174-168635-0007 THE MAN NO LONGER PRODUCED ON HER THE EFFECT OF BEING OLD OR POOR SHE THOUGHT JEAN VALJEAN HANDSOME JUST AS SHE THOUGHT THE HOVEL PRETTY
    174-168635-0008 NATURE A DIFFERENCE OF FIFTY YEARS HAD SET A PROFOUND GULF BETWEEN JEAN VALJEAN AND COSETTE DESTINY FILLED IN THIS GULF
    174-168635-0009 TO MEET WAS TO FIND EACH OTHER
    174-168635-0010 WHEN THESE TWO SOULS PERCEIVED EACH OTHER THEY RECOGNIZED EACH OTHER AS NECESSARY TO EACH OTHER AND EMBRACED EACH OTHER CLOSELY
    174-168635-0011 MOREOVER JEAN VALJEAN HAD CHOSEN HIS REFUGE WELL
    174-168635-0012 HE HAD PAID HER SIX MONTHS IN ADVANCE AND HAD COMMISSIONED THE OLD WOMAN TO FURNISH THE CHAMBER AND DRESSING ROOM AS WE HAVE SEEN
    174-168635-0013 WEEK FOLLOWED WEEK THESE TWO BEINGS LED A HAPPY LIFE IN THAT HOVEL
    174-168635-0014 COSETTE WAS NO LONGER IN RAGS SHE WAS IN MOURNING
    174-168635-0015 AND THEN HE TALKED OF HER MOTHER AND HE MADE HER PRAY
    174-168635-0016 HE PASSED HOURS IN WATCHING HER DRESSING AND UNDRESSING HER DOLL AND IN LISTENING TO HER PRATTLE
    174-168635-0017 THE BEST OF US ARE NOT EXEMPT FROM EGOTISTICAL THOUGHTS
    174-168635-0018 HE HAD RETURNED TO PRISON THIS TIME FOR HAVING DONE RIGHT HE HAD QUAFFED FRESH BITTERNESS DISGUST AND LASSITUDE WERE OVERPOWERING HIM EVEN THE MEMORY OF THE BISHOP PROBABLY SUFFERED A TEMPORARY ECLIPSE THOUGH SURE TO REAPPEAR LATER ON LUMINOUS AND TRIUMPHANT BUT AFTER ALL THAT SACRED MEMORY WAS GROWING DIM
    174-168635-0019 WHO KNOWS WHETHER JEAN VALJEAN HAD NOT BEEN ON THE EVE OF GROWING DISCOURAGED AND OF FALLING ONCE MORE
    174-168635-0020 ALAS HE WALKED WITH NO LESS INDECISION THAN COSETTE
    174-168635-0021 HE PROTECTED HER AND SHE STRENGTHENED HIM
    174-168635-0022 HE WAS THAT CHILD'S STAY AND SHE WAS HIS PROP
    174-50561-0000 FORGOTTEN TOO THE NAME OF GILLIAN THE LOVELY CAPTIVE
    174-50561-0001 WORSE AND WORSE HE IS EVEN PRESUMED TO BE THE CAPTIVE'S SWEETHEART WHO WHEEDLES THE FLOWER THE RING AND THE PRISON KEY OUT OF THE STRICT VIRGINS FOR HIS OWN PURPOSES AND FLIES WITH HER AT LAST IN HIS SHALLOP ACROSS THE SEA TO LIVE WITH HER HAPPILY EVER AFTER
    174-50561-0002 BUT THIS IS A FALLACY
    174-50561-0003 THE WANDERING SINGER APPROACHES THEM WITH HIS LUTE
    174-50561-0004 THE EMPEROR'S DAUGHTER
    174-50561-0005 LADY LADY MY ROSE WHITE LADY BUT WILL YOU NOT HEAR A ROUNDEL LADY
    174-50561-0006 O IF YOU PLAY US A ROUNDEL SINGER HOW CAN THAT HARM THE EMPEROR'S DAUGHTER
    174-50561-0007 SHE WOULD NOT SPEAK THOUGH WE DANCED A WEEK WITH HER THOUGHTS A THOUSAND LEAGUES OVER THE WATER SINGER SINGER WANDERING SINGER O MY HONEY SWEET SINGER
    174-50561-0008 BUT IF I PLAY YOU A ROUNDEL LADY GET ME A GIFT FROM THE EMPEROR'S DAUGHTER HER FINGER RING FOR MY FINGER BRING THOUGH SHE'S PLEDGED A THOUSAND LEAGUES OVER THE WATER LADY LADY MY FAIR LADY O MY ROSE WHITE LADY
    174-50561-0009 THE WANDERING SINGER
    174-50561-0010 BUT I DID ONCE HAVE THE LUCK TO HEAR AND SEE THE LADY PLAYED IN ENTIRETY THE CHILDREN HAD BEEN GRANTED LEAVE TO PLAY JUST ONE MORE GAME BEFORE BED TIME AND OF COURSE THEY CHOSE THE LONGEST AND PLAYED IT WITHOUT MISSING A SYLLABLE
    174-50561-0011 THE LADIES IN YELLOW DRESSES STAND AGAIN IN A RING ABOUT THE EMPEROR'S DAUGHTER AND ARE FOR THE LAST TIME ACCOSTED BY THE SINGER WITH HIS LUTE
    174-50561-0012 THE WANDERING SINGER
    174-50561-0013 I'LL PLAY FOR YOU NOW NEATH THE APPLE BOUGH AND YOU SHALL DREAM ON THE LAWN SO SHADY LADY LADY MY FAIR LADY O MY APPLE GOLD LADY
    174-50561-0014 THE LADIES
    174-50561-0015 NOW YOU MAY PLAY A SERENA SINGER A DREAM OF NIGHT FOR AN APPLE GOLD LADY FOR THE FRUIT IS NOW ON THE APPLE BOUGH AND THE MOON IS UP AND THE LAWN IS SHADY SINGER SINGER WANDERING SINGER O MY HONEY SWEET SINGER
    174-50561-0016 ONCE MORE THE SINGER PLAYS AND THE LADIES DANCE BUT ONE BY ONE THEY FALL ASLEEP TO THE DROWSY MUSIC AND THEN THE SINGER STEPS INTO THE RING AND UNLOCKS THE TOWER AND KISSES THE EMPEROR'S DAUGHTER
    174-50561-0017 I DON'T KNOW WHAT BECOMES OF THE LADIES
    174-50561-0018 BED TIME CHILDREN
    174-50561-0019 YOU SEE THE TREATMENT IS A TRIFLE FANCIFUL
    174-84280-0000 HOW WE MUST SIMPLIFY
    174-84280-0001 IT SEEMS TO ME MORE AND MORE AS I LIVE LONGER THAT MOST POETRY AND MOST LITERATURE AND PARTICULARLY THE LITERATURE OF THE PAST IS DISCORDANT WITH THE VASTNESS AND VARIETY THE RESERVES AND RESOURCES AND RECUPERATIONS OF LIFE AS WE LIVE IT TO DAY
    174-84280-0002 IT IS THE EXPRESSION OF LIFE UNDER CRUDER AND MORE RIGID CONDITIONS THAN OURS LIVED BY PEOPLE WHO LOVED AND HATED MORE NAIVELY AGED SOONER AND DIED YOUNGER THAN WE DO
    174-84280-0003 WE RANGE WIDER LAST LONGER AND ESCAPE MORE AND MORE FROM INTENSITY TOWARDS UNDERSTANDING
    174-84280-0004 AND ALREADY THIS ASTOUNDING BLOW BEGINS TO TAKE ITS PLACE AMONG OTHER EVENTS AS A THING STRANGE AND TERRIBLE INDEED BUT RELATED TO ALL THE STRANGENESS AND MYSTERY OF LIFE PART OF THE UNIVERSAL MYSTERIES OF DESPAIR AND FUTILITY AND DEATH THAT HAVE TROUBLED MY CONSCIOUSNESS SINCE CHILDHOOD
    174-84280-0005 FOR A TIME THE DEATH OF MARY OBSCURED HER LIFE FOR ME BUT NOW HER LIVING PRESENCE IS MORE IN MY MIND AGAIN
    174-84280-0006 IT WAS THAT IDEA OF WASTE THAT DOMINATED MY MIND IN A STRANGE INTERVIEW I HAD WITH JUSTIN
    174-84280-0007 I BECAME GROTESQUELY ANXIOUS TO ASSURE HIM THAT INDEED SHE AND I HAD BEEN AS THEY SAY INNOCENT THROUGHOUT OUR LAST DAY TOGETHER
    174-84280-0008 YOU WERE WRONG IN ALL THAT I SAID SHE KEPT HER FAITH WITH YOU
    174-84280-0009 WE NEVER PLANNED TO MEET AND WHEN WE MET
    174-84280-0010 IF WE HAD BEEN BROTHER AND SISTER INDEED THERE WAS NOTHING
    174-84280-0011 BUT NOW IT DOESN'T SEEM TO MATTER VERY MUCH
    174-84280-0012 AND IT IS UPON THIS EFFECT OF SWEET AND BEAUTIFUL POSSIBILITIES CAUGHT IN THE NET OF ANIMAL JEALOUSIES AND THOUGHTLESS MOTIVES AND ANCIENT RIGID INSTITUTIONS THAT I WOULD END THIS WRITING
    174-84280-0013 IN MARY IT SEEMS TO ME I FOUND BOTH WOMANHOOD AND FELLOWSHIP I FOUND WHAT MANY HAVE DREAMT OF LOVE AND FRIENDSHIP FREELY GIVEN AND I COULD DO NOTHING BUT CLUTCH AT HER TO MAKE HER MY POSSESSION
    174-84280-0014 WHAT ALTERNATIVE WAS THERE FOR HER
    174-84280-0015 SHE WAS DESTROYED NOT MERELY BY THE UNCONSIDERED UNDISCIPLINED PASSIONS OF HER HUSBAND AND HER LOVER BUT BY THE VAST TRADITION THAT SUSTAINS AND ENFORCES THE SUBJUGATION OF HER SEX
    1919-142785-0000 ILLUSTRATION LONG PEPPER
    1919-142785-0001 LONG PEPPER THIS IS THE PRODUCE OF A DIFFERENT PLANT FROM THAT WHICH PRODUCES THE BLACK IT CONSISTING OF THE HALF RIPE FLOWER HEADS OF WHAT NATURALISTS CALL PIPER LONGUM AND CHABA
    1919-142785-0002 ORIGINALLY THE MOST VALUABLE OF THESE WERE FOUND IN THE SPICE ISLANDS OR MOLUCCAS OF THE INDIAN OCEAN AND WERE HIGHLY PRIZED BY THE NATIONS OF ANTIQUITY
    1919-142785-0003 THE LONG PEPPER IS LESS AROMATIC THAN THE BLACK BUT ITS OIL IS MORE PUNGENT
    1919-142785-0004 THEN ADD THE YOLKS OF THE EGGS WELL BEATEN STIR THEM TO THE SAUCE BUT DO NOT ALLOW IT TO BOIL AND SERVE VERY HOT
    1919-142785-0005 MODE PARE AND SLICE THE CUCUMBERS AS FOR THE TABLE SPRINKLE WELL WITH SALT AND LET THEM REMAIN FOR TWENTY FOUR HOURS STRAIN OFF THE LIQUOR PACK IN JARS A THICK LAYER OF CUCUMBERS AND SALT ALTERNATELY TIE DOWN CLOSELY AND WHEN WANTED FOR USE TAKE OUT THE QUANTITY REQUIRED
    1919-142785-0006 ILLUSTRATION THE CUCUMBER
    1919-142785-0007 MODE CHOOSE THE GREENEST CUCUMBERS AND THOSE THAT ARE MOST FREE FROM SEEDS PUT THEM IN STRONG SALT AND WATER WITH A CABBAGE LEAF TO KEEP THEM DOWN TIE A PAPER OVER THEM AND PUT THEM IN A WARM PLACE TILL THEY ARE YELLOW THEN WASH THEM AND SET THEM OVER THE FIRE IN FRESH WATER WITH A VERY LITTLE SALT AND ANOTHER CABBAGE LEAF OVER THEM COVER VERY CLOSELY BUT TAKE CARE THEY DO NOT BOIL
    1919-142785-0008 PUT THE SUGAR WITH ONE QUARTER PINT OF WATER IN A SAUCEPAN OVER THE FIRE REMOVE THE SCUM AS IT RISES AND ADD THE LEMON PEEL AND GINGER WITH THE OUTSIDE SCRAPED OFF WHEN THE SYRUP IS TOLERABLY THICK TAKE IT OFF THE FIRE AND WHEN COLD WIPE THE CUCUMBERS DRY AND PUT THEM IN
    1919-142785-0009 SEASONABLE THIS RECIPE SHOULD BE USED IN JUNE JULY OR AUGUST
    1919-142785-0010 SOLID ROCKS OF SALT ARE ALSO FOUND IN VARIOUS PARTS OF THE WORLD AND THE COUNTY OF CHESTER CONTAINS MANY OF THESE MINES AND IT IS FROM THERE THAT MUCH OF OUR SALT COMES
    1919-142785-0011 SOME SPRINGS ARE SO HIGHLY IMPREGNATED WITH SALT AS TO HAVE RECEIVED THE NAME OF BRINE SPRINGS AND ARE SUPPOSED TO HAVE BECOME SO BY PASSING THROUGH THE SALT ROCKS BELOW GROUND AND THUS DISSOLVING A PORTION OF THIS MINERAL SUBSTANCE
    1919-142785-0012 MODE PUT THE MILK IN A VERY CLEAN SAUCEPAN AND LET IT BOIL
    1919-142785-0013 BEAT THE EGGS STIR TO THEM THE MILK AND POUNDED SUGAR AND PUT THE MIXTURE INTO A JUG
    1919-142785-0014 PLACE THE JUG IN A SAUCEPAN OF BOILING WATER KEEP STIRRING WELL UNTIL IT THICKENS BUT DO NOT ALLOW IT TO BOIL OR IT WILL CURDLE
    1919-142785-0015 WHEN IT IS SUFFICIENTLY THICK TAKE IT OFF AS IT SHOULD NOT BOIL
    1919-142785-0016 ILLUSTRATION THE LEMON
    1919-142785-0017 THE LEMON THIS FRUIT IS A NATIVE OF ASIA AND IS MENTIONED BY VIRGIL AS AN ANTIDOTE TO POISON
    1919-142785-0018 IT IS HARDIER THAN THE ORANGE AND AS ONE OF THE CITRON TRIBE WAS BROUGHT INTO EUROPE BY THE ARABIANS
    1919-142785-0019 THE LEMON WAS FIRST CULTIVATED IN ENGLAND IN THE BEGINNING OF THE SEVENTEENTH CENTURY AND IS NOW OFTEN TO BE FOUND IN OUR GREEN HOUSES
    1919-142785-0020 THIS JUICE WHICH IS CALLED CITRIC ACID MAY BE PRESERVED IN BOTTLES FOR A CONSIDERABLE TIME BY COVERING IT WITH A THIN STRATUM OF OIL
    1919-142785-0021 TO PICKLE EGGS
    1919-142785-0022 SEASONABLE THIS SHOULD BE MADE ABOUT EASTER AS AT THIS TIME EGGS ARE PLENTIFUL AND CHEAP
    1919-142785-0023 A STORE OF PICKLED EGGS WILL BE FOUND VERY USEFUL AND ORNAMENTAL IN SERVING WITH MANY FIRST AND SECOND COURSE DISHES
    1919-142785-0024 ILLUSTRATION GINGER
    1919-142785-0025 THE GINGER PLANT KNOWN TO NATURALISTS AS ZINGIBER OFFICINALE IS A NATIVE OF THE EAST AND WEST INDIES
    1919-142785-0026 IN JAMAICA IT FLOWERS ABOUT AUGUST OR SEPTEMBER FADING ABOUT THE END OF THE YEAR
    1919-142785-0027 BEAT THE YOLKS OF THE OTHER TWO EGGS ADD THEM WITH A LITTLE FLOUR AND SALT TO THOSE POUNDED MIX ALL WELL TOGETHER AND ROLL INTO BALLS
    1919-142785-0028 BOIL THEM BEFORE THEY ARE PUT INTO THE SOUP OR OTHER DISH THEY MAY BE INTENDED FOR
    1919-142785-0029 LEMON JUICE MAY BE ADDED AT PLEASURE
    1919-142785-0030 MODE PUT THE WHOLE OF THE INGREDIENTS INTO A BOTTLE AND LET IT REMAIN FOR A FORTNIGHT IN A WARM PLACE OCCASIONALLY SHAKING UP THE CONTENTS
    1919-142785-0031 THEY OUGHT TO BE TAKEN UP IN THE AUTUMN AND WHEN DRIED IN THE HOUSE WILL KEEP TILL SPRING
    1919-142785-0032 ADD THE WINE AND IF NECESSARY A SEASONING OF CAYENNE WHEN IT WILL BE READY TO SERVE
    1919-142785-0033 NOTE THE WINE IN THIS SAUCE MAY BE OMITTED AND AN ONION SLICED AND FRIED OF A NICE BROWN SUBSTITUTED FOR IT
    1919-142785-0034 SIMMER FOR A MINUTE OR TWO AND SERVE IN A TUREEN
    1919-142785-0035 SUFFICIENT TO SERVE WITH FIVE OR SIX MACKEREL
    1919-142785-0036 VARIOUS DISHES ARE FREQUENTLY ORNAMENTED AND GARNISHED WITH ITS GRACEFUL LEAVES AND THESE ARE SOMETIMES BOILED IN SOUPS ALTHOUGH IT IS MORE USUALLY CONFINED IN ENGLISH COOKERY TO THE MACKEREL SAUCE AS HERE GIVEN
    1919-142785-0037 FORCEMEAT FOR COLD SAVOURY PIES
    1919-142785-0038 POUND WELL AND BIND WITH ONE OR TWO EGGS WHICH HAVE BEEN PREVIOUSLY BEATEN AND STRAINED
    1919-142785-0039 ILLUSTRATION MARJORAM
    1919-142785-0040 IT IS A NATIVE OF PORTUGAL AND WHEN ITS LEAVES ARE USED AS A SEASONING HERB THEY HAVE AN AGREEABLE AROMATIC FLAVOUR
    1919-142785-0041 MODE MIX ALL THE INGREDIENTS WELL TOGETHER CAREFULLY MINCING THEM VERY FINELY BEAT UP THE EGG MOISTEN WITH IT AND WORK THE WHOLE VERY SMOOTHLY TOGETHER
    1919-142785-0042 SUFFICIENT FOR A MODERATE SIZED HADDOCK OR PIKE
    1919-142785-0043 NOW BEAT AND STRAIN THE EGGS WORK THESE UP WITH THE OTHER INGREDIENTS AND THE FORCEMEAT WILL BE READY FOR USE
    1919-142785-0044 BOIL FOR FIVE MINUTES MINCE IT VERY SMALL AND MIX IT WITH THE OTHER INGREDIENTS
    1919-142785-0045 IF IT SHOULD BE IN AN UNSOUND STATE IT MUST BE ON NO ACCOUNT MADE USE OF
    1919-142785-0046 ILLUSTRATION BASIL
    1919-142785-0047 OTHER SWEET HERBS ARE CULTIVATED FOR PURPOSES OF MEDICINE AND PERFUMERY THEY ARE MOST GRATEFUL BOTH TO THE ORGANS OF TASTE AND SMELLING AND TO THE AROMA DERIVED FROM THEM IS DUE IN A GREAT MEASURE THE SWEET AND EXHILARATING FRAGRANCE OF OUR FLOWERY MEADS
    1919-142785-0048 FRENCH FORCEMEAT
    1919-142785-0049 IT WILL BE WELL TO STATE IN THE BEGINNING OF THIS RECIPE THAT FRENCH FORCEMEAT OR QUENELLES CONSIST OF THE BLENDING OF THREE SEPARATE PROCESSES NAMELY PANADA UDDER AND WHATEVER MEAT YOU INTEND USING PANADA
    1919-142785-0050 PLACE IT OVER THE FIRE KEEP CONSTANTLY STIRRING TO PREVENT ITS BURNING AND WHEN QUITE DRY PUT IN A SMALL PIECE OF BUTTER
    1919-142785-0051 PUT THE UDDER INTO A STEWPAN WITH SUFFICIENT WATER TO COVER IT LET IT STEW GENTLY TILL QUITE DONE WHEN TAKE IT OUT TO COOL
    1919-142785-0052 ILLUSTRATION PESTLE AND MORTAR
    1919-142785-0053 WHEN THE THREE INGREDIENTS ARE PROPERLY PREPARED POUND THEM ALTOGETHER IN A MORTAR FOR SOME TIME FOR THE MORE QUENELLES ARE POUNDED THE MORE DELICATE THEY ARE
    1919-142785-0054 IF THE QUENELLES ARE NOT FIRM ENOUGH ADD THE YOLK OF ANOTHER EGG BUT OMIT THE WHITE WHICH ONLY MAKES THEM HOLLOW AND PUFFY INSIDE
    1919-142785-0055 ANY ONE WITH THE SLIGHTEST PRETENSIONS TO REFINED COOKERY MUST IN THIS PARTICULAR IMPLICITLY FOLLOW THE EXAMPLE OF OUR FRIENDS ACROSS THE CHANNEL
    1919-142785-0056 FRIED BREAD CRUMBS
    1919-142785-0057 THE FAT THEY ARE FRIED IN SHOULD BE CLEAR AND THE CRUMBS SHOULD NOT HAVE THE SLIGHTEST APPEARANCE OR TASTE OF HAVING BEEN IN THE LEAST DEGREE BURNT
    1919-142785-0058 FRIED BREAD FOR BORDERS
    1919-142785-0059 WHEN QUITE CRISP DIP ONE SIDE OF THE SIPPET INTO THE BEATEN WHITE OF AN EGG MIXED WITH A LITTLE FLOUR AND PLACE IT ON THE EDGE OF THE DISH
    1919-142785-0060 CONTINUE IN THIS MANNER TILL THE BORDER IS COMPLETED ARRANGING THE SIPPETS A PALE AND A DARK ONE ALTERNATELY
    1919-142785-0061 MODE CUT UP THE ONION AND CARROT INTO SMALL RINGS AND PUT THEM INTO A STEWPAN WITH THE HERBS MUSHROOMS BAY LEAF CLOVES AND MACE ADD THE BUTTER AND SIMMER THE WHOLE VERY GENTLY OVER A SLOW FIRE UNTIL THE ONION IS QUITE TENDER
    1919-142785-0062 SUFFICIENT HALF THIS QUANTITY FOR TWO SLICES OF SALMON
    1919-142785-0063 ILLUSTRATION SAGE
    1988-147956-0000 FUCHS BROUGHT UP A SACK OF POTATOES AND A PIECE OF CURED PORK FROM THE CELLAR AND GRANDMOTHER PACKED SOME LOAVES OF SATURDAY'S BREAD A JAR OF BUTTER AND SEVERAL PUMPKIN PIES IN THE STRAW OF THE WAGON BOX
    1988-147956-0001 OCCASIONALLY ONE OF THE HORSES WOULD TEAR OFF WITH HIS TEETH A PLANT FULL OF BLOSSOMS AND WALK ALONG MUNCHING IT THE FLOWERS NODDING IN TIME TO HIS BITES AS HE ATE DOWN TOWARD THEM
    1988-147956-0002 IT'S NO BETTER THAN A BADGER HOLE NO PROPER DUGOUT AT ALL
    1988-147956-0003 NOW WHY IS THAT OTTO
    1988-147956-0004 PRESENTLY AGAINST ONE OF THOSE BANKS I SAW A SORT OF SHED THATCHED WITH THE SAME WINE COLORED GRASS THAT GREW EVERYWHERE
    1988-147956-0005 VERY GLAD VERY GLAD SHE EJACULATED
    1988-147956-0006 YOU'LL GET FIXED UP COMFORTABLE AFTER WHILE MISSUS SHIMERDA MAKE GOOD HOUSE
    1988-147956-0007 MY GRANDMOTHER ALWAYS SPOKE IN A VERY LOUD TONE TO FOREIGNERS AS IF THEY WERE DEAF
    1988-147956-0008 SHE MADE MISSUS SHIMERDA UNDERSTAND THE FRIENDLY INTENTION OF OUR VISIT AND THE BOHEMIAN WOMAN HANDLED THE LOAVES OF BREAD AND EVEN SMELLED THEM AND EXAMINED THE PIES WITH LIVELY CURIOSITY EXCLAIMING MUCH GOOD MUCH THANK
    1988-147956-0009 THE FAMILY HAD BEEN LIVING ON CORNCAKES AND SORGHUM MOLASSES FOR THREE DAYS
    1988-147956-0010 I REMEMBERED WHAT THE CONDUCTOR HAD SAID ABOUT HER EYES
    1988-147956-0011 HER SKIN WAS BROWN TOO AND IN HER CHEEKS SHE HAD A GLOW OF RICH DARK COLOR
    1988-147956-0012 EVEN FROM A DISTANCE ONE COULD SEE THAT THERE WAS SOMETHING STRANGE ABOUT THIS BOY
    1988-147956-0013 HE WAS BORN LIKE THAT THE OTHERS ARE SMART
    1988-147956-0014 AMBROSCH HE MAKE GOOD FARMER
    1988-147956-0015 HE STRUCK AMBROSCH ON THE BACK AND THE BOY SMILED KNOWINGLY
    1988-147956-0016 AT THAT MOMENT THE FATHER CAME OUT OF THE HOLE IN THE BANK
    1988-147956-0017 IT WAS SO LONG THAT IT BUSHED OUT BEHIND HIS EARS AND MADE HIM LOOK LIKE THE OLD PORTRAITS I REMEMBERED IN VIRGINIA
    1988-147956-0018 I NOTICED HOW WHITE AND WELL SHAPED HIS OWN HANDS WERE
    1988-147956-0019 WE STOOD PANTING ON THE EDGE OF THE RAVINE LOOKING DOWN AT THE TREES AND BUSHES THAT GREW BELOW US
    1988-147956-0020 THE WIND WAS SO STRONG THAT I HAD TO HOLD MY HAT ON AND THE GIRLS SKIRTS WERE BLOWN OUT BEFORE THEM
    1988-147956-0021 SHE LOOKED AT ME HER EYES FAIRLY BLAZING WITH THINGS SHE COULD NOT SAY
    1988-147956-0022 SHE POINTED INTO THE GOLD COTTONWOOD TREE BEHIND WHOSE TOP WE STOOD AND SAID AGAIN WHAT NAME
    1988-147956-0023 ANTONIA POINTED UP TO THE SKY AND QUESTIONED ME WITH HER GLANCE
    1988-147956-0024 SHE GOT UP ON HER KNEES AND WRUNG HER HANDS
    1988-147956-0025 SHE WAS QUICK AND VERY EAGER
    1988-147956-0026 WE WERE SO DEEP IN THE GRASS THAT WE COULD SEE NOTHING BUT THE BLUE SKY OVER US AND THE GOLD TREE IN FRONT OF US
    1988-147956-0027 AFTER ANTONIA HAD SAID THE NEW WORDS OVER AND OVER SHE WANTED TO GIVE ME A LITTLE CHASED SILVER RING SHE WORE ON HER MIDDLE FINGER
    1988-147956-0028 WHEN I CAME UP HE TOUCHED MY SHOULDER AND LOOKED SEARCHINGLY DOWN INTO MY FACE FOR SEVERAL SECONDS
    1988-147956-0029 I BECAME SOMEWHAT EMBARRASSED FOR I WAS USED TO BEING TAKEN FOR GRANTED BY MY ELDERS
    1988-148538-0000 IN ARISTOCRATIC COMMUNITIES THE PEOPLE READILY GIVE THEMSELVES UP TO BURSTS OF TUMULTUOUS AND BOISTEROUS GAYETY WHICH SHAKE OFF AT ONCE THE RECOLLECTION OF THEIR PRIVATIONS THE NATIVES OF DEMOCRACIES ARE NOT FOND OF BEING THUS VIOLENTLY BROKEN IN UPON AND THEY NEVER LOSE SIGHT OF THEIR OWN SELVES WITHOUT REGRET
    1988-148538-0001 AN AMERICAN INSTEAD OF GOING IN A LEISURE HOUR TO DANCE MERRILY AT SOME PLACE OF PUBLIC RESORT AS THE FELLOWS OF HIS CALLING CONTINUE TO DO THROUGHOUT THE GREATER PART OF EUROPE SHUTS HIMSELF UP AT HOME TO DRINK
    1988-148538-0002 I BELIEVE THE SERIOUSNESS OF THE AMERICANS ARISES PARTLY FROM THEIR PRIDE
    1988-148538-0003 THIS IS MORE ESPECIALLY THE CASE AMONGST THOSE FREE NATIONS WHICH FORM DEMOCRATIC COMMUNITIES
    1988-148538-0004 THEN THERE ARE IN ALL CLASSES A VERY LARGE NUMBER OF MEN CONSTANTLY OCCUPIED WITH THE SERIOUS AFFAIRS OF THE GOVERNMENT AND THOSE WHOSE THOUGHTS ARE NOT ENGAGED IN THE DIRECTION OF THE COMMONWEALTH ARE WHOLLY ENGROSSED BY THE ACQUISITION OF A PRIVATE FORTUNE
    1988-148538-0005 I DO NOT BELIEVE IN SUCH REPUBLICS ANY MORE THAN IN THAT OF PLATO OR IF THE THINGS WE READ OF REALLY HAPPENED I DO NOT HESITATE TO AFFIRM THAT THESE SUPPOSED DEMOCRACIES WERE COMPOSED OF VERY DIFFERENT ELEMENTS FROM OURS AND THAT THEY HAD NOTHING IN COMMON WITH THE LATTER EXCEPT THEIR NAME
    1988-148538-0006 IN ARISTOCRACIES EVERY MAN HAS ONE SOLE OBJECT WHICH HE UNCEASINGLY PURSUES BUT AMONGST DEMOCRATIC NATIONS THE EXISTENCE OF MAN IS MORE COMPLEX THE SAME MIND WILL ALMOST ALWAYS EMBRACE SEVERAL OBJECTS AT THE SAME TIME AND THESE OBJECTS ARE FREQUENTLY WHOLLY FOREIGN TO EACH OTHER AS IT CANNOT KNOW THEM ALL WELL THE MIND IS READILY SATISFIED WITH IMPERFECT NOTIONS OF EACH
    1988-148538-0007 CHAPTER SIXTEEN WHY THE NATIONAL VANITY OF THE AMERICANS IS MORE RESTLESS AND CAPTIOUS THAN THAT OF THE ENGLISH
    1988-148538-0008 THE AMERICANS IN THEIR INTERCOURSE WITH STRANGERS APPEAR IMPATIENT OF THE SMALLEST CENSURE AND INSATIABLE OF PRAISE
    1988-148538-0009 IF I SAY TO AN AMERICAN THAT THE COUNTRY HE LIVES IN IS A FINE ONE AY HE REPLIES THERE IS NOT ITS FELLOW IN THE WORLD
    1988-148538-0010 IF I APPLAUD THE FREEDOM WHICH ITS INHABITANTS ENJOY HE ANSWERS FREEDOM IS A FINE THING BUT FEW NATIONS ARE WORTHY TO ENJOY IT
    1988-148538-0011 IN ARISTOCRATIC COUNTRIES THE GREAT POSSESS IMMENSE PRIVILEGES UPON WHICH THEIR PRIDE RESTS WITHOUT SEEKING TO RELY UPON THE LESSER ADVANTAGES WHICH ACCRUE TO THEM
    1988-148538-0012 THEY THEREFORE ENTERTAIN A CALM SENSE OF THEIR SUPERIORITY THEY DO NOT DREAM OF VAUNTING PRIVILEGES WHICH EVERYONE PERCEIVES AND NO ONE CONTESTS AND THESE THINGS ARE NOT SUFFICIENTLY NEW TO THEM TO BE MADE TOPICS OF CONVERSATION
    1988-148538-0013 THEY STAND UNMOVED IN THEIR SOLITARY GREATNESS WELL ASSURED THAT THEY ARE SEEN OF ALL THE WORLD WITHOUT ANY EFFORT TO SHOW THEMSELVES OFF AND THAT NO ONE WILL ATTEMPT TO DRIVE THEM FROM THAT POSITION
    1988-148538-0014 WHEN AN ARISTOCRACY CARRIES ON THE PUBLIC AFFAIRS ITS NATIONAL PRIDE NATURALLY ASSUMES THIS RESERVED INDIFFERENT AND HAUGHTY FORM WHICH IS IMITATED BY ALL THE OTHER CLASSES OF THE NATION
    1988-148538-0015 THESE PERSONS THEN DISPLAYED TOWARDS EACH OTHER PRECISELY THE SAME PUERILE JEALOUSIES WHICH ANIMATE THE MEN OF DEMOCRACIES THE SAME EAGERNESS TO SNATCH THE SMALLEST ADVANTAGES WHICH THEIR EQUALS CONTESTED AND THE SAME DESIRE TO PARADE OSTENTATIOUSLY THOSE OF WHICH THEY WERE IN POSSESSION
    1988-24833-0000 THE TWO STRAY KITTENS GRADUALLY MAKE THEMSELVES AT HOME
    1988-24833-0001 SOMEHOW OR OTHER CAT HAS TAUGHT THEM THAT HE'S IN CHARGE HERE AND HE JUST CHASES THEM FOR FUN NOW AND AGAIN WHEN HE'S NOT BUSY SLEEPING
    1988-24833-0002 SHE DOESN'T PICK THEM UP BUT JUST HAVING THEM IN THE ROOM SURE DOESN'T GIVE HER ASTHMA
    1988-24833-0003 WHEN ARE YOU GETTING RID OF THESE CATS I'M NOT FIXING TO START AN ANNEX TO KATE'S CAT HOME
    1988-24833-0004 RIGHT AWAY WHEN I BRING HOME MY NEW PROGRAM HE SAYS HOW COME YOU'RE TAKING ONE LESS COURSE THIS HALF
    1988-24833-0005 I EXPLAIN THAT I'M TAKING MUSIC AND ALSO BIOLOGY ALGEBRA ENGLISH AND FRENCH MUSIC HE SNORTS
    1988-24833-0006 POP IT'S A COURSE
    1988-24833-0007 HE DOES AND FOR ONCE I WIN A ROUND I KEEP MUSIC FOR THIS SEMESTER
    1988-24833-0008 I'LL BE LUCKY IF I HAVE TIME TO BREATHE
    1988-24833-0009 SOMETIMES SCHOOLS DO LET KIDS TAKE A LOT OF SOFT COURSES AND THEN THEY'RE OUT ON A LIMB LATER HUH
    1988-24833-0010 SO HE CARES HUH
    1988-24833-0011 BESIDES SAYS TOM HALF THE REASON YOU AND YOUR FATHER ARE ALWAYS BICKERING IS THAT YOU'RE SO MUCH ALIKE ME LIKE HIM SURE
    1988-24833-0012 AS LONG AS THERE'S A BONE ON THE FLOOR THE TWO OF YOU WORRY IT
    1988-24833-0013 I GET THE PILLOWS COMFORTABLY ARRANGED ON THE FLOOR WITH A BIG BOTTLE OF SODA AND A BAG OF POPCORN WITHIN EASY REACH
    1988-24833-0014 POP GOES RIGHT ON TUNING HIS CHANNEL
    1988-24833-0015 YOU'RE GETTING ALTOGETHER TOO UPSET ABOUT THESE PROGRAMS STOP IT AND BEHAVE YOURSELF
    1988-24833-0016 IT'S YOUR FAULT MOP IT UP YOURSELF
    1988-24833-0017 I HEAR THE T V GOING FOR A FEW MINUTES THEN POP TURNS IT OFF AND GOES IN THE KITCHEN TO TALK TO MOM
    1988-24833-0018 WELL I DON'T THINK YOU SHOULD TURN A GUY'S T V PROGRAM OFF IN THE MIDDLE WITHOUT EVEN FINDING OUT ABOUT IT
    1988-24833-0019 I LOOK AT MY WATCH IT'S A QUARTER TO ELEVEN
    1988-24833-0020 I TURN OFF THE TELEVISION SET I'VE LOST TRACK OF WHAT'S HAPPENING AND IT DOESN'T SEEM TO BE THE GRANDFATHER WHO'S THE SPOOK AFTER ALL
    1988-24833-0021 IT'S THE FIRST TIME HILDA HAS BEEN TO OUR HOUSE AND TOM INTRODUCES HER AROUND
    1988-24833-0022 I TOLD TOM WE SHOULDN'T COME SO LATE SAYS HILDA
    1988-24833-0023 TOM SAYS THANKS AND LOOKS AT HILDA AND SHE BLUSHES REALLY
    1988-24833-0024 TOM DRINKS A LITTLE MORE COFFEE AND THEN HE GOES ON THE TROUBLE IS I CAN'T GET MARRIED ON THIS FLOWER SHOP JOB
    1988-24833-0025 YOU KNOW I'D GET DRAFTED IN A YEAR OR TWO ANYWAY
    1988-24833-0026 I'VE DECIDED TO ENLIST IN THE ARMY
    1988-24833-0027 I'LL HAVE TO CHECK SOME MORE SAYS TOM
    1988-24833-0028 HERE'S TO YOU A LONG HAPPY LIFE
    
    


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
    ('gutenbergtm', 'electronic')|306
    ('old', 'man')|306
    ('mr', 'bounderby')|294
    ('public', 'domain')|293
    ('every', 'one')|291
    ('young', 'man')|284
    ('mrs', 'sparsit')|282
    ('one', 'day')|281
    ('one', 'another')|280
    ('gutenberg', 'literary')|279
    ('literary', 'archive')|279
    ('archive', 'foundation')|279
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
    ('of', 'the')|198
    ('said', 'mr')|198
    ('first', 'time')|196
    ('one', 'thing')|193
    ('every', 'day')|193
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
     dict_keys(['sovereign', 'therefore', 'proportion', 'inconveniency', 'vitality', 'tenant', 'sin', 'facility', 'alterations', 'body', 'levying', 'gold', 'sorrow', 'rights', 'producing', 'distress', 'leader', 'expression', 'clerk', 'mind', 'confidence', 'money', 'advantage', 'its', 'thirst', 'variety', 'wisdom', 'like', 'rapidity', 'seems', 'foregoing', 'personal', 'difficulty', 'steps', 'numbers', 'favours', 'goodness', 'annoyance', 'thoughts', 'produce', 'relevance', 'shall', 'grief', 'importation', 'kindness', 'gehenna', 'acorn', 'come', 'circumstances', 'renown', 'superabundance', 'desire', 'distance', 'extent', 'done', 'whole', 'supply', 'believed', 'parsimony', 'pride', 'corn', 'intelligence', 'need', 'strides', 'yet', 'business', 'solidarity', 'original', 'every', 'first', 'dangers', 'point', 'mountains', 'pieces', 'gain', 'returns', 'exportation', 'never', 'fury', 'force', 'competition', 'hope', 'economy', 'inferiority', 'worldly', 'pasture', 'ruritania', 'service', 'want', 'costage', 'chance', 'quantities', 'action', 'things', 'age', 'care', 'general', 'life', 'lesser', 'time', 'past', 'content', 'contemporary', 'speed', 'attractions', 'place', 'imprudence', 'rapture', 'among', 'guilds', 'fault', 'sanctity', 'circulation', 'balance', 'distinctness', 'success', 'extensive', 'necessary', 'annual', 'brightness', 'made', 'quantity', 'waxen', 'stock', 'divinity', 'desolation', 'rent', 'great', 'lawe', 'mass', 'use', 'depth', 'woe', 'advantages', 'range', 'such', 'peril', 'change', 'he', 'influence', 'difference', 'anyone', 'length', 'cheapness', 'but', 'tartness', 'america', 'triumph', 'rum', 'fund', 'actually', 'contained', 'confusion', 'wonder', 'worlds', 'disorders', 'africa', 'formerly', 'usual', 'authority', 'ever', 'went', 'return', 'group', 'taint', 'continuing', 'pain', 'honour', 'depths', 'found', 'effect', 'melodies', 'greater', 'difficulties', 'goods', 'glory', 'dole', 'in', 'stocks', 'injustice', 'professed', 'still', 'than', 'sadness', 'rich', 'ii', 'brewery', 'as', 'harm', 'number', 'present', 'fire', 'lights', 'either', 'pomp', 'second', 'warmth', 'rank', 'fortitude', 'perhaps', 'freedom', 'degree', 'glorious', 'taxes', 'cost', 'honourable', 'sometimes', 'wiser', 'splendour', 'frequent', 'velocity', 'gladness', 'countries', 'silence', 'sums', 'boldness', 'thing', 'gifts', 'rice', 'amount', 'mans', 'real', 'individual', 'prospective', 'fame', 'impersonal', 'gift', 'could', 'land', 'account', 'well', 'strain', 'singleness', 'suited', 'grace', 'variation', 'weal', 'left', 'though', 'opportunities', 'capital', 'moment', 'common', 'frequency', 'offence', 'shame', 'it', 'at', 'without', 'expense', 'security', 'admiration', 'transgression', 'events', 'almost', 'height', 'vessels', 'slumbers', 'trade', 'demand', 'perpendicular', 'fast', 'honor', 'name', 'slaves', 'field', 'power', 'beauty', 'jefferies', 'salaries', 'consequence', 'portion', 'otherwise', 'latitude', 'english', 'might', 'value', 'opening', 'higher', 'ones', 'view', 'scarcity', 'told', 'abroad', 'abundance', 'surplus', 'haytime', 'price', 'universal', 'practicality', 'indeed', 'heat', 'enthusiasm', 'energy', 'incorporation', 'crop', 'london', 'fixed', 'crime', 'latter', 'indignation', 'profusion', 'sufficient', 'whatever', 'pleasure', 'environment', 'comfort', 'strength', 'degrees', 'windbag', 'capacities', 'zeal', 'heartiness', 'mystery', 'smaller', 'dignity', 'vanquished', 'violence', 'grew', 'if', 'share', 'the', 'subconscious', 'none', 'riches', 'wealth', 'love', 'enduring', 'dilatation', 'far', 'importance', 'must', 'reprobate', 'trespass', 'and', 'teacher', 'seignorage', 'titan', 'death', 'perithous', 'calamity', 'require', 'labourers', 'remoteness', 'ease', 'sun', 'writings', 'revenue', 'danger', 'men', 'already', 'diligence', 'many', 'restoration', 'parts', 'favour', 'horn', 'convenience', 'fiercer', 'prince', 'insult', 'flourish', 'interest', 'dexterity', 'equal', 'reduction', 'ordinary', 'sum', 'mastery', 'less', 'satisfaction', 'one', 'simplicity', 'part', 'cheap', 'weight', 'pressure', 'upon', 'approximation', 'semblance', 'wrong', 'poets', 'haste', 'would', 'dawn', 'injudicious', 'sensation', 'former', 'no', 'little', 'activity', 'tax', 'cultivation', 'crown', 'frequently', 'understanding', 'to', 'employs', 'peace', 'evils', 'france', 'valuable', 'loss', 'togetherness', 'modern', 'evil', 'claim', 'delectation', 'malversation', 'obstacles', 'lasting', 'eloquence', 'saving', 'effort', 'antipathy', 'profit', 'end', 'liberty', 'ned', 'encountering', 'nails', 'talents', 'expected', 'agony', 'art', 'knave', 'beginning', 'herein', 'clearness', 'fortune', 'favours\x94', 'rise', 'deviation', 'fear', 'augmentation', 'benefit'])
    
    Listing 20 most frequent words to come after 'greater':
     [('part', 532), ('quantity', 105), ('number', 50), ('proportion', 43), ('value', 24), ('greater', 16), ('smaller', 16), ('share', 16), ('less', 12), ('profit', 11), ('capital', 9), ('the', 9), ('importance', 9), ('revenue', 9), ('surplus', 8), ('variety', 7), ('distance', 7), ('degree', 7), ('stock', 6), ('length', 6)]



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
     {"but I'm less than 5 minutes the staircase groaned when he's an extraordinary way.": 0.80222582407295706, "but in less than five minutes the staircase groaned when he's an extraordinary wait": 0.78471516072750092, "but in less than 5 minutes the staircase groaned when he's an extraordinary wait": 0.82827072050422434, "but I'm less than 5 minutes the staircase groaned when he's an extraordinary wait": 0.81634781546890733, "but I'm less than 5 minutes the staircase groaned when he's an extraordinary weight": 0.81634781546890733, "but in less than 5 minutes the staircase groaned when he's an extraordinary way": 0.82827072050422434, "but in less than 5 minutes the staircase groaned when he's an extraordinary way.": 0.8133681526407599, "but I'm less than 5 minutes the staircase groaned when he's an extraordinary way": 0.81634781546890733, "but in less than five minutes the staircase groaned when he's an extraordinary weight": 0.78471516072750092, "but in less than 5 minutes the staircase groaned when he's an extraordinary weight": 0.80774457957595591}
    
    
    ORIGINAL Transcript: 
    'but I'm less than 5 minutes the staircase groaned when he's an extraordinary way.' 
    with a confidence_score of: 0.8898342847824097
    
    
    RE-RANKED Transcript: 
    'but in less than 5 minutes the staircase groaned when he's an extraordinary way' 
    with a confidence_score of: 0.8282707205042243
    
    
    GROUND TRUTH TRANSCRIPT: 
    BUT IN LESS THAN FIVE MINUTES THE STAIRCASE GROANED BENEATH AN EXTRAORDINARY WEIGHT
    
    The original transcript was RE-RANKED. The transcripts do not match!
    Differences between original and re-ranked:  ["'m", 'i', '.']
    
    
    The original transcript DOES NOT MATCH ground truth.
    Differences between original and ground truth:  ['i', 'when', '5', 'he', "'s", "'m", 'way', '.']
    
    
    The RE_RANKED transcript DOES NOT MATCH ground truth.
    Differences between Reranked and ground truth:  ['5', "'s", 'way', 'he', 'when']
    
    
    
    
    ORIGINAL Edit Distance: 
    18
    RE-RANKED Edit Distance: 
    16
    
    
    Waiting for operation to complete...
    
    
    RE-RANKED Results: 
     {'at this moment of the whole soul of the old man seem centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.87969393432140353, 'at this moment of the whole soul of the old man seemed centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.87348153144121177, 'at this moment of the whole soul of the Old Man scene centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.86476921737194068, 'at this moment the whole soul of the Old Man scene centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.8672342523932457, 'at this moment of the whole soul of the old man seems centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.88766976147890087, 'at this moment the whole soul of the old man seems centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.89042613655328751, 'at this moment the whole soul of the old man seem centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.8822889491915703, 'at this moment to the whole soul of the old man seem centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.88070062100887292, 'at this moment the whole soul of the old man seemed centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry': 0.87594656497240075, 'at this moment the whole soul of the old man seem centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this put the utterance of a cry': 0.88256364762783057}
    
    
    ORIGINAL Transcript: 
    'at this moment of the whole soul of the old man seems centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry' 
    with a confidence_score of: 0.9613686203956604
    
    
    RE-RANKED Transcript: 
    'at this moment the whole soul of the old man seems centered in his eyes which became bloodshot the veins of the throat swelled his cheeks and temples became purple as though he was struck with epilepsy nothing was wanting to complete this but the utterance of a cry' 
    with a confidence_score of: 0.8904261365532875
    
    
    GROUND TRUTH TRANSCRIPT: 
    AT THIS MOMENT THE WHOLE SOUL OF THE OLD MAN SEEMED CENTRED IN HIS EYES WHICH BECAME BLOODSHOT THE VEINS OF THE THROAT SWELLED HIS CHEEKS AND TEMPLES BECAME PURPLE AS THOUGH HE WAS STRUCK WITH EPILEPSY NOTHING WAS WANTING TO COMPLETE THIS BUT THE UTTERANCE OF A CRY
    
    No reranking was performed. The transcripts match!
    
    
    The original transcript DOES NOT MATCH ground truth.
    Differences between original and ground truth:  ['seems', 'centered']
    
    
    The RE_RANKED transcript DOES NOT MATCH ground truth.
    Differences between Reranked and ground truth:  ['seems', 'centered']
    
    
    
    
    ORIGINAL Edit Distance: 
    6
    RE-RANKED Edit Distance: 
    3
    
    
    Waiting for operation to complete...
    
    
    RE-RANKED Results: 
     {'Devin he rushed towards the old man and made him in Halo powerful restorative': 0.67532151639461524, 'deveney rushed towards the old man and made him in Halo powerful restorative': 0.70675845444202434, 'deveny Rush towards the old man and made him inhaler powerful restorative': 0.67607779577374461, 'deveney rushed towards the old man and made him and Haley powerful restorative': 0.76639919579029081, 'Devin he rushed towards the old man and made him inhaler powerful restorative': 0.72984181940555581, 'deveney rushed towards the old man and made him and Halo powerful restorative': 0.74139880836009986, 'deveny rushed towards the old man and made him in Halo powerful restorative': 0.68296468555927281, 'deveney rushed towards the old man and made him inhaler powerful restorative': 0.76849636733531956, 'Devon he rushed towards the old man and made him inhaler powerful restorative': 0.73141573965549478, 'deveny rushed towards the old man and made him inhaler powerful restorative': 0.74271974861621859}
    
    
    ORIGINAL Transcript: 
    'Devin he rushed towards the old man and made him in Halo powerful restorative' 
    with a confidence_score of: 0.7367948293685913
    
    
    RE-RANKED Transcript: 
    'deveney rushed towards the old man and made him inhaler powerful restorative' 
    with a confidence_score of: 0.7684963673353196
    
    
    GROUND TRUTH TRANSCRIPT: 
    D'AVRIGNY RUSHED TOWARDS THE OLD MAN AND MADE HIM INHALE A POWERFUL RESTORATIVE
    
    The original transcript was RE-RANKED. The transcripts do not match!
    Differences between original and re-ranked:  ['in', 'devin', 'halo', 'he']
    
    
    The original transcript DOES NOT MATCH ground truth.
    Differences between original and ground truth:  ['in', 'devin', 'halo', 'he']
    
    
    The RE_RANKED transcript DOES NOT MATCH ground truth.
    Differences between Reranked and ground truth:  ['deveney', 'inhaler']
    
    
    
    
    ORIGINAL Edit Distance: 
    11
    RE-RANKED Edit Distance: 
    8
    
    
    Waiting for operation to complete...
    
    
    RE-RANKED Results: 
     {'and the cry issued from his pores if we made the speak a cry frightful and it silence': 0.75123548712581401, 'and the cry issued from his pores if we made the speak a cry frightful in its silence': 0.82769908700138328, 'and the cry issued from his pores if we made the speak a cry frightful and its silence': 0.78830538596957933, 'and the cry issued from his pores if we may the speak a cry frightful and it silence': 0.73610656317323453, 'and the cry issued from his pores if we made the speak a cry frightful in it silence': 0.79064654335379603, 'and the cry issued from his pores if we made us speak a cry frightful in its silence': 0.82640005201101308, 'and the cry issued from his pores if we may the speak a cry frightful in it silence': 0.77551761940121655, 'and the cry issued from his pores if we may the speak a cry frightful in its silence': 0.81628829501569278, 'and the cry issued from his pores if we may the speak a cry frightful and its silence': 0.77317651566118006, 'and the cry issued from his pores if we may the speak I cry frightful in its silence': 0.81369256041944027}
    
    
    ORIGINAL Transcript: 
    'and the cry issued from his pores if we may the speak I cry frightful in its silence' 
    with a confidence_score of: 0.9014739990234375
    
    
    RE-RANKED Transcript: 
    'and the cry issued from his pores if we made the speak a cry frightful in its silence' 
    with a confidence_score of: 0.8276990870013833
    
    
    GROUND TRUTH TRANSCRIPT: 
    AND THE CRY ISSUED FROM HIS PORES IF WE MAY THUS SPEAK A CRY FRIGHTFUL IN ITS SILENCE
    
    The original transcript was RE-RANKED. The transcripts do not match!
    Differences between original and re-ranked:  ['i', 'may']
    
    
    The original transcript DOES NOT MATCH ground truth.
    Differences between original and ground truth:  ['i']
    
    
    The RE_RANKED transcript DOES NOT MATCH ground truth.
    Differences between Reranked and ground truth:  ['made']
    
    
    
    
    ORIGINAL Edit Distance: 
    3
    RE-RANKED Edit Distance: 
    4
    
    
    Waiting for operation to complete...
    
    
    RE-RANKED Results: 
     {'goat do you here': 0.85191820678301156, 'I go do you here': 0.82690102872438731, 'goat do you hear': 0.8522125408519059, 'do you here': 0.75063112792558973, 'go to you here': 0.84135989267379052, 'go do you hear': 0.78242248152382665, 'I go do you hear': 0.82719536279328165, 'go do you here': 0.85950413760729139, 'go do U hear': 0.84622236298156761, 'do you hear': 0.75092546199448407}
    
    
    ORIGINAL Transcript: 
    'goat do you here' 
    with a confidence_score of: 0.9461572766304016
    
    
    RE-RANKED Transcript: 
    'go do you here' 
    with a confidence_score of: 0.8595041376072914
    
    
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

    Epoch   0 Batch  500/2400 - Train Accuracy: 0.6104, Validation Accuracy: 0.6339, Loss: 3.1317
    Epoch   0 Batch 1000/2400 - Train Accuracy: 0.6228, Validation Accuracy: 0.6339, Loss: 2.5567
    Epoch   0 Batch 1500/2400 - Train Accuracy: 0.5750, Validation Accuracy: 0.6339, Loss: 2.9485
    Epoch   0 Batch 2000/2400 - Train Accuracy: 0.5288, Validation Accuracy: 0.6339, Loss: 3.0546
    Epoch   1 Batch  500/2400 - Train Accuracy: 0.6229, Validation Accuracy: 0.6339, Loss: 2.4715
    Epoch   1 Batch 1000/2400 - Train Accuracy: 0.6384, Validation Accuracy: 0.6339, Loss: 1.9880
    Epoch   1 Batch 1500/2400 - Train Accuracy: 0.6083, Validation Accuracy: 0.6339, Loss: 2.4104
    Epoch   1 Batch 2000/2400 - Train Accuracy: 0.5938, Validation Accuracy: 0.6339, Loss: 2.4717
    Epoch   2 Batch  500/2400 - Train Accuracy: 0.6333, Validation Accuracy: 0.6339, Loss: 2.1433
    Epoch   2 Batch 1000/2400 - Train Accuracy: 0.6741, Validation Accuracy: 0.6339, Loss: 1.6420
    Epoch   2 Batch 1500/2400 - Train Accuracy: 0.6333, Validation Accuracy: 0.6339, Loss: 2.0805
    Epoch   2 Batch 2000/2400 - Train Accuracy: 0.6154, Validation Accuracy: 0.6339, Loss: 2.1243
    Epoch   3 Batch  500/2400 - Train Accuracy: 0.6646, Validation Accuracy: 0.6339, Loss: 1.9158
    Epoch   3 Batch 1000/2400 - Train Accuracy: 0.7098, Validation Accuracy: 0.6339, Loss: 1.4255
    Epoch   3 Batch 1500/2400 - Train Accuracy: 0.6542, Validation Accuracy: 0.6339, Loss: 1.8819
    Epoch   3 Batch 2000/2400 - Train Accuracy: 0.6514, Validation Accuracy: 0.6339, Loss: 1.9255
    Epoch   4 Batch  500/2400 - Train Accuracy: 0.6896, Validation Accuracy: 0.6339, Loss: 1.7041
    Epoch   4 Batch 1000/2400 - Train Accuracy: 0.7388, Validation Accuracy: 0.6339, Loss: 1.2449
    Epoch   4 Batch 1500/2400 - Train Accuracy: 0.6792, Validation Accuracy: 0.6339, Loss: 1.7161
    Epoch   4 Batch 2000/2400 - Train Accuracy: 0.6562, Validation Accuracy: 0.6339, Loss: 1.7167
    Epoch   5 Batch  500/2400 - Train Accuracy: 0.7021, Validation Accuracy: 0.6339, Loss: 1.5465
    Epoch   5 Batch 1000/2400 - Train Accuracy: 0.7478, Validation Accuracy: 0.6339, Loss: 1.0918
    Epoch   5 Batch 1500/2400 - Train Accuracy: 0.6729, Validation Accuracy: 0.6339, Loss: 1.5846
    Epoch   5 Batch 2000/2400 - Train Accuracy: 0.6755, Validation Accuracy: 0.6339, Loss: 1.5621
    Epoch   6 Batch  500/2400 - Train Accuracy: 0.7063, Validation Accuracy: 0.6339, Loss: 1.3880
    Epoch   6 Batch 1000/2400 - Train Accuracy: 0.7790, Validation Accuracy: 0.6339, Loss: 0.9889
    Epoch   6 Batch 1500/2400 - Train Accuracy: 0.6896, Validation Accuracy: 0.6339, Loss: 1.4396
    Epoch   6 Batch 2000/2400 - Train Accuracy: 0.7019, Validation Accuracy: 0.6339, Loss: 1.4300
    Epoch   7 Batch  500/2400 - Train Accuracy: 0.7167, Validation Accuracy: 0.6339, Loss: 1.2615
    Epoch   7 Batch 1000/2400 - Train Accuracy: 0.7902, Validation Accuracy: 0.6339, Loss: 0.9046
    Epoch   7 Batch 1500/2400 - Train Accuracy: 0.7063, Validation Accuracy: 0.6339, Loss: 1.3476
    Epoch   7 Batch 2000/2400 - Train Accuracy: 0.6851, Validation Accuracy: 0.6339, Loss: 1.2886
    Epoch   8 Batch  500/2400 - Train Accuracy: 0.7271, Validation Accuracy: 0.6339, Loss: 1.1384
    Epoch   8 Batch 1000/2400 - Train Accuracy: 0.8147, Validation Accuracy: 0.6339, Loss: 0.8100
    Epoch   8 Batch 1500/2400 - Train Accuracy: 0.7188, Validation Accuracy: 0.6339, Loss: 1.2236
    Epoch   8 Batch 2000/2400 - Train Accuracy: 0.7452, Validation Accuracy: 0.6339, Loss: 1.1284
    Epoch   9 Batch  500/2400 - Train Accuracy: 0.7604, Validation Accuracy: 0.6339, Loss: 1.0112
    Epoch   9 Batch 1000/2400 - Train Accuracy: 0.8259, Validation Accuracy: 0.6339, Loss: 0.7391
    Epoch   9 Batch 1500/2400 - Train Accuracy: 0.7125, Validation Accuracy: 0.6339, Loss: 1.1240
    Epoch   9 Batch 2000/2400 - Train Accuracy: 0.7548, Validation Accuracy: 0.6339, Loss: 1.0348
    Epoch  10 Batch  500/2400 - Train Accuracy: 0.7854, Validation Accuracy: 0.6339, Loss: 0.9253
    Epoch  10 Batch 1000/2400 - Train Accuracy: 0.8371, Validation Accuracy: 0.6339, Loss: 0.6697
    Epoch  10 Batch 1500/2400 - Train Accuracy: 0.7229, Validation Accuracy: 0.6339, Loss: 1.0596
    Epoch  10 Batch 2000/2400 - Train Accuracy: 0.7692, Validation Accuracy: 0.6339, Loss: 0.9546
    Epoch  11 Batch  500/2400 - Train Accuracy: 0.8146, Validation Accuracy: 0.6339, Loss: 0.8065
    Epoch  11 Batch 1000/2400 - Train Accuracy: 0.8326, Validation Accuracy: 0.6339, Loss: 0.6378
    Epoch  11 Batch 1500/2400 - Train Accuracy: 0.7521, Validation Accuracy: 0.6339, Loss: 0.9820
    Epoch  11 Batch 2000/2400 - Train Accuracy: 0.7981, Validation Accuracy: 0.6339, Loss: 0.8791
    Epoch  12 Batch  500/2400 - Train Accuracy: 0.8146, Validation Accuracy: 0.6339, Loss: 0.7376
    Epoch  12 Batch 1000/2400 - Train Accuracy: 0.8415, Validation Accuracy: 0.6362, Loss: 0.5700
    Epoch  12 Batch 1500/2400 - Train Accuracy: 0.7625, Validation Accuracy: 0.6362, Loss: 0.8795
    Epoch  12 Batch 2000/2400 - Train Accuracy: 0.8005, Validation Accuracy: 0.6339, Loss: 0.7803
    Epoch  13 Batch  500/2400 - Train Accuracy: 0.8250, Validation Accuracy: 0.6362, Loss: 0.6813
    Epoch  13 Batch 1000/2400 - Train Accuracy: 0.8571, Validation Accuracy: 0.6362, Loss: 0.5575
    Epoch  13 Batch 1500/2400 - Train Accuracy: 0.7625, Validation Accuracy: 0.6384, Loss: 0.8212
    Epoch  13 Batch 2000/2400 - Train Accuracy: 0.8005, Validation Accuracy: 0.6384, Loss: 0.7248
    Epoch  14 Batch  500/2400 - Train Accuracy: 0.8438, Validation Accuracy: 0.6362, Loss: 0.6037
    Epoch  14 Batch 1000/2400 - Train Accuracy: 0.8549, Validation Accuracy: 0.6362, Loss: 0.5107
    Epoch  14 Batch 1500/2400 - Train Accuracy: 0.7833, Validation Accuracy: 0.6384, Loss: 0.7766
    Epoch  14 Batch 2000/2400 - Train Accuracy: 0.8005, Validation Accuracy: 0.6384, Loss: 0.6752
    Epoch  15 Batch  500/2400 - Train Accuracy: 0.8292, Validation Accuracy: 0.6384, Loss: 0.5610
    Epoch  15 Batch 1000/2400 - Train Accuracy: 0.8594, Validation Accuracy: 0.6384, Loss: 0.4734
    Epoch  15 Batch 1500/2400 - Train Accuracy: 0.7979, Validation Accuracy: 0.6384, Loss: 0.7091
    Epoch  15 Batch 2000/2400 - Train Accuracy: 0.8053, Validation Accuracy: 0.6384, Loss: 0.6130
    Epoch  16 Batch  500/2400 - Train Accuracy: 0.8438, Validation Accuracy: 0.6384, Loss: 0.5184
    Epoch  16 Batch 1000/2400 - Train Accuracy: 0.8616, Validation Accuracy: 0.6384, Loss: 0.4571
    Epoch  16 Batch 1500/2400 - Train Accuracy: 0.7729, Validation Accuracy: 0.6384, Loss: 0.6398
    Epoch  16 Batch 2000/2400 - Train Accuracy: 0.8341, Validation Accuracy: 0.6384, Loss: 0.5870
    Epoch  17 Batch  500/2400 - Train Accuracy: 0.8625, Validation Accuracy: 0.6384, Loss: 0.4516
    Epoch  17 Batch 1000/2400 - Train Accuracy: 0.8571, Validation Accuracy: 0.6384, Loss: 0.4784
    Epoch  17 Batch 1500/2400 - Train Accuracy: 0.7896, Validation Accuracy: 0.6384, Loss: 0.5962
    Epoch  17 Batch 2000/2400 - Train Accuracy: 0.8510, Validation Accuracy: 0.6384, Loss: 0.5335
    Epoch  18 Batch  500/2400 - Train Accuracy: 0.8625, Validation Accuracy: 0.6384, Loss: 0.4342
    Epoch  18 Batch 1000/2400 - Train Accuracy: 0.8728, Validation Accuracy: 0.6384, Loss: 0.4458
    Epoch  18 Batch 1500/2400 - Train Accuracy: 0.7979, Validation Accuracy: 0.6384, Loss: 0.5735
    Epoch  18 Batch 2000/2400 - Train Accuracy: 0.8438, Validation Accuracy: 0.6384, Loss: 0.5050
    Epoch  19 Batch  500/2400 - Train Accuracy: 0.8792, Validation Accuracy: 0.6384, Loss: 0.4221
    Epoch  19 Batch 1000/2400 - Train Accuracy: 0.8772, Validation Accuracy: 0.6384, Loss: 0.4116
    Epoch  19 Batch 1500/2400 - Train Accuracy: 0.7896, Validation Accuracy: 0.6384, Loss: 0.5449
    Epoch  19 Batch 2000/2400 - Train Accuracy: 0.8510, Validation Accuracy: 0.6362, Loss: 0.4441
    Epoch  20 Batch  500/2400 - Train Accuracy: 0.8708, Validation Accuracy: 0.6384, Loss: 0.3794
    Epoch  20 Batch 1000/2400 - Train Accuracy: 0.8683, Validation Accuracy: 0.6362, Loss: 0.3766
    Epoch  20 Batch 1500/2400 - Train Accuracy: 0.8021, Validation Accuracy: 0.6362, Loss: 0.5020
    Epoch  20 Batch 2000/2400 - Train Accuracy: 0.8293, Validation Accuracy: 0.6362, Loss: 0.4445
    Epoch  21 Batch  500/2400 - Train Accuracy: 0.9021, Validation Accuracy: 0.6384, Loss: 0.3620
    Epoch  21 Batch 1000/2400 - Train Accuracy: 0.8750, Validation Accuracy: 0.6362, Loss: 0.3491
    Epoch  21 Batch 1500/2400 - Train Accuracy: 0.8063, Validation Accuracy: 0.6384, Loss: 0.4737
    Epoch  21 Batch 2000/2400 - Train Accuracy: 0.8678, Validation Accuracy: 0.6384, Loss: 0.3935
    Epoch  22 Batch  500/2400 - Train Accuracy: 0.9021, Validation Accuracy: 0.6384, Loss: 0.3227
    Epoch  22 Batch 1000/2400 - Train Accuracy: 0.8750, Validation Accuracy: 0.6362, Loss: 0.3521
    Epoch  22 Batch 1500/2400 - Train Accuracy: 0.8229, Validation Accuracy: 0.6362, Loss: 0.4271
    Epoch  22 Batch 2000/2400 - Train Accuracy: 0.8798, Validation Accuracy: 0.6362, Loss: 0.3700
    Epoch  23 Batch  500/2400 - Train Accuracy: 0.9083, Validation Accuracy: 0.6384, Loss: 0.3031
    Epoch  23 Batch 1000/2400 - Train Accuracy: 0.8862, Validation Accuracy: 0.6384, Loss: 0.3177
    Epoch  23 Batch 1500/2400 - Train Accuracy: 0.8271, Validation Accuracy: 0.6384, Loss: 0.4105
    Epoch  23 Batch 2000/2400 - Train Accuracy: 0.8389, Validation Accuracy: 0.6362, Loss: 0.3436
    Epoch  24 Batch  500/2400 - Train Accuracy: 0.9250, Validation Accuracy: 0.6362, Loss: 0.2825
    Epoch  24 Batch 1000/2400 - Train Accuracy: 0.8795, Validation Accuracy: 0.6362, Loss: 0.3156
    Epoch  24 Batch 1500/2400 - Train Accuracy: 0.8729, Validation Accuracy: 0.6384, Loss: 0.3738
    Epoch  24 Batch 2000/2400 - Train Accuracy: 0.8798, Validation Accuracy: 0.6362, Loss: 0.3440
    Epoch  25 Batch  500/2400 - Train Accuracy: 0.9021, Validation Accuracy: 0.6384, Loss: 0.2614
    Epoch  25 Batch 1000/2400 - Train Accuracy: 0.9018, Validation Accuracy: 0.6362, Loss: 0.2896
    Epoch  25 Batch 1500/2400 - Train Accuracy: 0.8521, Validation Accuracy: 0.6406, Loss: 0.4006
    Epoch  25 Batch 2000/2400 - Train Accuracy: 0.8918, Validation Accuracy: 0.6362, Loss: 0.3105
    Epoch  26 Batch  500/2400 - Train Accuracy: 0.9125, Validation Accuracy: 0.6384, Loss: 0.2411
    Epoch  26 Batch 1000/2400 - Train Accuracy: 0.8862, Validation Accuracy: 0.6362, Loss: 0.2981
    Epoch  26 Batch 1500/2400 - Train Accuracy: 0.8896, Validation Accuracy: 0.6384, Loss: 0.3628
    Epoch  26 Batch 2000/2400 - Train Accuracy: 0.9135, Validation Accuracy: 0.6362, Loss: 0.2937
    Epoch  27 Batch  500/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6384, Loss: 0.2200
    Epoch  27 Batch 1000/2400 - Train Accuracy: 0.9085, Validation Accuracy: 0.6384, Loss: 0.2803
    Epoch  27 Batch 1500/2400 - Train Accuracy: 0.8750, Validation Accuracy: 0.6384, Loss: 0.3574
    Epoch  27 Batch 2000/2400 - Train Accuracy: 0.9087, Validation Accuracy: 0.6384, Loss: 0.2813
    Epoch  28 Batch  500/2400 - Train Accuracy: 0.9167, Validation Accuracy: 0.6362, Loss: 0.1985
    Epoch  28 Batch 1000/2400 - Train Accuracy: 0.8862, Validation Accuracy: 0.6384, Loss: 0.2913
    Epoch  28 Batch 1500/2400 - Train Accuracy: 0.8854, Validation Accuracy: 0.6384, Loss: 0.3258
    Epoch  28 Batch 2000/2400 - Train Accuracy: 0.9255, Validation Accuracy: 0.6362, Loss: 0.2379
    Epoch  29 Batch  500/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6384, Loss: 0.1943
    Epoch  29 Batch 1000/2400 - Train Accuracy: 0.9308, Validation Accuracy: 0.6362, Loss: 0.2413
    Epoch  29 Batch 1500/2400 - Train Accuracy: 0.9042, Validation Accuracy: 0.6362, Loss: 0.2785
    Epoch  29 Batch 2000/2400 - Train Accuracy: 0.9351, Validation Accuracy: 0.6362, Loss: 0.2479
    Epoch  30 Batch  500/2400 - Train Accuracy: 0.9187, Validation Accuracy: 0.6362, Loss: 0.2043
    Epoch  30 Batch 1000/2400 - Train Accuracy: 0.9308, Validation Accuracy: 0.6362, Loss: 0.2523
    Epoch  30 Batch 1500/2400 - Train Accuracy: 0.8688, Validation Accuracy: 0.6362, Loss: 0.3151
    Epoch  30 Batch 2000/2400 - Train Accuracy: 0.9062, Validation Accuracy: 0.6362, Loss: 0.2243
    Epoch  31 Batch  500/2400 - Train Accuracy: 0.9604, Validation Accuracy: 0.6362, Loss: 0.1881
    Epoch  31 Batch 1000/2400 - Train Accuracy: 0.9174, Validation Accuracy: 0.6362, Loss: 0.2445
    Epoch  31 Batch 1500/2400 - Train Accuracy: 0.8854, Validation Accuracy: 0.6362, Loss: 0.2618
    Epoch  31 Batch 2000/2400 - Train Accuracy: 0.9351, Validation Accuracy: 0.6362, Loss: 0.1901
    Epoch  32 Batch  500/2400 - Train Accuracy: 0.9437, Validation Accuracy: 0.6362, Loss: 0.1753
    Epoch  32 Batch 1000/2400 - Train Accuracy: 0.9219, Validation Accuracy: 0.6362, Loss: 0.1970
    Epoch  32 Batch 1500/2400 - Train Accuracy: 0.8938, Validation Accuracy: 0.6384, Loss: 0.2547
    Epoch  32 Batch 2000/2400 - Train Accuracy: 0.9231, Validation Accuracy: 0.6362, Loss: 0.1703
    Epoch  33 Batch  500/2400 - Train Accuracy: 0.9479, Validation Accuracy: 0.6384, Loss: 0.1528
    Epoch  33 Batch 1000/2400 - Train Accuracy: 0.9263, Validation Accuracy: 0.6362, Loss: 0.2081
    Epoch  33 Batch 1500/2400 - Train Accuracy: 0.9229, Validation Accuracy: 0.6362, Loss: 0.2178
    Epoch  33 Batch 2000/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6362, Loss: 0.1853
    Epoch  34 Batch  500/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6362, Loss: 0.1620
    Epoch  34 Batch 1000/2400 - Train Accuracy: 0.9464, Validation Accuracy: 0.6362, Loss: 0.2241
    Epoch  34 Batch 1500/2400 - Train Accuracy: 0.8833, Validation Accuracy: 0.6362, Loss: 0.2412
    Epoch  34 Batch 2000/2400 - Train Accuracy: 0.9111, Validation Accuracy: 0.6362, Loss: 0.1426
    Epoch  35 Batch  500/2400 - Train Accuracy: 0.9458, Validation Accuracy: 0.6362, Loss: 0.1431
    Epoch  35 Batch 1000/2400 - Train Accuracy: 0.9308, Validation Accuracy: 0.6362, Loss: 0.2018
    Epoch  35 Batch 1500/2400 - Train Accuracy: 0.9083, Validation Accuracy: 0.6362, Loss: 0.2119
    Epoch  35 Batch 2000/2400 - Train Accuracy: 0.9495, Validation Accuracy: 0.6362, Loss: 0.1822
    Epoch  36 Batch  500/2400 - Train Accuracy: 0.9521, Validation Accuracy: 0.6362, Loss: 0.1330
    Epoch  36 Batch 1000/2400 - Train Accuracy: 0.9420, Validation Accuracy: 0.6362, Loss: 0.1832
    Epoch  36 Batch 1500/2400 - Train Accuracy: 0.9208, Validation Accuracy: 0.6362, Loss: 0.2538
    Epoch  36 Batch 2000/2400 - Train Accuracy: 0.9567, Validation Accuracy: 0.6362, Loss: 0.1514
    Epoch  37 Batch  500/2400 - Train Accuracy: 0.9521, Validation Accuracy: 0.6362, Loss: 0.1227
    Epoch  37 Batch 1000/2400 - Train Accuracy: 0.9464, Validation Accuracy: 0.6362, Loss: 0.1782
    Epoch  37 Batch 1500/2400 - Train Accuracy: 0.9396, Validation Accuracy: 0.6384, Loss: 0.2203
    Epoch  37 Batch 2000/2400 - Train Accuracy: 0.9832, Validation Accuracy: 0.6362, Loss: 0.1357
    Epoch  38 Batch  500/2400 - Train Accuracy: 0.9437, Validation Accuracy: 0.6384, Loss: 0.1310
    Epoch  38 Batch 1000/2400 - Train Accuracy: 0.9420, Validation Accuracy: 0.6384, Loss: 0.1863
    Epoch  38 Batch 1500/2400 - Train Accuracy: 0.9104, Validation Accuracy: 0.6406, Loss: 0.1825
    Epoch  38 Batch 2000/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.1456
    Epoch  39 Batch  500/2400 - Train Accuracy: 0.9583, Validation Accuracy: 0.6384, Loss: 0.1296
    Epoch  39 Batch 1000/2400 - Train Accuracy: 0.9219, Validation Accuracy: 0.6384, Loss: 0.1508
    Epoch  39 Batch 1500/2400 - Train Accuracy: 0.9333, Validation Accuracy: 0.6384, Loss: 0.2060
    Epoch  39 Batch 2000/2400 - Train Accuracy: 0.9567, Validation Accuracy: 0.6362, Loss: 0.1224
    Epoch  40 Batch  500/2400 - Train Accuracy: 0.9521, Validation Accuracy: 0.6384, Loss: 0.1072
    Epoch  40 Batch 1000/2400 - Train Accuracy: 0.9598, Validation Accuracy: 0.6384, Loss: 0.1702
    Epoch  40 Batch 1500/2400 - Train Accuracy: 0.9062, Validation Accuracy: 0.6384, Loss: 0.1843
    Epoch  40 Batch 2000/2400 - Train Accuracy: 0.9591, Validation Accuracy: 0.6384, Loss: 0.1202
    Epoch  41 Batch  500/2400 - Train Accuracy: 0.9604, Validation Accuracy: 0.6384, Loss: 0.1162
    Epoch  41 Batch 1000/2400 - Train Accuracy: 0.9442, Validation Accuracy: 0.6406, Loss: 0.1703
    Epoch  41 Batch 1500/2400 - Train Accuracy: 0.9313, Validation Accuracy: 0.6406, Loss: 0.1889
    Epoch  41 Batch 2000/2400 - Train Accuracy: 0.9519, Validation Accuracy: 0.6384, Loss: 0.1027
    Epoch  42 Batch  500/2400 - Train Accuracy: 0.9708, Validation Accuracy: 0.6384, Loss: 0.1116
    Epoch  42 Batch 1000/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6384, Loss: 0.1699
    Epoch  42 Batch 1500/2400 - Train Accuracy: 0.9250, Validation Accuracy: 0.6406, Loss: 0.1836
    Epoch  42 Batch 2000/2400 - Train Accuracy: 0.9832, Validation Accuracy: 0.6384, Loss: 0.1057
    Epoch  43 Batch  500/2400 - Train Accuracy: 0.9625, Validation Accuracy: 0.6384, Loss: 0.1315
    Epoch  43 Batch 1000/2400 - Train Accuracy: 0.9330, Validation Accuracy: 0.6384, Loss: 0.1550
    Epoch  43 Batch 1500/2400 - Train Accuracy: 0.9167, Validation Accuracy: 0.6384, Loss: 0.1464
    Epoch  43 Batch 2000/2400 - Train Accuracy: 0.9784, Validation Accuracy: 0.6384, Loss: 0.1410
    Epoch  44 Batch  500/2400 - Train Accuracy: 0.9542, Validation Accuracy: 0.6384, Loss: 0.1059
    Epoch  44 Batch 1000/2400 - Train Accuracy: 0.9420, Validation Accuracy: 0.6384, Loss: 0.1310
    Epoch  44 Batch 1500/2400 - Train Accuracy: 0.9542, Validation Accuracy: 0.6384, Loss: 0.1562
    Epoch  44 Batch 2000/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0981
    Epoch  45 Batch  500/2400 - Train Accuracy: 0.9771, Validation Accuracy: 0.6384, Loss: 0.1143
    Epoch  45 Batch 1000/2400 - Train Accuracy: 0.9464, Validation Accuracy: 0.6406, Loss: 0.1576
    Epoch  45 Batch 1500/2400 - Train Accuracy: 0.9104, Validation Accuracy: 0.6406, Loss: 0.1704
    Epoch  45 Batch 2000/2400 - Train Accuracy: 0.9760, Validation Accuracy: 0.6384, Loss: 0.0918
    Epoch  46 Batch  500/2400 - Train Accuracy: 0.9771, Validation Accuracy: 0.6384, Loss: 0.0955
    Epoch  46 Batch 1000/2400 - Train Accuracy: 0.9442, Validation Accuracy: 0.6384, Loss: 0.1452
    Epoch  46 Batch 1500/2400 - Train Accuracy: 0.9458, Validation Accuracy: 0.6384, Loss: 0.1613
    Epoch  46 Batch 2000/2400 - Train Accuracy: 0.9591, Validation Accuracy: 0.6384, Loss: 0.1141
    Epoch  47 Batch  500/2400 - Train Accuracy: 0.9625, Validation Accuracy: 0.6384, Loss: 0.0944
    Epoch  47 Batch 1000/2400 - Train Accuracy: 0.9442, Validation Accuracy: 0.6384, Loss: 0.1473
    Epoch  47 Batch 1500/2400 - Train Accuracy: 0.9437, Validation Accuracy: 0.6384, Loss: 0.1300
    Epoch  47 Batch 2000/2400 - Train Accuracy: 0.9639, Validation Accuracy: 0.6384, Loss: 0.0880
    Epoch  48 Batch  500/2400 - Train Accuracy: 0.9771, Validation Accuracy: 0.6362, Loss: 0.0841
    Epoch  48 Batch 1000/2400 - Train Accuracy: 0.9487, Validation Accuracy: 0.6362, Loss: 0.1205
    Epoch  48 Batch 1500/2400 - Train Accuracy: 0.9250, Validation Accuracy: 0.6384, Loss: 0.1742
    Epoch  48 Batch 2000/2400 - Train Accuracy: 0.9760, Validation Accuracy: 0.6384, Loss: 0.0718
    Epoch  49 Batch  500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0815
    Epoch  49 Batch 1000/2400 - Train Accuracy: 0.9196, Validation Accuracy: 0.6384, Loss: 0.1340
    Epoch  49 Batch 1500/2400 - Train Accuracy: 0.9313, Validation Accuracy: 0.6384, Loss: 0.1400
    Epoch  49 Batch 2000/2400 - Train Accuracy: 0.9712, Validation Accuracy: 0.6362, Loss: 0.0830
    Epoch  50 Batch  500/2400 - Train Accuracy: 0.9812, Validation Accuracy: 0.6362, Loss: 0.0782
    Epoch  50 Batch 1000/2400 - Train Accuracy: 0.9509, Validation Accuracy: 0.6384, Loss: 0.1313
    Epoch  50 Batch 1500/2400 - Train Accuracy: 0.9187, Validation Accuracy: 0.6384, Loss: 0.1251
    Epoch  50 Batch 2000/2400 - Train Accuracy: 0.9639, Validation Accuracy: 0.6384, Loss: 0.0729
    Epoch  51 Batch  500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0558
    Epoch  51 Batch 1000/2400 - Train Accuracy: 0.9442, Validation Accuracy: 0.6362, Loss: 0.1224
    Epoch  51 Batch 1500/2400 - Train Accuracy: 0.9458, Validation Accuracy: 0.6406, Loss: 0.1134
    Epoch  51 Batch 2000/2400 - Train Accuracy: 0.9784, Validation Accuracy: 0.6384, Loss: 0.0652
    Epoch  52 Batch  500/2400 - Train Accuracy: 0.9646, Validation Accuracy: 0.6384, Loss: 0.0769
    Epoch  52 Batch 1000/2400 - Train Accuracy: 0.9442, Validation Accuracy: 0.6384, Loss: 0.1129
    Epoch  52 Batch 1500/2400 - Train Accuracy: 0.9604, Validation Accuracy: 0.6384, Loss: 0.1391
    Epoch  52 Batch 2000/2400 - Train Accuracy: 0.9928, Validation Accuracy: 0.6384, Loss: 0.0652
    Epoch  53 Batch  500/2400 - Train Accuracy: 0.9771, Validation Accuracy: 0.6362, Loss: 0.0690
    Epoch  53 Batch 1000/2400 - Train Accuracy: 0.9263, Validation Accuracy: 0.6384, Loss: 0.1315
    Epoch  53 Batch 1500/2400 - Train Accuracy: 0.9313, Validation Accuracy: 0.6384, Loss: 0.1253
    Epoch  53 Batch 2000/2400 - Train Accuracy: 0.9808, Validation Accuracy: 0.6362, Loss: 0.0595
    Epoch  54 Batch  500/2400 - Train Accuracy: 0.9708, Validation Accuracy: 0.6384, Loss: 0.0667
    Epoch  54 Batch 1000/2400 - Train Accuracy: 0.9464, Validation Accuracy: 0.6384, Loss: 0.1024
    Epoch  54 Batch 1500/2400 - Train Accuracy: 0.9417, Validation Accuracy: 0.6384, Loss: 0.1325
    Epoch  54 Batch 2000/2400 - Train Accuracy: 0.9760, Validation Accuracy: 0.6406, Loss: 0.0741
    Epoch  55 Batch  500/2400 - Train Accuracy: 0.9875, Validation Accuracy: 0.6384, Loss: 0.0728
    Epoch  55 Batch 1000/2400 - Train Accuracy: 0.9442, Validation Accuracy: 0.6384, Loss: 0.1059
    Epoch  55 Batch 1500/2400 - Train Accuracy: 0.9375, Validation Accuracy: 0.6384, Loss: 0.1062
    Epoch  55 Batch 2000/2400 - Train Accuracy: 0.9880, Validation Accuracy: 0.6384, Loss: 0.0564
    Epoch  56 Batch  500/2400 - Train Accuracy: 0.9854, Validation Accuracy: 0.6362, Loss: 0.0583
    Epoch  56 Batch 1000/2400 - Train Accuracy: 0.9442, Validation Accuracy: 0.6384, Loss: 0.0892
    Epoch  56 Batch 1500/2400 - Train Accuracy: 0.9354, Validation Accuracy: 0.6406, Loss: 0.1044
    Epoch  56 Batch 2000/2400 - Train Accuracy: 0.9832, Validation Accuracy: 0.6406, Loss: 0.0562
    Epoch  57 Batch  500/2400 - Train Accuracy: 0.9729, Validation Accuracy: 0.6362, Loss: 0.0621
    Epoch  57 Batch 1000/2400 - Train Accuracy: 0.9509, Validation Accuracy: 0.6384, Loss: 0.1023
    Epoch  57 Batch 1500/2400 - Train Accuracy: 0.9396, Validation Accuracy: 0.6384, Loss: 0.1250
    Epoch  57 Batch 2000/2400 - Train Accuracy: 0.9952, Validation Accuracy: 0.6384, Loss: 0.0470
    Epoch  58 Batch  500/2400 - Train Accuracy: 0.9708, Validation Accuracy: 0.6362, Loss: 0.0548
    Epoch  58 Batch 1000/2400 - Train Accuracy: 0.9509, Validation Accuracy: 0.6362, Loss: 0.1055
    Epoch  58 Batch 1500/2400 - Train Accuracy: 0.9437, Validation Accuracy: 0.6362, Loss: 0.1019
    Epoch  58 Batch 2000/2400 - Train Accuracy: 0.9904, Validation Accuracy: 0.6362, Loss: 0.0470
    Epoch  59 Batch  500/2400 - Train Accuracy: 0.9833, Validation Accuracy: 0.6362, Loss: 0.0600
    Epoch  59 Batch 1000/2400 - Train Accuracy: 0.9643, Validation Accuracy: 0.6384, Loss: 0.0958
    Epoch  59 Batch 1500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.1019
    Epoch  59 Batch 2000/2400 - Train Accuracy: 0.9904, Validation Accuracy: 0.6384, Loss: 0.0639
    Epoch  60 Batch  500/2400 - Train Accuracy: 0.9708, Validation Accuracy: 0.6362, Loss: 0.0632
    Epoch  60 Batch 1000/2400 - Train Accuracy: 0.9531, Validation Accuracy: 0.6384, Loss: 0.0869
    Epoch  60 Batch 1500/2400 - Train Accuracy: 0.9583, Validation Accuracy: 0.6384, Loss: 0.0940
    Epoch  60 Batch 2000/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0639
    Epoch  61 Batch  500/2400 - Train Accuracy: 0.9646, Validation Accuracy: 0.6362, Loss: 0.0734
    Epoch  61 Batch 1000/2400 - Train Accuracy: 0.9531, Validation Accuracy: 0.6362, Loss: 0.1083
    Epoch  61 Batch 1500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.1029
    Epoch  61 Batch 2000/2400 - Train Accuracy: 0.9832, Validation Accuracy: 0.6384, Loss: 0.0375
    Epoch  62 Batch  500/2400 - Train Accuracy: 0.9604, Validation Accuracy: 0.6384, Loss: 0.0530
    Epoch  62 Batch 1000/2400 - Train Accuracy: 0.9442, Validation Accuracy: 0.6362, Loss: 0.0819
    Epoch  62 Batch 1500/2400 - Train Accuracy: 0.9625, Validation Accuracy: 0.6362, Loss: 0.0925
    Epoch  62 Batch 2000/2400 - Train Accuracy: 0.9904, Validation Accuracy: 0.6384, Loss: 0.0440
    Epoch  63 Batch  500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6362, Loss: 0.0428
    Epoch  63 Batch 1000/2400 - Train Accuracy: 0.9554, Validation Accuracy: 0.6384, Loss: 0.0673
    Epoch  63 Batch 1500/2400 - Train Accuracy: 0.9750, Validation Accuracy: 0.6406, Loss: 0.0797
    Epoch  63 Batch 2000/2400 - Train Accuracy: 0.9880, Validation Accuracy: 0.6384, Loss: 0.0533
    Epoch  64 Batch  500/2400 - Train Accuracy: 0.9854, Validation Accuracy: 0.6384, Loss: 0.0441
    Epoch  64 Batch 1000/2400 - Train Accuracy: 0.9554, Validation Accuracy: 0.6406, Loss: 0.0785
    Epoch  64 Batch 1500/2400 - Train Accuracy: 0.9729, Validation Accuracy: 0.6384, Loss: 0.0903
    Epoch  64 Batch 2000/2400 - Train Accuracy: 0.9736, Validation Accuracy: 0.6384, Loss: 0.0497
    Epoch  65 Batch  500/2400 - Train Accuracy: 0.9833, Validation Accuracy: 0.6384, Loss: 0.0518
    Epoch  65 Batch 1000/2400 - Train Accuracy: 0.9621, Validation Accuracy: 0.6384, Loss: 0.0911
    Epoch  65 Batch 1500/2400 - Train Accuracy: 0.9479, Validation Accuracy: 0.6384, Loss: 0.0695
    Epoch  65 Batch 2000/2400 - Train Accuracy: 0.9736, Validation Accuracy: 0.6384, Loss: 0.0594
    Epoch  66 Batch  500/2400 - Train Accuracy: 0.9917, Validation Accuracy: 0.6384, Loss: 0.0545
    Epoch  66 Batch 1000/2400 - Train Accuracy: 0.9576, Validation Accuracy: 0.6384, Loss: 0.0739
    Epoch  66 Batch 1500/2400 - Train Accuracy: 0.9583, Validation Accuracy: 0.6384, Loss: 0.0606
    Epoch  66 Batch 2000/2400 - Train Accuracy: 0.9784, Validation Accuracy: 0.6384, Loss: 0.0524
    Epoch  67 Batch  500/2400 - Train Accuracy: 0.9729, Validation Accuracy: 0.6384, Loss: 0.0360
    Epoch  67 Batch 1000/2400 - Train Accuracy: 0.9554, Validation Accuracy: 0.6384, Loss: 0.0667
    Epoch  67 Batch 1500/2400 - Train Accuracy: 0.9667, Validation Accuracy: 0.6384, Loss: 0.0860
    Epoch  67 Batch 2000/2400 - Train Accuracy: 0.9976, Validation Accuracy: 0.6406, Loss: 0.0416
    Epoch  68 Batch  500/2400 - Train Accuracy: 0.9708, Validation Accuracy: 0.6406, Loss: 0.0395
    Epoch  68 Batch 1000/2400 - Train Accuracy: 0.9732, Validation Accuracy: 0.6384, Loss: 0.0740
    Epoch  68 Batch 1500/2400 - Train Accuracy: 0.9417, Validation Accuracy: 0.6384, Loss: 0.0820
    Epoch  68 Batch 2000/2400 - Train Accuracy: 0.9904, Validation Accuracy: 0.6384, Loss: 0.0369
    Epoch  69 Batch  500/2400 - Train Accuracy: 0.9646, Validation Accuracy: 0.6406, Loss: 0.0675
    Epoch  69 Batch 1000/2400 - Train Accuracy: 0.9509, Validation Accuracy: 0.6384, Loss: 0.1020
    Epoch  69 Batch 1500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0792
    Epoch  69 Batch 2000/2400 - Train Accuracy: 0.9832, Validation Accuracy: 0.6384, Loss: 0.0450
    Epoch  70 Batch  500/2400 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0477
    Epoch  70 Batch 1000/2400 - Train Accuracy: 0.9643, Validation Accuracy: 0.6384, Loss: 0.0592
    Epoch  70 Batch 1500/2400 - Train Accuracy: 0.9458, Validation Accuracy: 0.6384, Loss: 0.0786
    Epoch  70 Batch 2000/2400 - Train Accuracy: 0.9399, Validation Accuracy: 0.6406, Loss: 0.0292
    Epoch  71 Batch  500/2400 - Train Accuracy: 0.9958, Validation Accuracy: 0.6384, Loss: 0.0480
    Epoch  71 Batch 1000/2400 - Train Accuracy: 0.9598, Validation Accuracy: 0.6384, Loss: 0.0945
    Epoch  71 Batch 1500/2400 - Train Accuracy: 0.9417, Validation Accuracy: 0.6406, Loss: 0.0734
    Epoch  71 Batch 2000/2400 - Train Accuracy: 0.9952, Validation Accuracy: 0.6406, Loss: 0.0462
    Epoch  72 Batch  500/2400 - Train Accuracy: 0.9979, Validation Accuracy: 0.6384, Loss: 0.0450
    Epoch  72 Batch 1000/2400 - Train Accuracy: 0.9665, Validation Accuracy: 0.6384, Loss: 0.0678
    Epoch  72 Batch 1500/2400 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0723
    Epoch  72 Batch 2000/2400 - Train Accuracy: 0.9904, Validation Accuracy: 0.6384, Loss: 0.0320
    Epoch  73 Batch  500/2400 - Train Accuracy: 0.9750, Validation Accuracy: 0.6384, Loss: 0.0439
    Epoch  73 Batch 1000/2400 - Train Accuracy: 0.9464, Validation Accuracy: 0.6384, Loss: 0.0437
    Epoch  73 Batch 1500/2400 - Train Accuracy: 0.9938, Validation Accuracy: 0.6406, Loss: 0.0581
    Epoch  73 Batch 2000/2400 - Train Accuracy: 0.9712, Validation Accuracy: 0.6384, Loss: 0.0318
    Epoch  74 Batch  500/2400 - Train Accuracy: 0.9771, Validation Accuracy: 0.6384, Loss: 0.0300
    Epoch  74 Batch 1000/2400 - Train Accuracy: 0.9598, Validation Accuracy: 0.6384, Loss: 0.0770
    Epoch  74 Batch 1500/2400 - Train Accuracy: 0.9729, Validation Accuracy: 0.6384, Loss: 0.0657
    Epoch  74 Batch 2000/2400 - Train Accuracy: 0.9976, Validation Accuracy: 0.6384, Loss: 0.0341
    Epoch  75 Batch  500/2400 - Train Accuracy: 0.9583, Validation Accuracy: 0.6384, Loss: 0.0432
    Epoch  75 Batch 1000/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0615
    Epoch  75 Batch 1500/2400 - Train Accuracy: 0.9667, Validation Accuracy: 0.6406, Loss: 0.0854
    Epoch  75 Batch 2000/2400 - Train Accuracy: 0.9880, Validation Accuracy: 0.6384, Loss: 0.0347
    Epoch  76 Batch  500/2400 - Train Accuracy: 0.9833, Validation Accuracy: 0.6384, Loss: 0.0451
    Epoch  76 Batch 1000/2400 - Train Accuracy: 0.9732, Validation Accuracy: 0.6384, Loss: 0.0500
    Epoch  76 Batch 1500/2400 - Train Accuracy: 0.9646, Validation Accuracy: 0.6384, Loss: 0.0582
    Epoch  76 Batch 2000/2400 - Train Accuracy: 0.9904, Validation Accuracy: 0.6384, Loss: 0.0502
    Epoch  77 Batch  500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0423
    Epoch  77 Batch 1000/2400 - Train Accuracy: 0.9554, Validation Accuracy: 0.6384, Loss: 0.0505
    Epoch  77 Batch 1500/2400 - Train Accuracy: 0.9583, Validation Accuracy: 0.6384, Loss: 0.0560
    Epoch  77 Batch 2000/2400 - Train Accuracy: 0.9904, Validation Accuracy: 0.6384, Loss: 0.0364
    Epoch  78 Batch  500/2400 - Train Accuracy: 0.9750, Validation Accuracy: 0.6384, Loss: 0.0409
    Epoch  78 Batch 1000/2400 - Train Accuracy: 0.9598, Validation Accuracy: 0.6384, Loss: 0.0873
    Epoch  78 Batch 1500/2400 - Train Accuracy: 0.9708, Validation Accuracy: 0.6429, Loss: 0.0721
    Epoch  78 Batch 2000/2400 - Train Accuracy: 0.9952, Validation Accuracy: 0.6384, Loss: 0.0275
    Epoch  79 Batch  500/2400 - Train Accuracy: 0.9896, Validation Accuracy: 0.6384, Loss: 0.0288
    Epoch  79 Batch 1000/2400 - Train Accuracy: 0.9621, Validation Accuracy: 0.6406, Loss: 0.0493
    Epoch  79 Batch 1500/2400 - Train Accuracy: 0.9437, Validation Accuracy: 0.6384, Loss: 0.0632
    Epoch  79 Batch 2000/2400 - Train Accuracy: 0.9928, Validation Accuracy: 0.6384, Loss: 0.0263
    Epoch  80 Batch  500/2400 - Train Accuracy: 0.9979, Validation Accuracy: 0.6406, Loss: 0.0311
    Epoch  80 Batch 1000/2400 - Train Accuracy: 0.9799, Validation Accuracy: 0.6406, Loss: 0.0505
    Epoch  80 Batch 1500/2400 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0590
    Epoch  80 Batch 2000/2400 - Train Accuracy: 0.9904, Validation Accuracy: 0.6384, Loss: 0.0365
    Epoch  81 Batch  500/2400 - Train Accuracy: 0.9896, Validation Accuracy: 0.6384, Loss: 0.0409
    Epoch  81 Batch 1000/2400 - Train Accuracy: 0.9710, Validation Accuracy: 0.6384, Loss: 0.0651
    Epoch  81 Batch 1500/2400 - Train Accuracy: 0.9646, Validation Accuracy: 0.6384, Loss: 0.0750
    Epoch  81 Batch 2000/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0258
    Epoch  82 Batch  500/2400 - Train Accuracy: 0.9792, Validation Accuracy: 0.6384, Loss: 0.0334
    Epoch  82 Batch 1000/2400 - Train Accuracy: 0.9799, Validation Accuracy: 0.6384, Loss: 0.0694
    Epoch  82 Batch 1500/2400 - Train Accuracy: 0.9542, Validation Accuracy: 0.6384, Loss: 0.0614
    Epoch  82 Batch 2000/2400 - Train Accuracy: 0.9784, Validation Accuracy: 0.6384, Loss: 0.0207
    Epoch  83 Batch  500/2400 - Train Accuracy: 0.9917, Validation Accuracy: 0.6384, Loss: 0.0443
    Epoch  83 Batch 1000/2400 - Train Accuracy: 0.9777, Validation Accuracy: 0.6384, Loss: 0.0422
    Epoch  83 Batch 1500/2400 - Train Accuracy: 0.9667, Validation Accuracy: 0.6384, Loss: 0.0801
    Epoch  83 Batch 2000/2400 - Train Accuracy: 0.9928, Validation Accuracy: 0.6384, Loss: 0.0224
    Epoch  84 Batch  500/2400 - Train Accuracy: 1.0000, Validation Accuracy: 0.6384, Loss: 0.0326
    Epoch  84 Batch 1000/2400 - Train Accuracy: 0.9732, Validation Accuracy: 0.6384, Loss: 0.0428
    Epoch  84 Batch 1500/2400 - Train Accuracy: 0.9917, Validation Accuracy: 0.6384, Loss: 0.0445
    Epoch  84 Batch 2000/2400 - Train Accuracy: 0.9952, Validation Accuracy: 0.6384, Loss: 0.0202
    Epoch  85 Batch  500/2400 - Train Accuracy: 0.9854, Validation Accuracy: 0.6384, Loss: 0.0401
    Epoch  85 Batch 1000/2400 - Train Accuracy: 0.9754, Validation Accuracy: 0.6384, Loss: 0.0510
    Epoch  85 Batch 1500/2400 - Train Accuracy: 0.9771, Validation Accuracy: 0.6384, Loss: 0.0610
    Epoch  85 Batch 2000/2400 - Train Accuracy: 0.9928, Validation Accuracy: 0.6384, Loss: 0.0367
    Epoch  86 Batch  500/2400 - Train Accuracy: 0.9896, Validation Accuracy: 0.6384, Loss: 0.0244
    Epoch  86 Batch 1000/2400 - Train Accuracy: 0.9710, Validation Accuracy: 0.6384, Loss: 0.0555
    Epoch  86 Batch 1500/2400 - Train Accuracy: 0.9875, Validation Accuracy: 0.6384, Loss: 0.0467
    Epoch  86 Batch 2000/2400 - Train Accuracy: 0.9832, Validation Accuracy: 0.6384, Loss: 0.0281
    Epoch  87 Batch  500/2400 - Train Accuracy: 0.9812, Validation Accuracy: 0.6384, Loss: 0.0373
    Epoch  87 Batch 1000/2400 - Train Accuracy: 0.9754, Validation Accuracy: 0.6406, Loss: 0.0676
    Epoch  87 Batch 1500/2400 - Train Accuracy: 0.9875, Validation Accuracy: 0.6384, Loss: 0.0354
    Epoch  87 Batch 2000/2400 - Train Accuracy: 0.9784, Validation Accuracy: 0.6384, Loss: 0.0251
    Epoch  88 Batch  500/2400 - Train Accuracy: 0.9854, Validation Accuracy: 0.6406, Loss: 0.0374
    Epoch  88 Batch 1000/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6406, Loss: 0.0392
    Epoch  88 Batch 1500/2400 - Train Accuracy: 0.9688, Validation Accuracy: 0.6384, Loss: 0.0613
    Epoch  88 Batch 2000/2400 - Train Accuracy: 0.9904, Validation Accuracy: 0.6384, Loss: 0.0291
    Epoch  89 Batch  500/2400 - Train Accuracy: 0.9854, Validation Accuracy: 0.6384, Loss: 0.0281
    Epoch  89 Batch 1000/2400 - Train Accuracy: 0.9955, Validation Accuracy: 0.6384, Loss: 0.0439
    Epoch  89 Batch 1500/2400 - Train Accuracy: 0.9646, Validation Accuracy: 0.6406, Loss: 0.0442
    Epoch  89 Batch 2000/2400 - Train Accuracy: 0.9952, Validation Accuracy: 0.6384, Loss: 0.0482


### Evaluate LSTM Net Only


```python
speaker_id, lexicon = list(lexicons.items())[0]
print("List of Speeches:", len(lexicon.speeches))
lexicon.evaluate_testset()
```

    List of Speeches: 20
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


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-19-f6ded009cc9f> in <module>()
          4 speaker_id, lexicon = list(lexicons.items())[0]
          5 _, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
    ----> 6 load_path = helper.load_params(lexicon.cache_dir)
    

    /src/lexicon/helper.py in load_params(cache_dir)
         78     if not cache_dir:
         79         cache_dir = os.path.getcwd()
    ---> 80     with open(os.path.join(cache_dir, 'params.p'), mode='rb') as in_file:
         81         return pickle.load(in_file)
         82 


    FileNotFoundError: [Errno 2] No such file or directory: '/src/lexicon/datacache/lexicon_objects/AlanSiegel_2010/params.p'


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
