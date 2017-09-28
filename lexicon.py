import numpy as np
import os
import unicodedata
import pickle


class Lexicon(object):
    def load_preprocess(cache_file):
        """
        Load the Preprocessed Training data and return them
        """
        if os.path.exists(cache_file):
            return pickle.load(open(cache_file, mode='rb'))
        else:
            print('Nothing saved in the preprocess directory')
            return None
            
    def __init__(self, base_corpus, name='base_corpus', print_report=False):
        cache_file = os.path.join(os.getcwd(), 'datacache', 'lexicon_objects',
                                       '{}_preprocess.p'.format(name.strip()))

        if os.path.exists(cache_file):
            # Load cached object
            (self._name,
             self._base_corpus,
             self._full_corpus,
             self._int_text, 
             self._vocab_to_int, 
             self._int_to_vocab) = Lexicon.load_preprocess(cache_file)
            
            self._speeches = []
        else:
            # Create new object
        
            self._name = name.strip()
            self._base_corpus = base_corpus
            self._full_corpus = base_corpus

            # Each speech
            self._speeches = []
            
            # Preprocess and save lookup data
            self.preprocess_and_save()
        
        
        # Print Loading Report
        if print_report:
                self.print_loading_report()
        

        
    @property
    def speeches(self):
        return self._speeches
    @property
    def name(self):
        return self._name
    @property
    def base_corpus(self):
        return self._base_corpus
    @property
    def full_corpus(self):
        return self._full_corpus
    @property
    def split_corpus(self):
        return self._full_corpus.split()
    @property
    def filenames(self):
        return self._filenames

    def add_speech(self, speech):
        self._speeches.append(speech)
        self._full_corpus += speech.ground_truth_transcript
    
    def print_loading_report(self):
        print()
        print('Lexicon: "{}" successfully loaded to memory location:'.format(self._name), self)
        print('Dataset Stats')
        print('Number of Unique Words in Base: {}'.format(len({word: None for word in self.base_corpus.split()})))
        
        words = u""
        for speech in self.speeches:
            words += speech.ground_truth_transcript
        print('Number of Unique Words in Speeches: {}'.format(len({word: None for word in words.split()})))

        print('Number of Speeches: {}'.format(len(self._speeches)))
        word_count_speech = [len(speech.ground_truth_transcript.split()) for speech in self._speeches]
        print('Average number of words in each speech: {}'.format(np.mean(word_count_speech)))
        print()
        print()
        
    
    # Preprocessing
    def token_lookup(self):
        """
        Generate a dict to turn punctuation into a token.
        :return: Tokenize dictionary where the key is the punctuation and the value is the token
        """
        return {
            '.': '||period||',
            ',': '||comma||',
            '"': '||quotation||',
            ';': '||semi_colon||',
            '!': '||exclamation||',
            '?': '||question||',
            '(': '||left_parentheses||',
            ')': '||right_parentheses||',
            '*': '||star||',
            '--': '||dash||',
            '{NOISE}': '',
            '{BREATH}': '',
            '{UH}': '',
            '{SMACK}': '',
            '{COUGH}': '',
            '<sil>': '',
            '\n': '||return||'
        }
   
    def create_lookup_tables(self, tokenized_text):
        """
        Create lookup tables for vocabulary
        :return: A tuple of dicts (vocab_to_int, int_to_vocab)
        """
        vocabs = set(tokenized_text)
        self._int_to_vocab = dict(enumerate(vocabs, 1))
        self._vocab_to_int = { v: k for k, v in self._int_to_vocab.items()}

        return self._vocab_to_int, self._int_to_vocab


    def tokenize_corpus(self):
        processed_text = unicodedata.normalize("NFKD", self._full_corpus.strip())  \
            .encode("ascii", "ignore") \
            .decode("ascii", "ignore")
            
        token_dict = self.token_lookup()
        for key, token in token_dict.items():
            processed_text = processed_text.replace(key, ' {} '.format(token))

        processed_text = processed_text.lower()
        processed_text = processed_text.split()
        return processed_text
    
    def preprocess_and_save(self):
        """
        Preprocess Text Data
        """
        cache_directory = os.path.join(os.getcwd(), 'datacache', 'lexicon_objects')
        if not os.path.exists(cache_directory):
            os.mkdir(cache_directory)
  
        tokenized_text = self.tokenize_corpus()
        self._vocab_to_int, self._int_to_vocab = self.create_lookup_tables(tokenized_text)
        self._int_text = [self._vocab_to_int[word] for word in tokenized_text]
        pickle.dump((self._name,
                     self._base_corpus,
                     self._full_corpus,
                     self._int_text, 
                     self._vocab_to_int, 
                     self._int_to_vocab), open(os.path.join(cache_directory, '{}_preprocess.p'.format(self.name)), 'wb'))
        
        

    