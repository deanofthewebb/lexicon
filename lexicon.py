import numpy as np
import os
import unicodedata
import pickle


class Lexicon(object):
    def __init__(self, base_corpus, name='base_corpus', print_report=False):
        cache_file = os.path.join(os.getcwd(), 'datacache', 'lexicon_objects
                                       '{}_preprocess.p'.format(name.strip()))

        if os.path.exists(cache_file):
            # Load cached object
            
            self._name = name.strip()
            self._base_corpus = base_corpus
            self._full_corpus = base_corpus
        else:
            # Create new object
        
            self._name = name.strip()
            self._base_corpus = base_corpus
            self._full_corpus = base_corpus

            # Each speech
            self._speeches = []
            self._vocab_to_int, self._int_to_vocab = self.create_lookup_tables()
        
        
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
        self._full_corpus += speech
    
    def print_loading_report(self):
        print()
        print('Lexicon: "{}" successfully loaded to memory location:'.format(self._name), self)
        print('Dataset Stats')
        print('Number of Unique Words in Base: {}'.format(len({word: None for word in self.base_corpus.split()})))
        print('Number of Unique Words in Speeches: {}'.format(len({word: None for word in ' '.join(self.speeches).split()})))

        print('Number of Speeches: {}'.format(len(self._speeches)))
        word_count_speech = [len(speech.split()) for speech in self._speeches]
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
            '\n': '||return||'
        }
   
    def create_lookup_tables(self):
        """
        Create lookup tables for vocabulary
        :param text: The text of speeches split into words
        :return: A tuple of dicts (vocab_to_int, int_to_vocab)
        """
        vocabs = set(self.split_corpus)
        self._int_to_vocab = dict(enumerate(vocabs, 1))
        self._vocab_to_int = { v: k for k, v in self._int_to_vocab.items()}

        return self._vocab_to_int, self._int_to_vocab


    def preprocess_and_save(self):
        """
        Preprocess Text Data
        """
        cache_directory = os.path.join(os.getcwd(), 'datacache', 'lexicon_objects')
        processed_text = unicodedata.normalize("NFKD", self._full_corpus.strip())  \
                                      .encode("ascii", "ignore") \
                                      .decode("ascii", "ignore")
                
        token_dict = self.token_lookup()
        for key, token in token_dict.items():
            processed_text = processed_text.replace(key, ' {} '.format(token))

        processed_text = processed_text.lower()
        processed_text = processed_text.split()
  
        
        if not os.path.exists(cache_directory):
            os.mkdir(cache_directory)
    
        self._vocab_to_int, self._int_to_vocab = self.create_lookup_tables(processed_text)
        self._int_text = [self._vocab_to_int[word] for word in processed_text]
        pickle.dump((self._int_text, self._vocab_to_int, self._int_to_vocab, token_dict), open(os.path.join(cache_directory, '{}_preprocess.p'.format(self.name)), 'wb'))
        
    def load_preprocess(speaker_id):
            """
            Load the Preprocessed Training data and return them in batches of <batch_size> or less
            """
            cache_directory = os.path.join(os.getcwd(), 'datacache', 'lexicon_objects')
            if os.path.exists(os.path.join(cache_directory, '{}_preprocess.p'.format(speaker_id.strip()))):
                pickle_file = open(os.path.join(cache_directory, '{}_preprocess.p'.format(speaker_id.strip())), mode='rb')
                return pickle.load(pickle_file)
            else:
                print('Nothing saved in the preprocess directory')
                return None
        

    