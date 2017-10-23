import numpy as np
import os
import unicodedata
import pickle
from seq2seq_model import Seq2SeqModel
import nltk
import operator
import time
import copy
import tensorflow as tf
import helper
from tempfile import mkdtemp


class Lexicon(object):
    def load_preprocess(self):
        """
        Load the Preprocessed Training data and return them
        """
        if os.path.exists(self.cache_file):
            return pickle.load(open(self.cache_file, mode='rb'))
        else:
            print('Nothing saved in the preprocess directory')
            return None
            
    def __init__(self, base_corpus, name='base_corpus', print_report=False, validation_set = None,
                 validation_portion = 0.15, test_set = None, testing_portion = 0.10):
        self.cache_dir = os.path.join(os.getcwd(), 'datacache', 'lexicon_objects',name)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir,'{}_preprocess.p'.format(name.strip()))
        

        
        
        if os.path.exists(self.cache_file):
            # Load cached object
            (self._name,
             self._base_corpus,
             self._full_corpus,
             self._int_text, 
             self._vocab_to_int, 
             self._int_to_vocab,
            self._speeches) = self.load_preprocess()
        else:
            # Create new object
        
            self._name = name.strip()
            self._base_corpus = base_corpus
            self._full_corpus = base_corpus

            # Each speech
            self._speeches = []
            
            
            # Preprocess and save lookup data
            self.preprocess_and_save()
        self.model = Seq2SeqModel()
        
        # Print Loading Report
        if print_report:
                self.print_loading_report()
                
        ## Training Properties ##
        self._corpus_sentences = self._full_corpus.split('.')
        self.training_set = (self._corpus_sentences.copy(), self._corpus_sentences.copy()) # (Examples, Targets)
        self.num_examples = len(self._corpus_sentences)
        if validation_set is None:
            validation_size = int(self.num_examples*validation_portion)
            self.validation_set = (self.training_set[0][:validation_size], self.training_set[1][:validation_size]) 
            self.training_set = (self.training_set[0][validation_size:], self.training_set[1][validation_size:]) 
        else:
            self.validation_set = validation_set
            
        if test_set is None:
            test_size = int(self.num_examples*testing_portion)
            self.testing_set = (self.training_set[0][:test_size], self.training_set[1][:test_size]) 
            self.training_set = (self.training_set[0][test_size:], self.training_set[1][test_size:]) 
        else:
            self.training_set = training_set
        
        self.shuffle_training_data()
        ## Early Stopping Parameters ##
        self.best_validation_accuracy = 0.0
        self.last_improvement = None
        self.require_improvement = 2500
        self.current_validation_accuracy = 0.0
        

        
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
    def speech_corpus(self):
        for speech in self.speeches:
            words += speech.ground_truth_transcript
        return ' '.join([speech.ground_truth_transcript for speech in self.speeches])
    @property
    def corpus_sentences(self):
        return self._corpus_sentences
         
    @property
    def full_corpus(self):
        return self._full_corpus
    @property
    def split_corpus(self):
        return self._full_corpus.split()
    @property
    def vocab_size(self):
        return len(set(self.split_corpus)) 
    

    def add_speech(self, speech):
        rand = np.random.randint(2)
        self._speeches.append(speech)

        # Update Training Properties #        
        self._full_corpus += speech.ground_truth_transcript
        self._corpus_sentences.append(speech.ground_truth_transcript)

            
        # Add Candidate Transcripts to Training Sets
        for candidate_transcript in speech.candidate_transcripts:
            self._full_corpus += candidate_transcript["transcript"]
            if rand == 0:
                self.training_set[0].append(candidate_transcript["transcript"])
                self.training_set[1].append(speech.ground_truth_transcript)
            elif rand == 1:
                self.validation_set[0].append(candidate_transcript["transcript"])
                self.validation_set[1].append(speech.ground_truth_transcript)
            else:
                self.testing_set[0].append(candidate_transcript["transcript"])
                self.testing_set[1].append(speech.ground_truth_transcript)
    
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
        
    
    def token_lookup(self):
        """
        Generate a dict to turn punctuation into a token.
        :return: Tokenize dictionary where the key is the punctuation and the value is the token
        """
        return {
            ',': '',
            '(1)': '',
            '(2)': '',
            '(3)': '',
            '(4)': '',
            '(5)': '',
            '(6)': '',
            '(7)': '',
            '(8)': '',
            '(9)': '',
            '"': '',
            ';': '',
            '!': '',
            '?': '',
            '*': '',
            '--': '',
            '{NOISE}': '',
            '{noise}': '',
            '{BREATH}': '',
            '{breath}': '',
            '{UH}': '',
            '{uh}': '',
            '{um}': '',
            '{SMACK}': '',
            '{smack}': '',
            '{COUGH}': '',
            '{cough}': '',
            '<sil>': ''
        }
   
    def create_lookup_tables(self, tokenized_text):
        """
        Create lookup tables for vocabulary
        """
        CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }
        if not isinstance(tokenized_text, list):
            vocab = set(tokenized_text.split())
        else: 
            vocab = set(tokenized_text)
        self.vocab_to_int = copy.copy(CODES)
        for v_i, v in enumerate(vocab, len(CODES)):
            self.vocab_to_int[v] = v_i
        self.int_to_vocab = {v_i: v for v, v_i in self.vocab_to_int.items()}
        return self.vocab_to_int, self.int_to_vocab


    def tokenize_corpus(self, text = None):
        if not text:
            text = self._full_corpus.strip()
        processed_text = unicodedata.normalize("NFKD", text)  \
            .encode("ascii", "ignore") \
            .decode("ascii", "ignore")
            
        token_dict = self.token_lookup()
        for key, token in token_dict.items():
            processed_text = processed_text.replace(key, ' {} '.format(token))

        processed_text = processed_text.lower()
        processed_text = processed_text.split()
        return processed_text
    
    
    def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
        """
        Convert source and target text to proper word ids
        :param source_text: String that contains all the source text.
        :param target_text: String that contains all the target text.
        :param source_vocab_to_int: Dictionary to go from the source words to an id
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :return: A tuple of lists (source_id_text, target_id_text)
        """
        # source_id_text and target_id_text are a list of lists where each list represent a line. 
        # That's why we use a first split('\n')] (not written in the statements)
        source_list = [sentence for sentence in source_text.split('\n')]
        target_list = [sentence for sentence in target_text.split('\n')]

        # Filling the lists
        source_id_text = list()
        target_id_text = list()
        for i in range(len(source_list)):
            source_id_text_temp = list()
            target_id_text_temp = list()
            for word in source_list[i].split():
                source_id_text_temp.append(source_vocab_to_int[word])
            for word in target_list[i].split():
                target_id_text_temp.append(target_vocab_to_int[word])
            # We need to add EOS for target    
            target_id_text_temp.append(target_vocab_to_int['<EOS>'])
            source_id_text.append(source_id_text_temp)
            target_id_text.append(target_id_text_temp)

        return source_id_text, target_id_text
    
    def preprocess_and_save(self):
        """
        Preprocess Text Data
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
  
        tokenized_text = self.tokenize_corpus()
        self._vocab_to_int, self._int_to_vocab = self.create_lookup_tables(tokenized_text)
        self._int_text = [self._vocab_to_int[word] for word in tokenized_text]
        pickle.dump((self._name,
                     self._base_corpus,
                     self._full_corpus,
                     self._int_text, 
                     self._vocab_to_int, 
                     self._int_to_vocab,
                     self._speeches), open(os.path.join(self.cache_dir, '{}_preprocess.p'.format(self.name)), 'wb'))
        
    
    def pad_sentence_batch(sentence_batch, pad_int):
        """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


    def get_batches(self, sources, targets, source_pad_int, target_pad_int):
        """Batch targets, sources, and the lengths of their sentences together"""
        for batch_i in range(0, len(sources)//self.model.batch_size):
            start_i = batch_i * self.model.batch_size

            # Slice the right amount for the batch
            sources_batch = sources[start_i:start_i + self.model.batch_size]
            targets_batch = targets[start_i:start_i + self.model.batch_size]

            # Pad
            pad_sources_batch = np.array(Lexicon.pad_sentence_batch(sources_batch, source_pad_int))
            pad_targets_batch = np.array(Lexicon.pad_sentence_batch(targets_batch, target_pad_int))

            # Need the lengths for the _lengths parameters
            pad_targets_lengths = []
            for target in pad_targets_batch:
                pad_targets_lengths.append(len(target))

            pad_source_lengths = []
            for source in pad_sources_batch:
                pad_source_lengths.append(len(source))

            yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

    
    
    def build_graph(self, source_vocab_to_int, target_vocab_to_int):
        checkpoint_dir = os.path.join(self.cache_dir,'checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')

        # load previously trained model if appilcable
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt:
            print('Preloading model')
            self.model.load_model(checkpoint_path)
        

        # Training Params
        self.train_graph = tf.Graph()
        
        with self.train_graph.as_default():
            (self.input_data_ph, 
             self.targets_ph, 
             self.lr_ph, 
             self.keep_prob_ph, 
             self.target_sequence_length_ph, 
             self.max_target_sequence_length_ph, 
             self.source_sequence_length_ph) = self.model.model_inputs()
            
            train_logits, inference_logits = self.model.seq2seq(tf.reverse(self.input_data_ph, [-1]),
                                                   self.targets_ph,
                                                   self.source_sequence_length_ph,
                                                   self.target_sequence_length_ph,
                                                   self.max_target_sequence_length_ph,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   target_vocab_to_int)
            self.training_logits = tf.identity(train_logits.rnn_output, name='logits')
            self.inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
            masks = tf.sequence_mask(self.target_sequence_length_ph, self.max_target_sequence_length_ph, dtype=tf.float32, name='masks')       
            with tf.name_scope("optimization"):
                # Loss function
                self.loss = tf.contrib.seq2seq.sequence_loss(self.training_logits, self.targets_ph, masks)
                gradients = self.model.train_op.compute_gradients(self.loss)
                capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
                self.train_op = self.model.train_op.apply_gradients(capped_gradients)
                return self.train_graph

        
    def get_accuracy(target, logits):
        max_seq = max(target.shape[1], logits.shape[1])
        if max_seq - target.shape[1]:
            target = np.pad(target, [(0,0),(0,max_seq - target.shape[1])], 'constant')
        if max_seq - logits.shape[1]:
            logits = np.pad(logits, [(0,0),(0,max_seq - logits.shape[1])], 'constant')

        return np.mean(np.equal(target, logits))

    
    def optimize(self, early_stop=True): # Return training Report
        start_time = time.time()
        #Set the epochs value high to try to trigger early_stop conditions.
        if early_stop:
            num_epochs = 90
        else:
            num_epochs = 10
        start_time = time.time()


        #Get Preprocessing Data - Also use helper.load_preprocess()
        source_text = '\n'.join(self.training_set[0])
        target_text = '\n'.join(self.training_set[1])
        source_validation_text = '\n'.join(self.validation_set[0])
        target_validation_text = '\n'.join(self.validation_set[1])
        
        # Join the Training And Validation Text for Creating Lookup Tables
        source_vocab_to_int, source_int_to_vocab = self.create_lookup_tables('\n'.join([source_text, source_validation_text]))
        target_vocab_to_int, target_int_to_vocab = self.create_lookup_tables('\n'.join([target_text, target_validation_text]))
        source_text_ids, target_text_ids = Lexicon.text_to_ids(source_text, target_text, source_vocab_to_int,
                                                                   target_vocab_to_int)
        # Build Graph
        self.train_graph = self.build_graph(source_vocab_to_int, target_vocab_to_int)


        # (val_source_vocab_to_int, 
        # val_source_int_to_vocab) = self.create_lookup_tables(source_validation_text)
        # (val_target_vocab_to_int, 
        # val_target_int_to_vocab) = self.create_lookup_tables(target_validation_text)
        source_validation_text_ids, target_validation_text_ids = Lexicon.text_to_ids(source_validation_text,
                                                            target_validation_text,
                                                            source_vocab_to_int,
                                                            target_vocab_to_int)
        (valid_sources_batch, 
         valid_targets_batch, 
         valid_sources_lengths, 
         valid_targets_lengths ) = next(self.get_batches(source_validation_text_ids, target_validation_text_ids,
                                                             source_vocab_to_int['<PAD>'],
                                                             target_vocab_to_int['<PAD>']))
        with tf.Session(graph=self.train_graph) as sess:
            init = tf.global_variables_initializer()
            self.model.sess = sess
            # Launch the session
            sess.run(init)
            for epoch_i in range(num_epochs):
                self.shuffle_training_data()
                early_stopping = False
                for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                    self.get_batches(source_text_ids, source_text_ids, source_vocab_to_int['<PAD>'],
                                     target_vocab_to_int['<PAD>'])):
                    total_iterations = int((epoch_i+1)*(batch_i+1))
                    _, loss = sess.run(
                        [self.train_op, self.loss],
                        {self.input_data_ph: source_batch,
                         self.targets_ph: target_batch,
                         self.lr_ph: self.model.learning_rate,
                         self.target_sequence_length_ph: targets_lengths,
                         self.source_sequence_length_ph: sources_lengths,
                         self.keep_prob_ph: 0.75})


                    if batch_i % self.model._display_step == 0 and batch_i > 0:
                        batch_train_logits = sess.run(
                            self.inference_logits,
                            {self.input_data_ph: source_batch,
                             self.source_sequence_length_ph: sources_lengths,
                             self.target_sequence_length_ph: targets_lengths,
                             self.keep_prob_ph: 1.0})


                        batch_valid_logits = sess.run(
                            self.inference_logits,
                            {self.input_data_ph: valid_sources_batch,
                             self.source_sequence_length_ph: valid_sources_lengths,
                             self.target_sequence_length_ph: valid_targets_lengths,
                             self.keep_prob_ph: 1.0})

                        train_acc = Lexicon.get_accuracy(target_batch, batch_train_logits)
                        valid_acc = Lexicon.get_accuracy(valid_targets_batch, batch_valid_logits)
                        
                        
                        if valid_acc > self.best_validation_accuracy:
                            # Update the best-known validation accuracy.
                            self.best_validation_accuracy = valid_acc

                            # Set the iteration for the last improvement to current.
                            self.last_improvement = total_iterations

                            # A string to be printed below, shows improvement found.
                            improved_str = '**'

                        elif self.current_validation_accuracy > valid_acc:
                            improved_str = '*'
                        else:
                            # An empty string to be printed below shows that no improvement was found.
                            improved_str = ''

                            if (valid_acc < self.best_validation_accuracy) and \
                            (total_iterations - self.last_improvement) > self.require_improvement:
                                print("No improvement found in a while, stopping optimization.")
                                # Break out from the for-loop.
                                early_stopping = True
                                break
                        
                        # Set Current Validation Accuracy
                        self.current_validation_accuracy = valid_acc

                        # Status-message for printing.
                        print('Epoch {0:>3} Batch {1:>4}/{2} - Train Accuracy: {3:>6.4f}, Validation Accuracy: {4:>6.4f}, Loss:{5:>6.4f} Improve?: {6}'.format(epoch_i, batch_i, len(source_text_ids) // self.model.batch_size, train_acc, valid_acc, loss, improved_str))

                        # Save Model
                        checkpoint_dir = os.path.join(self.cache_dir,'checkpoints')
                        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')

                        self.model.save_model(sess, checkpoint_path,batch_i*(epoch_i+1))
                        #print("Model saved in file: %s" % checkpoint_path)

                        helper.save_params(checkpoint_path)
                if early_stopping is True:
                    break
                        
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            
    def sentence_to_seq(sentence, vocab_to_int):
        """
        Convert a sentence to a sequence of ids
        :param sentence: String
        :param vocab_to_int: Dictionary to go from the words to an id
        :return: List of word ids
        """

        # Convert the sentence to lowercase and to list
        list_words = [word for word in sentence.lower().split() ]

        # Convert words into ids using vocab_to_int
        list_words_int = list()
        for word in list_words:
            # Convert words not in the vocabulary, to the <UNK> word id.
            if word not in vocab_to_int:
                list_words_int.append(vocab_to_int['<UNK>'])
            else:
                list_words_int.append(vocab_to_int[word])
        return list_words_int
    

    def evaluate_testset(self):
        steps = 0
        show_results = 1000

        # Load saved model
        checkpoint_dir = os.path.join(self.cache_dir,'checkpoints/')
        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')

        # Checkpoints
        loaded_graph  = self.train_graph
        with tf.Session(graph=loaded_graph) as sess:

            #loader = tf.train.import_meta_graph(os.path.join(checkpoint_dir, 'model.ckpt-5000.meta'))
            #lexicon.model.saver.restore(sess, checkpoint_path)
            #self.model.load_model(sess, checkpoint_path)

            input_data = loaded_graph.get_tensor_by_name('input:0')
            logits = loaded_graph.get_tensor_by_name('predictions:0')
            target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
            source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
            keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

            token_dict = self.token_lookup()
            cloud_speech_api_accuracy = []
            custom_lang_model_accuracy = []

            for speech in self.speeches:
                gt_transcript = speech.ground_truth_transcript.lower()
                for key, token in token_dict.items():
                    gt_transcript = gt_transcript.replace(key, ' {} '.format(token))




                # Collect Google API Transcript
                google_api_transcript = ""
                words = []
                if speech.candidate_timestamps:
                    print('if speech.candidate_timestamps')
                    for candidate_timestamp in speech.candidate_timestamps:
                        words.append(candidate_timestamp["word"])
                    google_api_transcript = " ".join(words)


                if speech.candidate_timestamps:
                    print('if speech.candidate_timestamps 2')
                    candidate_script_accuracy = []
                    for candidate_transcript in speech.candidate_transcripts:
                        steps +=1
                        transcription_sentence = Lexicon.sentence_to_seq(candidate_transcript["transcript"],
                                                                         source_vocab_to_int)
                        transcription_logits = sess.run(logits, 
                                                        {input_data: [transcription_sentence]*self.model.batch_size,
                                                         target_sequence_length: [len(transcription_sentence)*2]*self.model.batch_size,
                                                         source_sequence_length: [len(transcription_sentence)]*self.model.batch_size,
                                                         keep_prob: 1.0})[0]
                        prediction_transcript = " ".join([target_int_to_vocab[i] for i in transcription_logits])
                        # Remove <EOS> Token
                        prediction_transcript = prediction_transcript.replace('<EOS>','')

                        if steps % show_results == 0:  
                            print()
                            print('GCS Candidate Transcript: \n{}'.format(" ".join([source_int_to_vocab[i] for i in transcription_sentence])))
                            print('Seq2Seq Model Prediction Transcript: \n{}'.format(prediction_transcript))
                            print('Ground Truth Transcript: \n{}'.format(gt_transcript))
                            print()

                        # Compute the Candidate Transcript Edit Distance (a.k.a. From the Predicted Distance)
                        # Use this to determine how likely sentence would have been predicted
                        gct_ed = nltk.edit_distance(candidate_transcript["transcript"].lower(), prediction_transcript.lower())
                        gct_upper_bound = max(len(candidate_transcript["transcript"]),len(prediction_transcript))
                        gct_accuracy = (1.0 - gct_ed/gct_upper_bound)

                        gct_accuracy = gct_accuracy*candidate_transcript["confidence"]
                        candidate_script_accuracy.append(gct_accuracy)



                    # Select Candidate Transcript with the highest accuracy (to prediction)

                    index, value = max(enumerate(candidate_script_accuracy), key=operator.itemgetter(1))

                    tmp = []
                    for candidate_transcript in speech.candidate_transcripts:
                        tmp.append(candidate_transcript["transcript"])

                    reranked_transcript = tmp[index]


                    # Collect Accuracy between reranked transcript and Google transcript                      
                    gcs_ed = nltk.edit_distance(google_api_transcript.lower(), gt_transcript.lower())
                    gcs_upper_bound = max(len(google_api_transcript),len(gt_transcript))
                    gcs_accuracy = (1.0 - gcs_ed/gcs_upper_bound)

                    clm_ed = nltk.edit_distance(reranked_transcript.lower(), gt_transcript.lower())
                    clm_upper_bound = max(len(reranked_transcript),len(gt_transcript))
                    clm_accuracy = (1.0 - clm_ed/clm_upper_bound)

                    cloud_speech_api_accuracy.append(gcs_accuracy)
                    custom_lang_model_accuracy.append(clm_accuracy)

            print('Speech Results:')
            print('Average Candidate Transcript Accuracy:', np.mean(cloud_speech_api_accuracy))
            print('Average Seq2Seq Model Accuracy:', np.mean(custom_lang_model_accuracy))
            print()


        
        
    def shuffle_training_data(self):
        perm = [i for i in range(len(self.training_set[0]))]
        np.random.shuffle(perm)
        tmp = np.array(self.training_set[0])[perm]       
        tmp2 = np.array(self.training_set[1])[perm]
        self.training_set = (tmp,tmp2)
    