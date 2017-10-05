import tensorflow as tf
import os
from tensorflow.python.layers.core import Dense

class Seq2SeqModel():
    def __init__(self, batch_size=32, learning_rate= 0.00015, rnn_size = 128, num_layers=1, encoding_embedding_size= 128, decoding_embedding_size = 128, keep_probability = .75, display_step = 100):
        self.train_op = tf.train.AdamOptimizer(learning_rate)
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()
        # Launch the session
        self.sess = tf.InteractiveSession()
        #self.sess.run(init)
        ## Hyperparameters ##
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._rnn_size = rnn_size
        self._num_layers = num_layers
        self._enc_embedding_size = encoding_embedding_size
        self._dec_embedding_size = decoding_embedding_size
        self._keep_probability = keep_probability
        self._display_step = display_step
        
        
        
    ## Properties ##    
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def display_step(self):
        return self._display_step
    @property
    def learning_rate(self):
        return self._learning_rate
    @property
    def rnn_size(self):
        return self._rnn_size
    @property
    def num_layers(self):
        return self._num_layers
    @property
    def enc_embedding_size(self):
        return self._enc_embedding_size
    @property
    def dec_embedding_size(self):
        return self._dec_embedding_size
    @property
    def keep_probability(self):
        return self._keep_probability
    
        
    def model_inputs(self):
        """
        Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
        :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
        max target sequence length, source sequence length)
        """
        inputs = tf.placeholder(tf.int32,[None,None], name = "input")
        targets = tf.placeholder(tf.int32,[None,None], name = "target")
        learning_rate = tf.placeholder(tf.float32, name = "learning_rate")
        keep_probability = tf.placeholder(tf.float32, name = "keep_prob")
        target_sequence_length = tf.placeholder(tf.int32,[None], name = "target_sequence_length")
        max_target_sequence_length = tf.reduce_max(target_sequence_length, name = "max_target_len")
        source_sequence_length = tf.placeholder(tf.int32, [None], name = "source_sequence_length")
        
        return inputs, targets, learning_rate, keep_probability, target_sequence_length, max_target_sequence_length, source_sequence_length
    
    
    def process_decoder_input(self, target_data, target_vocab_to_int):
        """
        Preprocess target data for encoding
        :param target_data: Target Placehoder
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :return: Preprocessed target data
        """    
        target_data = tf.strided_slice(target_data,[0,0],[self.batch_size,-1],[1,1] )
        decoder_input = tf.concat([tf.fill([self.batch_size,1],target_vocab_to_int['<GO>']),target_data],1)
        return decoder_input

        
    def encoding_layer(self, rnn_inputs, source_sequence_length, source_vocab_size):
        """
        Create encoding layer
        :param rnn_inputs: Inputs for the RNN
        :param source_sequence_length: a list of the lengths of each sequence in the batch
        :param source_vocab_size: vocabulary size of source data
        :param enc_embedding_size: embedding size of source data
        :return: tuple (RNN output, RNN state)
        """
        inputs_embeded = tf.contrib.layers.embed_sequence(
                                        ids = rnn_inputs,
                                        vocab_size = source_vocab_size,
                                        embed_dim = self._enc_embedding_size)
        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.rnn_size) for _ in range(self.num_layers) ])
        # cell_dropout = tf.contrib.rnn.DropoutWrapper(cell, self.keep_probability) - #TODO: Optimize later
        # Pass cell and embedded input to tf.nn.dynamic_rnn()
        RNN_output, RNN_state = tf.nn.dynamic_rnn(
                                    cell = cell,
                                    inputs = inputs_embeded,
                                    sequence_length = source_sequence_length,
                                    dtype = tf.float32)
        return RNN_output, RNN_state


    def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input, 
                             target_sequence_length, max_summary_length, 
                             output_layer):
        """
        Create a decoding layer for training
        :param encoder_state: Encoder State
        :param dec_cell: Decoder RNN Cell
        :param dec_embed_input: Decoder embedded input
        :param target_sequence_length: The lengths of each sequence in the target batch
        :param max_summary_length: The length of the longest sequence in the batch
        :param output_layer: Function to apply the output layer
        :return: BasicDecoderOutput containing training logits and sample_id
        """
        training_helper = tf.contrib.seq2seq.TrainingHelper(
                                                inputs = dec_embed_input,
                                                sequence_length = target_sequence_length)
        basic_decoder = tf.contrib.seq2seq.BasicDecoder(
                                                cell = dec_cell,
                                                helper = training_helper,
                                                initial_state = encoder_state,
                                                output_layer = output_layer)
        BasicDecoderOutput = tf.contrib.seq2seq.dynamic_decode(
                                                decoder = basic_decoder,
                                                impute_finished = True,
                                                maximum_iterations = max_summary_length)
        return BasicDecoderOutput[0]
    
    
    def decoding_layer_infer(self, encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                             end_of_sequence_id, max_target_sequence_length,
                             vocab_size, output_layer):
        """
            Create a decoding layer for inference
            :param encoder_state: Encoder state
            :param dec_cell: Decoder RNN Cell
            :param dec_embeddings: Decoder embeddings
            :param start_of_sequence_id: GO ID
            :param end_of_sequence_id: EOS Id
            :param max_target_sequence_length: Maximum length of target sequences
            :param vocab_size: Size of decoder/target vocabulary
            :param decoding_scope: TenorFlow Variable Scope for decoding
            :param output_layer: Function to apply the output layer
            :return: BasicDecoderOutput containing inference logits and sample_id
            """
        # creates a new tensor by replicating start_of_sequence_id batch_size times.
        start_tokens = tf.tile(tf.constant([start_of_sequence_id],dtype = tf.int32),[self.batch_size], name = 'start_tokens' )
        embedding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding = dec_embeddings,
            start_tokens = start_tokens, 
            end_token = end_of_sequence_id)
        basic_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = dec_cell,
            helper = embedding_helper,
            initial_state = encoder_state,
            output_layer = output_layer)
        BasicDecoderOutput = tf.contrib.seq2seq.dynamic_decode(
            decoder = basic_decoder,
            impute_finished = True,
            maximum_iterations = max_target_sequence_length)
        return BasicDecoderOutput[0]
    
    
    def decoding_layer(self, dec_input, encoder_state,
                       target_sequence_length, max_target_sequence_length,
                       target_vocab_to_int, target_vocab_size):
        """
        Create decoding layer
        :param dec_input: Decoder input
        :param encoder_state: Encoder state
        :param target_sequence_length: The lengths of each sequence in the target batch
        :param max_target_sequence_length: Maximum length of target sequences
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :param target_vocab_size: Size of target vocabulary
        :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
        """    
        # Embed the target sequences
        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, self._dec_embedding_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
        # Construct the decoder LSTM cell (just the constructed the encoder cell above)
        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.rnn_size) for _ in range(self.num_layers) ])
        cell_dropout = tf.contrib.rnn.DropoutWrapper(cell, self.keep_probability)
        # Create an output layer to map the outputs of the decoder to the elements of our vocabulary
        output_layer = Dense(target_vocab_size)
        # Use the decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, 
        # max_target_sequence_length, output_layer, keep_prob) function to get the training logits.
        with tf.variable_scope("decode"):
            Training_BasicDecoderOutput = self.decoding_layer_train(encoder_state, 
                                                           cell_dropout, 
                                                           dec_embed_input, 
                                                           target_sequence_length, 
                                                           max_target_sequence_length, 
                                                           output_layer)
        # Use decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, 
        # end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob) 
        # function to get the inference logits.
        with tf.variable_scope("decode", reuse=True):
            Inference_BasicDecoderOutput = self.decoding_layer_infer(encoder_state, 
                                                            cell_dropout, 
                                                            dec_embeddings, 
                                                            target_vocab_to_int['<GO>'], 
                                                            target_vocab_to_int['<EOS>'],
                                                            max_target_sequence_length, 
                                                            target_vocab_size,
                                                            output_layer)
        return Training_BasicDecoderOutput, Inference_BasicDecoderOutput
    
    
    def sentence_to_seq(self, sentence, vocab_to_int):
        """
        Convert a sentence to a sequence of ids
        :param sentence: String
        :param vocab_to_int: Dictionary to go from the words to an id
        :return: List of word ids
        """
        # TODO: Implement Function
        return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sentence.split()]

    def seq2seq(self, input_data, target_data,
                source_sequence_length, target_sequence_length,
                max_target_sentence_length,
                source_vocab_size, target_vocab_size,
                target_vocab_to_int):
        """
        Build the Sequence-to-Sequence part of the neural network
        :param input_data: Input placeholder
        :param target_data: Target placeholder
        :param source_sequence_length: Sequence Lengths of source sequences in the batch
        :param target_sequence_length: Sequence Lengths of target sequences in the batch
        : max_target_sentence_length,
        :param source_vocab_size: Source vocabulary size
        :param target_vocab_size: Target vocabulary size
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
        """
        # Encode the input using the encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,  source_sequence_length, source_vocab_size).
        rnn_output , rnn_state = self.encoding_layer(input_data, 
                       source_sequence_length, 
                       source_vocab_size)
        # Process target data using your process_decoder_input(target_data, target_vocab_to_int, batch_size) function.
        decoder_input = self.process_decoder_input(target_data,
                                            target_vocab_to_int)
        Training_BasicDecoderOutput, Inference_BasicDecoderOutput = self.decoding_layer(
                                            decoder_input,
                                            rnn_state,
                                            target_sequence_length,
                                            max_target_sentence_length,
                                            target_vocab_to_int,
                                            target_vocab_size)
        return Training_BasicDecoderOutput, Inference_BasicDecoderOutput
    

    def save_model(self, sess, checkpoint_path, epoch):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.save(sess, checkpoint_path)
        
        
    def close_session(self):
        self.sess.close()

        
    def load_model(self, sess, checkpoint_path):
        ckpt = tf.train.get_checkpoint_state(os.path.join(self.cache_dir,'checkpoints'))
        if ckpt:
            print("loading model: ",ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)