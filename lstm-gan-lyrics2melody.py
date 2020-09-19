import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.python.ops import array_ops
from shutil import copyfile
import utils
import midi_statistics
import argparse
import os
import time
import datetime
import shutil
import mmd


def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell,
                  state_is_tuple=True,
                  reuse=False):
    '''
    Makes a RNN cell from the given hyperparameters.

      Args:
        rnn_layer_sizes: A list of integer sizes (in units) for each layer of the RNN.
        dropout_keep_prob: The float probability to keep the output of any given sub-cell.
        attn_length: The size of the attention vector.
        base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
        state_is_tuple: A boolean specifying whether to use tuple of hidden matrix
            and cell matrix as a state instead of a concatenated matrix.

      Returns:
          A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
    '''

    cells = []
    for num_units in rnn_layer_sizes:
        cell = base_cell(num_units, state_is_tuple=state_is_tuple)  # , reuse=reuse)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
        cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
    print(cell)
    if attn_length:
        cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length, state_is_tuple=state_is_tuple, reuse=reuse)
        #cell = tf.contrib.seq2seq.AttentionWrapper(cell, tf.contrib.seq2seq.AttentionMechanism(attn_length,attn_length))

    return cell


def linear(inp, output_dim, scope=None, stddev=1.0, reuse_scope=False):
    """
    Linear layer returning Wx + b, where x is the input and W is [output_dim x length(input)]
    """
    norm = tf.random_normal_initializer(stddev=stddev, dtype=tf.float32)
    const = tf.constant_initializer(0.0, dtype=tf.float32)
    with tf.variable_scope(scope or 'linear') as scope:
        scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=REG_SCALE))
        if reuse_scope:
            scope.reuse_variables()
        w = tf.get_variable('w', [inp.get_shape()[1], output_dim], initializer=norm, dtype=tf.float32)
        b = tf.get_variable('b', [output_dim], initializer=const, dtype=tf.float32)
    return tf.matmul(inp, w) + b


class RNNGAN(object):
    """
    The RNNGAN model
    """
    def __init__(self, is_training, num_song_features=None, num_meta_features=None, conditioning='multi'):
        songlength = SONGLENGTH
        self.songlength = songlength
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        keep_prob = self.keep_prob
        self._input_songdata = tf.placeholder(shape=[None, songlength, num_song_features], dtype=tf.float32,
                                              name="input_data")
        if conditioning is not 'none':
            self._input_metadata = tf.placeholder(shape=[None, songlength, num_meta_features], dtype=tf.float32,
                                                  name="input_metadata")
            self._input_metadata_wrong = tf.placeholder(shape=[None, songlength, num_meta_features], dtype=tf.float32,
                                                        name="input_metadata_wrong")
        self.batch_size = array_ops.shape(self._input_songdata)[0]
        batch_size = self.batch_size
        songdata_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(self._input_songdata, songlength, 1)]
        if conditioning == 'multi':
            metadata_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(self._input_metadata, songlength, 1)]
            wrong_metadata_inputs = [tf.squeeze(input_, [1])
                                     for input_ in tf.split(self._input_metadata_wrong, songlength, 1)]

        with tf.variable_scope('G') as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=REG_SCALE))
            g_cell_fw = make_rnn_cell([HIDDEN_SIZE_G] * NUM_LAYERS_G, dropout_keep_prob=keep_prob,
                                    attn_length=ATTENTION_LENGTH)
            g_cell_bw = make_rnn_cell([HIDDEN_SIZE_G] * NUM_LAYERS_G, dropout_keep_prob=keep_prob,
                                    attn_length=ATTENTION_LENGTH)
            self.g_initial_state_fw = g_cell_fw.zero_state(batch_size, tf.float32)
            self.g_initial_state_bw = g_cell_bw.zero_state(batch_size, tf.float32)
            g_state_fw = self.g_initial_state_fw
            g_state_bw = self.g_initial_state_bw

            random_rnninputs = tf.random_uniform(
                shape=[batch_size, songlength, int(RANDOM_INPUT_SCALE * num_song_features)], minval=0.0,
                maxval=1.0, dtype=tf.float32)

            random_rnninputs = [tf.squeeze(input_, [1]) for input_ in tf.split(random_rnninputs, songlength, 1)]
            self._generated_features = []
            self._generated_features_pretraining = []
            prev_target = tf.random_uniform(shape=[batch_size, num_song_features], minval=0.0, maxval=1.0,
                                            dtype=tf.float32)

            # REAL GENERATOR:
            if BIDIRECTIONAL_G:
                outputs, g_state_fw, g_state_bw = tf.contrib.rnn.static_bidirectional_rnn(g_cell_fw, g_cell_bw,
                                                                                          random_rnninputs,
                                                                                          initial_state_fw=
                                                                                          self.g_initial_state_fw,
                                                                                          initial_state_bw=
                                                                                          self.g_initial_state_bw)
                for i, output_ in enumerate(outputs):
                    generated_point = linear(output_, num_song_features, scope='output_layer', reuse_scope=(i != 0))
                    self._generated_features.append(generated_point)
                    self._generated_features_pretraining.append(generated_point)
                    prev_target = songdata_inputs[i]

            else:
                # as we feed the output as the input to the next, we 'invent' the initial 'output'.
                generated_point = tf.random_uniform(shape=[batch_size, num_song_features], minval=0.0, maxval=1.0,
                                                    dtype=tf.float32)

                outputs = []
                for i, input_ in enumerate(random_rnninputs):
                    '''
                    if i > 0:
                        scope.reuse_variables()
                    '''
                    concat_values = [input_]
                    if not DISABLE_FEED_PREVIOUS:
                        concat_values.append(generated_point)
                    if conditioning == 'multi':
                        concat_values.append(metadata_inputs[i])
                    elif conditioning == 'unique':
                        concat_values.append(self._input_metadata)

                    if len(concat_values):
                        input_ = tf.concat(axis=1, values=concat_values)
                    input_ = tf.nn.relu(linear(input_, HIDDEN_SIZE_G,
                                               scope='input_layer', reuse_scope=(i != 0)))
                    output, g_state_fw = g_cell_fw(input_, g_state_fw)
                    outputs.append(output)
                    generated_point = linear(output, num_song_features, scope='output_layer', reuse_scope=(i != 0))
                    self._generated_features.append(generated_point)

                # PRETRAINING GENERATOR, will feed inputs, not generated outputs:
                scope.reuse_variables()
                # as we feed the output as the input to the next, we 'invent' the initial 'output'.
                outputs = []
                for i, input_ in enumerate(random_rnninputs):
                    concat_values = [input_]
                    if not DISABLE_FEED_PREVIOUS:
                        concat_values.append(prev_target)
                    if conditioning == 'multi':
                        concat_values.append(metadata_inputs[i])
                    elif conditioning == 'unique':
                        concat_values.append(self._input_metadata)
                    if len(concat_values):
                        input_ = tf.concat(axis=1, values=concat_values)
                    input_ = tf.nn.relu(linear(input_, HIDDEN_SIZE_G, scope='input_layer', reuse_scope=(i != 0)))
                    output, g_state_fw = g_cell_fw(input_, g_state_fw)
                    outputs.append(output)
                    generated_point = linear(output, num_song_features, scope='output_layer', reuse_scope=(i != 0))
                    self._generated_features_pretraining.append(generated_point)
                    prev_target = songdata_inputs[i]

            self._final_state = g_state_fw

        # These are used both for pretraining and for D/G training further down.
        self._lr = tf.Variable(LEARNING_RATE, trainable=False, dtype=tf.float32)
        self.g_params = [v for v in tf.trainable_variables() if v.name.startswith('model/G/')]
        if ADAM:
            g_optimizer = tf.train.AdamOptimizer(self._lr)
        else:
            g_optimizer = tf.train.GradientDescentOptimizer(self._lr)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = REG_CONSTANT * sum(reg_losses)
        reg_loss = tf.Print(reg_loss, reg_losses, 'reg_losses = ', summarize=20)
        # ---BEGIN, PRETRAINING. ---
        self.rnn_pretraining_loss = tf.reduce_mean(
            tf.squared_difference(x=tf.transpose(tf.stack(self._generated_features_pretraining), perm=[1, 0, 2]),
                                  y=self._input_songdata))
        if not DISABLE_L2_REG:
            self.rnn_pretraining_loss = self.rnn_pretraining_loss + reg_loss

        pretraining_grads, _ = tf.clip_by_global_norm(tf.gradients(self.rnn_pretraining_loss, self.g_params),
                                                      MAX_GRAD_NORM)
        self.opt_pretraining = g_optimizer.apply_gradients(zip(pretraining_grads, self.g_params))

        # ---END, PRETRAINING---

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('D') as scope:
            cell_fw = make_rnn_cell([HIDDEN_SIZE_D] * NUM_LAYERS_D, dropout_keep_prob=keep_prob,
                                    attn_length=ATTENTION_LENGTH)
            cell_bw = make_rnn_cell([HIDDEN_SIZE_D] * NUM_LAYERS_D, dropout_keep_prob=keep_prob,
                                    attn_length=ATTENTION_LENGTH)

            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=REG_SCALE))
            # Make list of tensors. One per step in recurrence.
            # Each tensor is batchsize * numfeatures.
            if conditioning == 'multi' and FEED_COND_D:
                conditioned_songdata_inputs = [tf.concat([metadata_input, songdata_input], 1) for metadata_input,
                                                                                                  songdata_input in
                                               zip(metadata_inputs, songdata_inputs)]
            elif conditioning == 'unique' and FEED_COND_D:
                conditioned_songdata_inputs = [tf.concat([self._input_metadata, songdata_input], 1) for songdata_input
                                               in
                                               songdata_inputs]
            else:
                conditioned_songdata_inputs = [tf.concat([songdata_input], 1) for songdata_input in songdata_inputs]

            self.real_d, self.real_d_features = self.discriminator(conditioned_songdata_inputs, cell_fw, cell_bw,
                                                                   is_training, msg='real',
                                                                   keep_prob=keep_prob)

            scope.reuse_variables()

            # real data but wrong condition
            if conditioning == 'multi' and FEED_COND_D:
                songdata_wrong_condition_inputs = [tf.concat([wrong_metadata_input, songdata_input], 1) for
                                                   wrong_metadata_input, songdata_input in zip(wrong_metadata_inputs,
                                                                                               songdata_inputs)]
                self.wrong_d, self.wrong_d_features = self.discriminator(songdata_wrong_condition_inputs,
                                                                         cell_fw, cell_bw, is_training,
                                                                         msg='wrong', keep_prob=keep_prob)

            elif conditioning == 'unique' and FEED_COND_D:
                songdata_wrong_condition_inputs = [tf.concat([self._input_metadata_wrong, songdata_input], 1) for
                                                   songdata_input in songdata_inputs]
                self.wrong_d, self.wrong_d_features = self.discriminator(songdata_wrong_condition_inputs,cell_fw,
                                                                         cell_bw, is_training,
                                                                         msg='wrong', keep_prob=keep_prob)

            if conditioning == 'multi' and FEED_COND_D:
                generated_data = [tf.concat([metadata_input, songdata_input], 1) for metadata_input, songdata_input in
                                  zip(metadata_inputs, self._generated_features)]
            elif conditioning == 'unique' and FEED_COND_D:
                generated_data = [tf.concat([self._input_metadata, songdata_input], 1) for songdata_input in
                                  self._generated_features]
            else:
                generated_data = [tf.concat(songdata_input, 1) for songdata_input in self._generated_features]

            self.generated_d, self.generated_d_features = self.discriminator(generated_data, cell_fw, cell_bw,
                                                                             is_training,
                                                                             msg='generated', keep_prob=keep_prob)
            # else:
            #  generated_data = self._generated_features

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        if conditioning == 'none' or LOSS_WRONG_D is False:
            self.d_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(self.real_d, 1e-1000000, 1.0))
                                         - tf.log(1 - tf.clip_by_value(self.generated_d, 0.0, 1.0 - 1e-1000000)))
        else:
            self.d_loss = tf.reduce_mean(-2 * tf.log(tf.clip_by_value(self.real_d, 1e-1000000, 1.0))
                                         - tf.log(1 - tf.clip_by_value(self.generated_d, 0.0, 1.0 - 1e-1000000))
                                         - 2 * tf.log(1 - tf.clip_by_value(self.wrong_d, 0.0, 1.0 - 1e-1000000)))
        self.g_loss_feature_matching = tf.reduce_sum(
            tf.squared_difference(self.real_d_features, self.generated_d_features))
        self.g_loss = tf.reduce_mean(tf.log(tf.clip_by_value(1-self.generated_d, 1e-1000000, 1.0)))

        if not DISABLE_L2_REG:
            self.d_loss = self.d_loss + reg_loss
            self.g_loss_feature_matching = self.g_loss_feature_matching + reg_loss
            self.g_loss = self.g_loss + reg_loss
        self.d_params = [v for v in tf.trainable_variables() if v.name.startswith('model/D/')]

        if not is_training:
            return

        d_optimizer = tf.train.GradientDescentOptimizer(self._lr * D_LR_FACTOR)
        d_grads, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss, self.d_params),
                                            MAX_GRAD_NORM)
        self.opt_d = d_optimizer.apply_gradients(zip(d_grads, self.d_params))
        if FEATURE_MATCHING:
            g_grads, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss_feature_matching,
                                                             self.g_params), MAX_GRAD_NORM)
        else:
            g_grads, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params), MAX_GRAD_NORM)
        self.opt_g = g_optimizer.apply_gradients(zip(g_grads, self.g_params))

        self._new_lr = tf.placeholder(shape=[], name="new_learning_rate", dtype=tf.float32)
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def discriminator(self, inputs, cell_fw, cell_bw, is_training, msg='', keep_prob=1.0):
        """
        RNN discriminator
        """

        inputs = [tf.nn.dropout(input_, keep_prob) for input_ in inputs]

        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size_d, forget_bias=1.0, state_is_tuple=True)
        print(cell_fw, msg)

        # cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range( FLAGS.num_layers_d)], state_is_tuple=True)
        self._initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        d_state_fw = self._initial_state_fw
        outputs = []
        if not UNIDIRECTIONAL_D:
            self._initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
            outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
                                                                                  initial_state_fw=
                                                                                  self._initial_state_fw,
                                                                                  initial_state_bw=
                                                                                  self._initial_state_bw)

        else:
            for i, input_ in enumerate(inputs):
                output, d_state_fw = cell_fw(input_, d_state_fw)
                outputs.append(output)

        decisions = [tf.sigmoid(linear(output, 1, 'decision', reuse_scope=(i != 0))) for i, output in
                     enumerate(outputs)]
        decisions = tf.stack(decisions)
        decisions = tf.transpose(decisions, perm=[1, 0, 2])
        print('shape, decisions: {}'.format(decisions.get_shape()))
        decision = tf.reduce_mean(decisions, reduction_indices=[1, 2])
        decision = tf.Print(decision, [decision], '{} decision = '.format(msg), first_n=20)
        return decision, tf.transpose(tf.stack(outputs), perm=[1, 0, 2])

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def generated_features(self):
        return self._generated_features

    @property
    def input_songdata(self):
        return self._input_songdata

    @property
    def input_metadata(self):
        return self._input_metadata

    @property
    def input_metadata_wrong(self):
        return self._input_metadata_wrong

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr


def run_epoch(session, model, train, validate, test, datasetlabel, eval_op_d, eval_op_g, pretraining=False,
              verbose=False, run_metadata=None, pretraining_d=True):
    """
    Runs the model on the given data.
    """

    start = time.time()
    # run your code
    pointer = 0
    g_loss, d_loss = 10.0, 10.0
    g_losses, d_losses = 0.0, 0.0
    iters = 0

    # Randomize song order
    np.random.shuffle(train)
    np.random.shuffle(validate)

    if MULTI and CONDITION:
        [batch_song, batch_meta, batch_meta_wrong, pointer] = utils.get_batch_multi(BATCH_SIZE, pointer, train,
                                                                                    validate, test, NUM_MIDI_FEATURES,
                                                                                    SONGLENGTH,
                                                                                    part=datasetlabel)
    elif CONDITION:
        [batch_song, batch_meta, batch_meta_wrong, pointer] = utils.get_batch(BATCH_SIZE, pointer, train, validate,
                                                                              test, NUM_MIDI_FEATURES,
                                                                              SONGLENGTH,
                                                                              NUM_SYLLABLE_FEATURES,
                                                                              part=datasetlabel)
    else:
        [batch_song, pointer] = utils.get_batch_no_cond(BATCH_SIZE, pointer, train, validate,
                                                        test, NUM_MIDI_FEATURES,
                                                        SONGLENGTH,
                                                        NUM_SYLLABLE_FEATURES,
                                                        part=datasetlabel)

    # invert the condition to revert the labels
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    while batch_song is not None:
        op_g = eval_op_g
        op_d = eval_op_d

        if pretraining:
            if pretraining_d:
                fetches = [model.rnn_pretraining_loss, model.d_loss, op_g, op_d]
            else:
                fetches = [model.rnn_pretraining_loss, tf.no_op(), op_g, op_d]
        elif not FEATURE_MATCHING:
            fetches = [model.g_loss, model.d_loss, op_g, op_d]
        else:
            fetches = [model.g_loss_feature_matching, model.d_loss, op_g, op_d]

        feed_dict = {}
        feed_dict[model.input_songdata.name] = batch_song
        feed_dict[model.keep_prob.name] = DROPOUT_KEEP_PROB
        if CONDITION:
            feed_dict[model.input_metadata.name] = batch_meta
            feed_dict[model.input_metadata_wrong.name] = batch_meta_wrong
        g_loss, d_loss, _, _ = session.run(fetches, feed_dict, options=run_options, run_metadata=run_metadata)
        g_losses += g_loss
        if not pretraining or pretraining_d:
            d_losses += d_loss
        iters += 1

        if verbose and iters % 10 == 9:
            # avg_time_batchreading = float(sum(times_in_batchreading))/float(len(times_in_batchreading))
            if pretraining:
                print(
                    "{}: {} (pretraining) batch loss: G: {:.3f}, avg loss: G: {:.3f}".format(
                        datasetlabel, iters, g_loss, float(g_losses) / float(iters)))
            else:
                print(
                    "{}: {} batch loss: G: {:.3f}, D: {:.3f}, avg loss: G: {:.3f}, D: {:.3f}".format(
                        datasetlabel, iters, g_loss, d_loss, float(g_losses) / float(iters),
                        float(d_losses) / float(iters)))
        # batchtime = time.time()
        if MULTI and CONDITION:
            [batch_song, batch_meta, batch_meta_wrong, pointer] = utils.get_batch_multi(BATCH_SIZE, pointer, train,
                                                                                        validate, test,
                                                                                        NUM_MIDI_FEATURES,
                                                                                        SONGLENGTH,
                                                                                        part=datasetlabel)
        elif CONDITION:
            [batch_song, batch_meta, batch_meta_wrong, pointer] = utils.get_batch(BATCH_SIZE, pointer, train,
                                                                                  validate, test, NUM_MIDI_FEATURES,
                                                                                  SONGLENGTH,
                                                                                  NUM_SYLLABLE_FEATURES,
                                                                                  part=datasetlabel)
        else:
            [batch_song, pointer] = utils.get_batch_no_cond(BATCH_SIZE, pointer, train, validate,
                                                            test, NUM_MIDI_FEATURES,
                                                            SONGLENGTH,
                                                            NUM_SYLLABLE_FEATURES,
                                                            part=datasetlabel)

    if iters == 0:
        return (None, None)
    end = time.time()
    print("Epoch time = ", end-start)
    g_mean_loss = g_losses / iters
    if pretraining and not pretraining_d:
        d_mean_loss = None
    else:
        d_mean_loss = d_losses / iters
    return g_mean_loss, d_mean_loss


def main():
    """
    Main function
    """

    '''
    Initialization: Loading and split data
    '''
    date_time = str(datetime.datetime.now())

    # create save directory if SAVE_DIR is True (see settings file)
    settings_file = vars(parser.parse_args())
    num_songs = NUM_SONGS

    # Loading data

    train = np.load(TRAIN_DATA_MATRIX)
    validate = np.load(VALIDATE_DATA_MATRIX)
    test = np.load(TEST_DATA_MATRIX)
    
    if CONDITION is False:
        train = train[:, 0:SONGLENGTH*NUM_MIDI_FEATURES]
        validate = validate[:, 0:SONGLENGTH*NUM_MIDI_FEATURES]
        test = test[:, 0:SONGLENGTH*NUM_MIDI_FEATURES]
         
    print("Training set: ", np.shape(train)[0], " songs, Validation set: ", np.shape(validate)[0], " songs, "
          "Test set: ", np.shape(test)[0], " songs.")

    # Epoch counter initialization
    global_step = 0

    # empty lists for saving loss values at the end of each epoch
    train_g_loss_output = []
    train_d_loss_output = []
    valid_g_loss_output = []
    valid_d_loss_output = []

    '''
    Training model: Train the model and output an example midi file at each epoch
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    mmd_pitch_list = []
    mmd_duration_list = []
    mmd_rest_list = []
    MMD_pitch_old = np.inf
    MMD_duration_old = np.inf
    MMD_rest_old = np.inf
    MMD_overall_old = np.inf

    num_good_songs_best = 0
    best_epoch = 0
    print(tf.__version__)

    with tf.Graph().as_default(), tf.Session(config=config) as session:
        with tf.variable_scope("model", reuse=None) as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=REG_SCALE))
            if MULTI and CONDITION is True:
                # Multi conditioning: only the syllable embedding corresponding to the current midi note is fed into the
                # GAN
                m = RNNGAN(is_training=True, num_song_features=NUM_MIDI_FEATURES,
                           num_meta_features=NUM_SYLLABLE_FEATURES, conditioning='multi')
            elif CONDITION is False:
                # No conditioning: unconditional GAN, doesn't care about syllable information
                m = RNNGAN(is_training=True, num_song_features=NUM_MIDI_FEATURES,
                           num_meta_features=NUM_SYLLABLE_FEATURES, conditioning='none')
            else:
                # Unique conditioning: The syllable embeddings for the whole song sequence are appended and fed together
                # in each LSTM cell of the GAN
                m = RNNGAN(is_training=True, num_song_features=NUM_MIDI_FEATURES,
                           num_meta_features=NUM_SYLLABLE_FEATURES, conditioning='unique')

        session.run(tf.initialize_all_variables())
        saver = tf.train.Saver()

        print("global step = ", global_step, "max epoch = ", MAX_EPOCH)

        model_stats_saved = []

        for i in range(global_step, MAX_EPOCH):

            # Update learning rate of the generator network after EPOCHS_BEFORE_DECAY epochs if Adam optimizer is used
            lr_decay = LR_DECAY**max(i - EPOCHS_BEFORE_DECAY, 0.0)
            if not ADAM:
                m.assign_lr(session, LEARNING_RATE * lr_decay)

            print("Epoch: {} Learning rate: {:.3f}, pretraining: {}".format(i, session.run(m.lr),
                                                                            i < PRETRAINING_EPOCHS))

            if i < PRETRAINING_EPOCHS:

                '''
                Pre-training: The Generator is trained with a different loss function and discriminator can be not 
                pre-trained depending on PRETRAINING_D
                '''
                opt_d = tf.no_op()
                if PRETRAINING_D:
                    opt_d = m.opt_d

                # One pre-training epoch
                train_g_loss, train_d_loss = run_epoch(session, m, train, validate, test, 'train', opt_d,
                                                       m.opt_pretraining,
                                                       pretraining=True, verbose=True,
                                                       pretraining_d=PRETRAINING_D)
                if PRETRAINING_D:
                    try:
                        print("Epoch: {} Pretraining loss: G: {:.3f}, D: {:.3f}".format(i, train_g_loss, train_d_loss))
                    except:
                        print(train_g_loss)
                        print(train_d_loss)
                else:
                    print("Epoch: {} Pretraining loss: G: {}".format(i, train_g_loss))

            else:

                '''
                Training: Training both the Generator and the Discriminator for one epoch using the hyper-parameters 
                given in the settings file
                '''
                train_g_loss, train_d_loss = run_epoch(session, m, train, validate, test, 'train', m.opt_d, m.opt_g,
                                                       verbose=True)
                try:
                    print("Epoch: {} Train loss: G: {:.3f}, D: {:.3f}".format(i, train_g_loss, train_d_loss))
                except:
                    print("Epoch: {} Train loss: G: {}, D: {}".format(i, train_g_loss, train_d_loss))

            '''
            Validation: Calculate both Generator and Discriminator losses on the Validation set.
            '''
            valid_g_loss, valid_d_loss = run_epoch(session, m, train, validate, test, 'validation', tf.no_op(),
                                                   tf.no_op())
            try:
                print("Epoch: {} Valid loss: G: {:.3f}, D: {:.3f}".format(i, valid_g_loss, valid_d_loss))
            except:
                print("Epoch: {} Valid loss: G: {}, D: {}".format(i, valid_g_loss, valid_d_loss))

            if train_d_loss is None:  # pretraining
                train_d_loss = 0.0
                valid_d_loss = 0.0
                valid_g_loss = 0.0

            # Saving models each fifth epoch
            if i % 5 == 0:
                print("Saving model")
                try:
                    shutil.rmtree("./saved_gan_models/epoch_models/model_epoch" + str(i))
                except:
                    a = 0
                output = tf.convert_to_tensor(m.generated_features, name="output_midi", dtype=tf.float32)
                builder = tf.saved_model.builder.SavedModelBuilder("./saved_gan_models/epoch_models/model_epoch" + str(i))
                if CONDITION:
                    builder.add_meta_graph_and_variables(session, [], signature_def_map={
                        "output": tf.saved_model.signature_def_utils.predict_signature_def(
                            inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata,
                                    m.input_metadata.name: m.input_metadata},
                            outputs={"output_midi": output})
                    })
                else:
                    builder.add_meta_graph_and_variables(session, [], signature_def_map={
                        "output": tf.saved_model.signature_def_utils.predict_signature_def(
                            inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata},
                            outputs={"output_midi": output})
                    })
                builder.save()
                # np.save('./saved_gan_models/epoch_models/model_epoch' + str(i) + '/train_data.npy', train)
                # np.save('./saved_gan_models/epoch_models/model_epoch' + str(i) + '/valid_data.npy', validate)
                # np.save('./saved_gan_models/epoch_models/model_epoch' + str(i) + '/test_data.npy', test)
                # copyfile('./settings/' + settings_file['settings_file'] + '.txt',
                #          './saved_gan_models/epoch_models/model_epoch' + str(i) + '/settings.txt')

            '''
            Generating Music : samples from the generative model. If INPUT_VECTOR, then takes the syllables embeddings 
            from an input vector, otherwise takes a random lyrics embedding in the test data    
            '''
            fetches = [m.generated_features]
            feed_dict = {}
            feed_dict[m.input_songdata] = np.random.uniform(size=(1, SONGLENGTH, NUM_MIDI_FEATURES))
            feed_dict[m.keep_prob.name] = 1.0
            if CONDITION:
                condition = []
                if not MULTI:
                    condition.append(test[0][NUM_MIDI_FEATURES * SONGLENGTH:])
                else:
                    condition.append(np.split(test[0][NUM_MIDI_FEATURES * SONGLENGTH:], SONGLENGTH))
                feed_dict[m.input_metadata.name] = condition

            print("EXAMPLE DATASET SONG MIDI PARAMETERS=============================================================\n")
            example_song = []
            np.set_printoptions(threshold=np.inf)
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=3)
            for iters in range(SONGLENGTH):
                example_song.append(test[0][NUM_MIDI_FEATURES*iters:NUM_MIDI_FEATURES*iters+NUM_MIDI_FEATURES])
            print(example_song)
            if NUM_MIDI_FEATURES == 3:
                stats = midi_statistics.get_all_stats(example_song)
                midi_statistics.print_stats(stats)

            print("*************************************************************************************************\n")

            print("AVERAGE STATS OVER {} GENERATED SONGS==========================================================\n"
                  .format(np.shape(test)[0]))

            if NUM_MIDI_FEATURES == 3:
                model_stats = {}
                model_stats['stats_scale_tot'] = 0
                model_stats['stats_repetitions_2_tot'] = 0
                model_stats['stats_repetitions_3_tot'] = 0
                model_stats['stats_span_tot'] = 0
                model_stats['stats_unique_tones_tot'] = 0
                model_stats['stats_avg_rest_tot'] = 0
                model_stats['num_of_null_rest_tot'] = 0
                model_stats['best_scale_score'] = 0
                model_stats['best_repetitions_2'] = 0
                model_stats['best_repetitions_3'] = 0
                model_stats['num_perfect_scale'] = 0
                model_stats['num_good_songs'] = 0

            feed_dict = {}
            feed_dict[m.keep_prob.name] = 1.0
            validation_songs = []
            for j in range(0, np.shape(validate)[0]):
                feed_dict[m.input_songdata] = np.random.uniform(size=(1, SONGLENGTH, NUM_MIDI_FEATURES))
                if CONDITION:
                    condition = []
                    if MULTI:
                        condition.append(np.split(validate[j][NUM_MIDI_FEATURES * SONGLENGTH:], SONGLENGTH))
                    else:
                        condition.append(validate[j][NUM_MIDI_FEATURES * SONGLENGTH:])
                    feed_dict[m.input_metadata.name] = condition
                generated_features, = session.run([m.generated_features], feed_dict)
                sample = [x[0, :] for x in generated_features]
                if NUM_MIDI_FEATURES == 3:
                    discretized_sample = utils.discretize(sample)
                    discretized_sample = midi_statistics.tune_song(discretized_sample)
                    if j < 3:
                        print(discretized_sample)
                        stats = midi_statistics.get_all_stats(discretized_sample)
                        midi_statistics.print_stats(stats)

                    validation_songs.append(discretized_sample)

                    # statistics to save
                    stats = midi_statistics.get_all_stats(discretized_sample)
                    model_stats['stats_scale_tot'] += stats['scale_score']
                    model_stats['stats_repetitions_2_tot'] += float(stats['repetitions_2'])
                    model_stats['stats_repetitions_3_tot'] += float(stats['repetitions_3'])
                    model_stats['stats_unique_tones_tot'] += float(stats['tones_unique'])
                    model_stats['stats_span_tot'] += stats['tone_span']
                    model_stats['stats_avg_rest_tot'] += stats['average_rest']
                    model_stats['num_of_null_rest_tot'] += stats['num_null_rest']
                    model_stats['best_scale_score'] = max(stats['scale_score'], model_stats['best_scale_score'])
                    model_stats['best_repetitions_2'] = max(stats['repetitions_2'], model_stats['best_repetitions_2'])
                    model_stats['best_repetitions_3'] = max(stats['repetitions_3'], model_stats['best_repetitions_3'])

                    # if stats['scale_score'] == 1.0:
                    #    model_stats['num_perfect_scale'] += 1

                    if stats['scale_score'] == 1.0 and stats['tones_unique'] > 3 \
                       and stats['tone_span'] > 4 and stats['num_null_rest'] > 8 and stats['tone_span'] < 13\
                       and stats['repetitions_2'] > 4:
                        model_stats['num_good_songs'] += 1
                elif NUM_MIDI_FEATURES ==1:
                    if ATTRIBUTE == 'length':
                        discretized_sample = utils.discretize_length(sample)
                        validation_songs.append(discretized_sample)
                    if ATTRIBUTE == 'pitch':
                        discretized_sample = utils.discretize_pitch(sample)
                        validation_songs.append(discretized_sample)
                    if ATTRIBUTE == 'rest':
                        discretized_sample = utils.discretize_rest(sample)
                        validation_songs.append(discretized_sample)
            print(validation_songs[0])
            print(validation_songs[1])

            if NUM_MIDI_FEATURES == 3:
                #utils.print_model_stats(model_stats, np.shape(validate)[0])
                # writer = tf.summary.FileWriter("output", session.graph)
                model_stats_saved.append(model_stats)
                if model_stats['num_good_songs'] > num_good_songs_best:
                    try:
                        shutil.rmtree("./saved_gan_models/saved_model")
                    except:
                        a = 0
                    output = tf.convert_to_tensor(m.generated_features, name="output_midi", dtype=tf.float32)
                    builder = tf.saved_model.builder.SavedModelBuilder("./saved_gan_models/saved_model")
                    if CONDITION:
                        builder.add_meta_graph_and_variables(session, [], signature_def_map={
                            "output": tf.saved_model.signature_def_utils.predict_signature_def(
                                inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata,
                                        m.input_metadata.name: m.input_metadata},
                                outputs={"output_midi": output})
                        })
                    else:
                        builder.add_meta_graph_and_variables(session, [], signature_def_map={
                            "output": tf.saved_model.signature_def_utils.predict_signature_def(
                                inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata},
                                outputs={"output_midi": output})
                        })

                    print('NEW MODEL SAVED!\n')
                    builder.save()
                    best_epoch = i
                    # np.save('saved_model/train_data.npy', train)
                    # np.save('saved_model/valid_data.npy', validate)
                    # np.save('saved_model/test_data.npy', test)
                    num_good_songs_best = model_stats['num_good_songs']

                print('Best ratio of good songs, ', num_good_songs_best, ' at epoch', best_epoch)

                print("MMD2=========================================================================================\n")
                val_gen_pitches = np.zeros((np.shape(validation_songs)[0],SONGLENGTH))
                val_dat_pitches = np.zeros((np.shape(validate)[0],SONGLENGTH))
                val_gen_duration = np.zeros((np.shape(validation_songs)[0],SONGLENGTH))
                val_dat_duration = np.zeros((np.shape(validate)[0],SONGLENGTH))
                val_gen_rests = np.zeros((np.shape(validation_songs)[0],SONGLENGTH))
                val_dat_rests = np.zeros((np.shape(validate)[0],SONGLENGTH))

                print(np.shape(validation_songs), np.shape(val_gen_pitches))

                for i in range(SONGLENGTH):
                    val_gen_pitches[:, i] = np.array(validation_songs)[:, i, 0]
                    val_gen_duration[:, i] = np.array(validation_songs)[:, i, 1]
                    val_gen_rests[:, i] = np.array(validation_songs)[:, i, 2]
                    val_dat_pitches[:, i] = np.array(validate)[:, NUM_MIDI_FEATURES * i]
                    val_dat_duration[:, i] = np.array(validate)[:, NUM_MIDI_FEATURES * i + 1]
                    val_dat_rests[:, i] = np.array(validate)[:, NUM_MIDI_FEATURES * i + 2]

                MMD_pitch = mmd.Compute_MMD(val_gen_pitches, val_dat_pitches)
                print("MMD pitch:", MMD_pitch)
                if MMD_pitch < MMD_pitch_old:
                    print("New lowest value of MMD for pitch", MMD_pitch)
                    try:
                        shutil.rmtree("./saved_gan_models/saved_model_best_pitch_mmd")
                    except:
                        a = 0
                    output = tf.convert_to_tensor(m.generated_features, name="output_midi", dtype=tf.float32)
                    builder = tf.saved_model.builder.SavedModelBuilder("./saved_gan_models/saved_model_best_pitch_mmd")
                    builder.add_meta_graph_and_variables(session, [], signature_def_map={
                        "output": tf.saved_model.signature_def_utils.predict_signature_def(
                            inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata,
                                    m.input_metadata.name: m.input_metadata},
                            outputs={"output_midi": output})
                    })
                    MMD_pitch_old = MMD_pitch
                    builder.save()
                mmd_pitch_list.append(MMD_pitch)

                MMD_duration = mmd.Compute_MMD(val_gen_duration,val_dat_duration)
                print("MMD duration:", MMD_duration)
                if MMD_duration < MMD_duration_old:
                    print("New lowest value of MMD for duration", MMD_duration)
                    try:
                        shutil.rmtree("./saved_gan_models/saved_model_best_duration_mmd")
                    except:
                        a = 0
                    output = tf.convert_to_tensor(m.generated_features, name="output_midi", dtype=tf.float32)
                    builder = tf.saved_model.builder.SavedModelBuilder("./saved_gan_models/saved_model_best_duration_mmd")
                    builder.add_meta_graph_and_variables(session, [], signature_def_map={
                        "output": tf.saved_model.signature_def_utils.predict_signature_def(
                            inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata,
                                    m.input_metadata.name: m.input_metadata},
                            outputs={"output_midi": output})
                    })
                    MMD_duration_old = MMD_duration
                    builder.save()
                mmd_duration_list.append(MMD_duration)

                MMD_rest = mmd.Compute_MMD(val_gen_rests,val_dat_rests)
                print("MMD rest:", MMD_rest)
                if MMD_rest < MMD_rest_old:
                    print("New lowest value of MMD for rest", MMD_rest)
                    try:
                        shutil.rmtree("./saved_gan_models/saved_model_best_rest_mmd")
                    except:
                        a = 0
                    output = tf.convert_to_tensor(m.generated_features, name="output_midi", dtype=tf.float32)
                    builder = tf.saved_model.builder.SavedModelBuilder("./saved_gan_models/saved_model_best_rest_mmd")
                    builder.add_meta_graph_and_variables(session, [], signature_def_map={
                        "output": tf.saved_model.signature_def_utils.predict_signature_def(
                            inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata,
                                    m.input_metadata.name: m.input_metadata},
                            outputs={"output_midi": output})
                    })
                    MMD_rest_old = MMD_rest
                    builder.save()
                mmd_rest_list.append(MMD_rest)

                MMD_overall = MMD_rest + MMD_duration + MMD_pitch
                print("MMD overall:", MMD_overall)
                if MMD_overall < MMD_overall_old:
                    print("New lowest value of MMD for overall", MMD_overall)
                    try:
                        shutil.rmtree("./saved_gan_models/saved_model_best_overall_mmd")
                    except:
                        a = 0
                    output = tf.convert_to_tensor(m.generated_features, name="output_midi", dtype=tf.float32)
                    builder = tf.saved_model.builder.SavedModelBuilder("./saved_gan_models/saved_model_best_overall_mmd")
                    builder.add_meta_graph_and_variables(session, [], signature_def_map={
                        "output": tf.saved_model.signature_def_utils.predict_signature_def(
                            inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata,
                                    m.input_metadata.name: m.input_metadata},
                            outputs={"output_midi": output})
                    })
                    MMD_overall_old = MMD_overall
                    builder.save()

            elif NUM_MIDI_FEATURES == 1:
                if ATTRIBUTE=='length':
                    val_gen_duration = np.zeros((np.shape(validation_songs)[0],SONGLENGTH))
                    val_dat_duration = np.zeros((np.shape(validate)[0],SONGLENGTH))
                    for i in range(SONGLENGTH):
                        val_gen_duration[:, i] = np.array(validation_songs)[:, i, 0]
                        val_dat_duration[:, i] = np.array(validate)[:, NUM_MIDI_FEATURES * i]
                    print(val_gen_duration[3])

                    MMD_duration = mmd.Compute_MMD(val_gen_duration, val_dat_duration)
                    print("MMD duration:", MMD_duration)
                    if MMD_duration < MMD_duration_old:
                        print("New lowest value of MMD for duration", MMD_duration)
                        try:
                            shutil.rmtree("./saved_gan_models/saved_model_best_duration_mmd")
                        except:
                            a = 0
                        output = tf.convert_to_tensor(m.generated_features, name="output_midi", dtype=tf.float32)
                        builder = tf.saved_model.builder.SavedModelBuilder("./saved_gan_models/saved_model_best_duration_mmd")
                        builder.add_meta_graph_and_variables(session, [], signature_def_map={
                            "output": tf.saved_model.signature_def_utils.predict_signature_def(
                                inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata,
                                        m.input_metadata.name: m.input_metadata},
                                outputs={"output_midi": output})
                        })
                        MMD_duration_old = MMD_duration
                        builder.save()
                    mmd_duration_list.append(MMD_duration)
                if ATTRIBUTE=='rest':
                    val_gen_rests = np.zeros((np.shape(validation_songs)[0], SONGLENGTH))
                    val_dat_rests = np.zeros((np.shape(validate)[0], SONGLENGTH))
                    for i in range(SONGLENGTH):
                        val_gen_rests[:, i] = np.array(validation_songs)[:, i, 0]
                        val_dat_rests[:, i] = np.array(validate)[:, NUM_MIDI_FEATURES * i]
                    print(val_gen_rests[3])
                    MMD_rest = mmd.Compute_MMD(val_gen_rests, val_dat_rests)
                    print("MMD rest:", MMD_rest)
                    if MMD_rest < MMD_rest_old:
                        print("New lowest value of MMD for rest", MMD_rest)
                        try:
                            shutil.rmtree("./saved_gan_models/saved_model_best_rest_mmd")
                        except:
                            a = 0
                        output = tf.convert_to_tensor(m.generated_features, name="output_midi", dtype=tf.float32)
                        builder = tf.saved_model.builder.SavedModelBuilder("./saved_gan_models/saved_model_best_rest_mmd")
                        builder.add_meta_graph_and_variables(session, [], signature_def_map={
                            "output": tf.saved_model.signature_def_utils.predict_signature_def(
                                inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata,
                                        m.input_metadata.name: m.input_metadata},
                                outputs={"output_midi": output})
                        })
                        MMD_rest_old = MMD_rest
                        builder.save()
                    mmd_rest_list.append(MMD_rest)
                if ATTRIBUTE == 'pitch':
                    val_gen_pitches = np.zeros((np.shape(validation_songs)[0], SONGLENGTH))
                    val_dat_pitches = np.zeros((np.shape(validate)[0], SONGLENGTH))
                    for i in range(SONGLENGTH):
                        val_gen_pitches[:, i] = np.array(validation_songs)[:, i, 0]
                        val_dat_pitches[:, i] = np.array(validate)[:, NUM_MIDI_FEATURES * i]
                    print(val_gen_pitches[3])
                    MMD_rest = mmd.Compute_MMD(val_gen_rests, val_dat_rests)
                    print("MMD rest:", MMD_rest)
                    if MMD_rest < MMD_rest_old:
                        print("New lowest value of MMD for rest", MMD_rest)
                        try:
                            shutil.rmtree("./saved_gan_models/saved_model_best_pitch_mmd")
                        except:
                            a = 0
                        output = tf.convert_to_tensor(m.generated_features, name="output_midi", dtype=tf.float32)
                        builder = tf.saved_model.builder.SavedModelBuilder("./saved_gan_models/saved_model_best_pitch_mmd")
                        builder.add_meta_graph_and_variables(session, [], signature_def_map={
                            "output": tf.saved_model.signature_def_utils.predict_signature_def(
                                inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata,
                                        m.input_metadata.name: m.input_metadata},
                                outputs={"output_midi": output})
                        })
                        MMD_rest_old = MMD_rest
                        builder.save()
                    mmd_rest_list.append(MMD_rest)

            print("*************************************************************************************************\n")
            print("*************************************************************************************************\n")
            print("*************************************************************************************************\n")


        # Save model
        try:
            shutil.rmtree("./saved_gan_models/saved_model_end_of_training")
        except:
            a = 0
        output = tf.convert_to_tensor(m.generated_features, name="output_midi", dtype=tf.float32)
        builder = tf.saved_model.builder.SavedModelBuilder("./saved_gan_models/saved_model_end_of_training")
        if CONDITION:
            builder.add_meta_graph_and_variables(session, [], signature_def_map={
                "output": tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata,
                                               m.input_metadata.name: m.input_metadata},
                    outputs={"output_midi": output})
            })
        else:
            builder.add_meta_graph_and_variables(session, [], signature_def_map={
                "output": tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs={m.keep_prob.name: m.keep_prob, m.input_songdata.name: m.input_songdata},
                    outputs={"output_midi": output})
            })

        builder.save()
        # np.save('./saved_gan_models/saved_model_end_of_training/train_data.npy', train)
        # np.save('./saved_gan_models/saved_model_end_of_training/valid_data.npy', validate)
        # np.save('./saved_gan_models/saved_model_end_of_training/test_data.npy', test)

        # np.save('./saved_gan_models/saved_model_end_of_training/MMD_pitch_list.npy', mmd_pitch_list)
        # np.save('./saved_gan_models/saved_model_end_of_training/MMD_dur_list.npy',   mmd_duration_list)
        # np.save('./saved_gan_models/saved_model_end_of_training/MMD_rest_list.npy',  mmd_rest_list)

        # copyfile('./settings/' + settings_file['settings_file'] + '.txt',
        #          './saved_gan_models/saved_model_end_of_training/settings.txt')

    return 0


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.ERROR)
    parser = argparse.ArgumentParser(description='Train a GAN to generate sequential, real-valued data.')
    parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str,
                        default='settings')
    settings = vars(parser.parse_args())
    if settings['settings_file']:
        settings = utils.load_settings_from_file(settings)
    for (k, v) in settings.items():
        print(v, '\t', k)
    locals().update(settings)
    main()
