#!/usr/bin/env python
from __future__ import absolute_import, print_function, division
from builtins import range

from datetime import datetime
from timeit import default_timer
from pkg_resources import resource_filename
import numpy as np
import tensorflow as tf
import h5py

from ..nn import NeuralNetworkModel, default_runstep, get_variable
from .phoneme import index2phoneme_dict


def load_timit(train=True, n_context=3, synthesized_data=False):
    # NOTE: avoid dependency on preproc
    filepath = '/data/speech/timit/timit.hdf5'
    if synthesized_data:
        filepath = resource_filename('fathom', 'data/syn_timit.hdf5')

    # TODO: load test also
    with h5py.File(filepath, 'r') as hf:
        train_spectrograms = np.array(hf['timit']['train']['spectrograms'])
        train_labels = np.array(hf['timit']['train']['labels'])
        train_seq_lens = np.array(hf['timit']['train']['seq_lens'])
        train_frame_lens = np.array(hf['timit']['train']['frame_lens'])

        return train_spectrograms, train_frame_lens, train_labels, train_seq_lens


def clipped_relu(inputs, clip=20):
    """Similar to tf.nn.relu6, but can clip at 20 as in Deep Speech."""
    return tf.minimum(tf.nn.relu(inputs), clip)


# TODO: show label error rate
# TODO: avoid labels and blank off-by-one error due to padding zeros
class Speech(NeuralNetworkModel):
    """RNN for speech recognition."""

    def __init__(self, device=None, init_options=None):
        super(Speech, self).__init__(device=device, init_options=init_options)

    def build_hyperparameters(self):
        self.n_labels = 61 + 1  # add blank

        self.n_coeffs = 26
        self.n_context = 0  # TODO: enable n_context = 3 in preproc.py

        self.batch_size = 32
        if self.init_options:
            self.batch_size = self.init_options.get('batch_size', self.batch_size)

        self._n_inputs = self.n_coeffs + 2 * self.n_coeffs * self.n_context

    def build_inputs(self):
        with self.G.as_default():
            # of shape [batch_size, <=max_frames, n_coeffs + 2 * n_coeffs * n_context]
            self._inputs = tf.placeholder(tf.float32, [None, None, self._n_inputs], name="inputs")
            self.frame_lens = tf.placeholder(tf.int32, [None], name="frame_lens")

    @property
    def inputs(self):
        return self._inputs

    def build_labels(self):
        with self.G.as_default():
            # of shape [batch_size, <=max_labels]
            self._labels = tf.placeholder(tf.int32, [None, None], name="labels")
            self.seq_lens = tf.placeholder(tf.int32, [None], name="seq_lens")

    @property
    def labels(self):
        return self._labels

    def build_inference(self, inputs, n_hidden=1024):
        with self.G.as_default():
            self.n_hidden = n_hidden

            # Architecture of Deep Speech [Hannun et al. 2014]
            # Adapted from mozilla/DeepSpeech code, without dropouts

            # Input shape: [batch_size, n_frames, _n_inputs]
            # Reshaping `inputs` to a tensor with shape `[n_frames*batch_size, _n_inputs]`.
            # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

            # Permute n_frames and batch_size
            inputs = tf.transpose(inputs, [1, 0, 2])
            # Reshape to prepare input for first layer
            # (n_frames*batch_size, _n_inputs)
            inputs = tf.reshape(inputs, [-1, self._n_inputs])

            # The next three blocks will pass `inputs` through three hidden layers with clipped RELU activation.
            # 1st layer
            layer_1 = self.mlp_layer(inputs, self.n_hidden, 'layer_1',
                                     weight_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            # 2nd layer
            layer_2 = self.mlp_layer(layer_1, self.n_hidden, 'layer_2')
            # 3rd layer
            layer_3 = self.mlp_layer(layer_2, 2 * self.n_hidden, 'layer_3')

            # BiRNN layer
            layer_4 = self.bidirectional_layer(layer_3, self.n_hidden)

            # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation
            layer_5 = self.mlp_layer(layer_4, self.n_hidden, 'layer_5')

            # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
            # creating `n_labels` dimensional vectors, the logits.
            layer_6 = self.mlp_layer(layer_5, self.n_labels, 'layer_6',
                                     weight_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                     activation=None)

            # Finally we reshape layer_6 from a tensor of shape [n_frames*batch_size, n_labels]
            # to the slightly more useful shape [n_frames, batch_size, n_labels].
            # Note, that this differs from the input in that it is time-major.
            layer_6 = tf.reshape(layer_6, [-1, self.batch_size, self.n_labels], name="logits")

            # Output shape: [n_frames, batch_size, n_hidden_6]
            self._outputs = layer_6
            return layer_6

    @property
    def outputs(self):
        return self._outputs

    def build_loss(self, logits, labels):
        with self.G.as_default():
            # NOTE: CTC does the softmax for us, according to the code

            # CTC loss requires sparse labels
            sparse_labels = self.ctc_label_dense_to_sparse(self.labels, self.seq_lens)

            # CTC
            self.loss_op = tf.nn.ctc_loss(inputs=self.outputs, labels=sparse_labels,
                                          sequence_length=self.seq_lens)

            return self.loss_op

    @property
    def loss(self):
        return self.loss_op

    def build_train(self, loss):
        with self.G.as_default():
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
                                               beta1=0.9,
                                               beta2=0.999,
                                               epsilon=1e-8)
            self.train_op = optimizer.minimize(loss)
            return self.train_op

    @property
    def train(self):
        return self.train_op

    def build_decoding(self):
        """Predict labels from learned sequence model."""
        # TODO: label error rate on validation set
        decoded, _ = tf.nn.ctc_greedy_decoder(self.outputs, self.seq_lens)
        sparse_decode_op = decoded[0]  # single-element list
        self.decode_op = tf.sparse_to_dense(
            sparse_decode_op.indices, sparse_decode_op.dense_shape, sparse_decode_op.values)
        return self.decode_op

    def build(self):
        super(Speech, self).build()

        with self.G.as_default():
            self.decode_op = self.build_decoding()
            self.salus_marker = tf.no_op(name="salus_main_iter")

    def mlp_layer(self, inputs, n_output, name, weight_initializer=None, activation=clipped_relu):
        with self.G.as_default():
            inputs_shape = inputs.get_shape().as_list()
            n_input = np.prod(inputs_shape[1:])

            # reshape inputs if necessary
            if len(inputs_shape) > 2:
                inputs = tf.reshape(inputs, [-1, n_input])

            if weight_initializer is None:
                weight_initializer = tf.random_normal_initializer()

            B = get_variable(name + '_B', [n_output], tf.random_normal_initializer())
            W = get_variable(name + '_W', [n_input, n_output], weight_initializer)
            outputs = tf.add(tf.matmul(inputs, W), B)
            if activation is not None:
                outputs = activation(outputs)
            return outputs

    def bidirectional_layer(self, inputs, n_cell_dim):
        """Bidirectional RNN layer."""
        with self.G.as_default():
            # Now we create the forward and backward LSTM units.
            # Both of which have inputs of length `n_hidden` and bias `1.0` for the forget gate of the LSTM.

            # Verify shapes. Inputs should be in shape `[n_frames * batch_size, 2 * n_cell_dim]`
            inputs_shape = inputs.get_shape().as_list()
            if len(inputs_shape) != 2 or inputs_shape[-1] != 2 * n_cell_dim:
                raise ValueError('BiRNN inputs should be in shape [n_frames * batch_size, 2 * n_cell_dim],'
                                 ' actually got: ' + str(inputs_shape))

            # Forward direction cell:
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(
                n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            # Backward direction cell:
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
                n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

            # `inputs` is now reshaped into `[n_frames, batch_size, 2*n_cell_dim]`,
            # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
            inputs = tf.reshape(inputs, [-1, self.batch_size, 2 * n_cell_dim])

            # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                         cell_bw=lstm_bw_cell,
                                                         inputs=inputs,
                                                         dtype=tf.float32,
                                                         time_major=True,
                                                         sequence_length=self.frame_lens)

            # Reshape outputs from two tensors each of shape [n_frames, batch_size, n_cell_dim]
            # to a single tensor of shape [n_frames*batch_size, 2*n_cell_dim]
            outputs = tf.concat(outputs, 2)
            outputs = tf.reshape(outputs, [-1, 2 * n_cell_dim])
            return outputs

    def ctc_label_dense_to_sparse(self, labels, label_lengths):
        """Mike Henry's implementation, with some minor modifications."""
        with self.G.as_default():
            label_shape = tf.shape(labels)
            num_batches_tns = tf.stack([label_shape[0]])
            max_num_labels_tns = tf.stack([label_shape[1]])

            def range_less_than(previous_state, current_input):
                return tf.expand_dims(tf.range(label_shape[1]), 0) < current_input

            init = tf.cast(tf.fill(max_num_labels_tns, 0), tf.bool)
            init = tf.expand_dims(init, 0)
            dense_mask = tf.scan(range_less_than, label_lengths, initializer=init, parallel_iterations=1)
            dense_mask = dense_mask[:, 0, :]

            label_array = tf.reshape(tf.tile(tf.range(0, label_shape[1]), num_batches_tns), label_shape)
            label_ind = tf.boolean_mask(label_array, dense_mask)

            batch_array = tf.transpose(tf.reshape(
                tf.tile(tf.range(0,  label_shape[0]), max_num_labels_tns), tf.reverse(label_shape, [0])))
            batch_ind = tf.boolean_mask(batch_array, dense_mask)

            indices = tf.transpose(tf.reshape(tf.concat(axis=0, values=[batch_ind, label_ind]), [2, -1]))
            vals_sparse = tf.gather_nd(labels, indices)
            return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64(label_shape))

    def load_data(self):
        # TODO: load test
        print('Using fake data: {}'.format(self.use_synthesized_data))
        (self.train_spectrograms, self.train_frame_lens,
         self.train_labels, self.train_seq_lens) = load_timit(train=True, n_context=self.n_context,
                                                              synthesized_data=self.use_synthesized_data)

    def get_random_batch(self):
        """Get random batch from np.arrays (not tf.train.shuffle_batch)."""
        n_examples = self.train_spectrograms.shape[0]
        random_sample = np.random.randint(n_examples, size=self.batch_size)
        batch = (self.train_spectrograms[random_sample, :, :],
                 self.train_frame_lens[random_sample],
                 self.train_labels[random_sample, :],
                 self.train_seq_lens[random_sample])
        return batch

    def setup(self, setup_options=None):
        """Make session and launch queue runners."""
        config = setup_options.pop('config', tf.ConfigProto())
        action = setup_options.pop('action', 'train')

        # set memory usage
        KB = 1024
        MB = 1024 * KB
        if action == 'train':
            memusages = {
                25: (2260 * MB - 500 * MB, 500 * MB),
                50: (4000 * MB - 500 * MB, 500 * KB),
                75: (5740 * MB - 500 * MB, 500 * MB),
            }
        elif action == 'test':
            memusages = {
                1: (145 * MB - 117 * MB, 117 * MB),
                5: (194 * MB - 500 * MB, 117 * MB),
                10: (263 * MB - 500 * MB, 117 * MB),
            }
        else:
            raise ValueError('Unknown action: ' + action)
        config.allow_soft_placement = True
        config.salus_options.resource_map.temporary['MEMORY:GPU'] = memusages[self.batch_size][0]
        config.salus_options.resource_map.persistant['MEMORY:GPU'] = memusages[self.batch_size][1]

        setup_options['config'] = config
        super(Speech, self).setup(setup_options=setup_options)

    def run(self, runstep=None, n_steps=1, *args, **kwargs):
        print("Loading spectrogram features...")
        self.load_data()

        with self.G.as_default():
            step_train_times = []
            for step in range(n_steps):
                print('Iteration {}'.format(step))
                start_time = default_timer()

                spectrogram_batch, frame_len_batch, label_batch, seq_len_batch = self.get_random_batch()
                feeds = {
                    self.inputs: spectrogram_batch,
                    self.frame_lens: frame_len_batch,
                    self.labels: label_batch,
                    self.seq_lens: seq_len_batch
                }
                lossval = 0
                if not self.forward_only:
                    _, lossval, _ = runstep(self.session, [self.train_op, self.loss_op, self.salus_marker],
                                            feed_dict=feeds)
                    lossval = lossval.mean()
                else:
                    # run forward-only on train batch
                    runstep(self.session, self.outputs, feed_dict=feeds)

                train_time = default_timer() - start_time
                step_train_times.append(train_time)
                print('{}: Step {}, loss={:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)'.format(
                    datetime.now(), step, lossval, self.batch_size / train_time, train_time))

                # print some decoded examples
                if False:
                    # decode the same batch, for debugging
                    decoded = self.session.run(self.decode_op, feed_dict=feeds)
                    print(' '.join(self.labels2phonemes(decoded[0])))
                    # TODO: fix dtypes in dataset (labels are accidentally floats right now)
                    print(' '.join(self.labels2phonemes(np.array(label_batch[0, :], dtype=np.int32))))

            print('Average: {:.3f} sec/batch'.format(np.mean(step_train_times)))
            print('First iteration: {:.3f} sec/batch'.format(step_train_times[0]))
            print('Average excluding first iteration: {:.3f} sec/batch'.format(np.mean(step_train_times[1:])))

    def labels2phonemes(self, decoded_labels):
        """Convert a list of label indices to a list of corresponding phonemes."""
        return [index2phoneme_dict[label] for label in decoded_labels]


class SpeechFwd(Speech):
    forward_only = True


if __name__ == '__main__':
    m = Speech()
    m.setup()
    m.run(runstep=default_runstep, n_steps=10)
    m.teardown()
