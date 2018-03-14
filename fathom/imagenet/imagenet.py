from __future__ import absolute_import, print_function, division

from datetime import datetime
from timeit import default_timer
import tensorflow as tf
import numpy as np

from fathom.nn import NeuralNetworkModel, default_runstep
from fathom.dataset import Dataset
from fathom.imagenet.image_processing import distorted_inputs

# TODO: don't hard-code this
imagenet_record_dir = '/data/ILSVRC2012/imagenet-tfrecord/'


class Imagenet(Dataset):
    """Design from TensorFlow Inception example."""

    def __init__(self, subset, record_dir=imagenet_record_dir, synthesized=False):
        if synthesized:
            record_dir = None
        super(Imagenet, self).__init__(subset, record_dir, synthesized)

    def num_classes(self):
        return 1000

    def synthesize_sample(self, sample_dim):
        # check if is synthesized
        super(Imagenet, self).synthesize_sample(sample_dim)

        image = tf.Variable(tf.random_normal(sample_dim, dtype=tf.float32), name='sample_image', trainable=False)
        label = tf.Variable(tf.random_uniform([1], minval=0, maxval=self.num_classes(), dtype=tf.int64),
                            name='ground_truth', trainable=False)
        return image, label

    def num_examples_per_epoch(self):
        # Bounding box data consists of 615299 bounding boxes for 544546 images.
        if self.subset == 'train':
            return 1281167
        if self.subset == 'validation':
            return 50000


class ImagenetModel(NeuralNetworkModel):
    @property
    def inputs(self):
        return self.images

    @property
    def labels(self):
        return self._labels

    @property
    def outputs(self):
        return self.logits

    @property
    def loss(self):
        return self.loss_op

    @property
    def train(self):
        return self.train_op

    def build_inputs(self):
        with self.G.as_default():
            # TODO: configure image_size in image_processing.py
            self.image_size = 224  # side of the square image
            self.channels = 3
            self.n_input = self.image_size * self.image_size * self.channels
            self.dataset = Imagenet('train', synthesized=self.use_synthesized_data)

            # add queue runners (evaluation dequeues records)
            self.images, self._labels = distorted_inputs(self.dataset, batch_size=self.batch_size)

    def build_labels(self):
        with self.G.as_default():
            self.n_classes = 1000 + 1  # background class
            # self._labels already set in build_inputs

    def build_evaluation(self):
        """Evaluation metrics (e.g., accuracy)."""
        self.correct_pred = tf.equal(tf.argmax(self.outputs, 1), self.labels)  # TODO: off-by-one?
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def build_hyperparameters(self):
        with self.G.as_default():
            self.learning_rate = 0.001
            self.batch_size = 64
            self.display_step = 1

            self.dropout = 0.8  # Dropout, probability to keep units

        # TODO: can this not be a placeholder?
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    def build_loss(self, logits, labels):
        with self.G.as_default():
            # Define loss
            # TODO: does this labels have unexpected state?
            self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return self.loss_op

    def build_train(self, total_loss):
        with self.G.as_default():
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # Compute and apply gradients.
            self.train_op = opt.minimize(total_loss)

        return self.train_op

    def load_data(self):
        # Grab the dataset from the internet, if necessary
        self.num_batches_per_epoch = self.dataset.num_examples_per_epoch() / self.batch_size

    def run(self, runstep=default_runstep, n_steps=1):
        self.load_data()

        with self.G.as_default():
            # Keep training until reach max iterations
            step_train_times = []
            for step in range(n_steps):
                start_time = default_timer()
                lossval = 0
                if not self.forward_only:
                    _, lossval, acc = runstep(self.session, [self.train, self.loss, self.accuracy],
                                              feed_dict={self.keep_prob: self.dropout})
                else:
                    # TODO: switch to test subset dataset
                    runstep(self.session, self.outputs, feed_dict={self.keep_prob: 1.})

                train_time = default_timer() - start_time
                step_train_times.append(train_time)
                if step % self.display_step == 0:
                    print('{}: Step {}, loss={:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)'.format(
                          datetime.now(), step, lossval, self.batch_size / train_time, train_time))

            print('Average: {:.3f} sec/batch'.format(np.mean(step_train_times)))
            print('First iteration: {:.3f} sec/batch'.format(step_train_times[0]))
            print('Average excluding first iteration: {:.3f} sec/batch'.format(np.mean(step_train_times[1:])))
