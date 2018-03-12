from __future__ import absolute_import, print_function, division

import os
import tensorflow as tf


class Dataset(object):
    """Simple wrapper for a dataset.

    Inspired by David Dao's TensorFlow models code.
    """

    def __init__(self, subset, record_dir=None, synthesized=False):
        """
        record_dir: Directory with TFRecords.
        """
        self.subset = subset
        self.record_dir = record_dir
        self._synthesized = self.record_dir is None
        if self._synthesized != synthesized:
            if self._synthesized:
                raise ValueError('No record_dir provided for dataset')
            else:
                raise ValueError('record_dir and synthesized cannot be set at the same time')

    def data_files(self):
        return tf.gfile.Glob(os.path.join(self.record_dir, "{}-*".format(self.subset)))

    @property
    def is_synthesized(self):
        return self._synthesized

    def synthesize_sample(self, sample_dim):
        """Return a sample of synthesized data"""
        if not self.is_synthesized:
            raise NotImplementedError('Not a synthesized dataset')

    def record_queue(self):
        """Return a TensorFlow queue of TFRecords."""
        return tf.train.string_input_producer(self.data_files())

    def reader(self):
        return tf.TFRecordReader()
