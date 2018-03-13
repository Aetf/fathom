from __future__ import absolute_import, print_function, division

import importlib

import tensorflow as tf

from . import _define_workload, _metadata
from .wrappers import prepare_speech
from ..nn import default_runstep


_define_workload('DeepQ')
_define_workload('AlexNet')
_define_workload('VGG')
_define_workload('Residual')
_define_workload('Autoenc')
_define_workload('MemNet')
_define_workload('Seq2Seq')
_define_workload('Speech', act_prepare=prepare_speech)

tf.app.flags.DEFINE_enum("workload", None, _metadata.keys(), "Workload to run")
tf.app.flags.DEFINE_enum("action", "train", ["train", "test", "prepare"], "Action")
tf.app.flags.DEFINE_string("target", "", "Session target")
tf.app.flags.DEFINE_integer("batch_size", None, "Batch size to use", lower_bound=1)
tf.app.flags.DEFINE_integer("num_iters", 20, "Iterations to run", lower_bound=1)
FLAGS = tf.app.flags.FLAGS


def _load_creator(classname):
    try:
        return importlib.import_module(classname)
    except ImportError as ex:
        print("Class not found: ", classname)
        raise tf.app.UsageError()


def _run_class(creator):
    init_options = {
        'batch_size': FLAGS.batch_size
    }
    setup_options = {
        'target': FLAGS.target
    }

    m = None
    try:
        m = creator(init_options=init_options)
        m.setup(setup_options=setup_options)
        m.run(runstep=default_runstep, n_steps=FLAGS.num_iters)
    except Exception as ex:
        print('Error when running: ', ex)
    finally:
        if m is not None:
            m.teardown()


def _default_train(classname):
    workload = _load_creator(classname)
    _run_class(workload)


def _default_test(classname):
    workload = _load_creator(classname + 'Fwd')
    _run_class(workload)


def _default_prepare(classname):
    pass


def main(argv):
    meta = _metadata[FLAGS.workload]

    default_actions = {
        'train': _default_train,
        'test': _default_test,
        'prepare': _default_prepare,
    }
    action = meta['actions'].get(FLAGS.action, default_actions[FLAGS.action])
    if action is None:
        return

    action(meta)


if __name__ == '__main__':
    tf.app.run(main)
