from __future__ import absolute_import, print_function, division

import tensorflow as tf

from . import _define_workload, _metadata
from .wrappers import prepare_speech
from ..nn import default_runstep


_define_workload('DeepQ')
_define_workload('AlexNet')
_define_workload('VGG')
_define_workload('Residual')
_define_workload('AutoencBase')
_define_workload('MemNet')
_define_workload('Seq2Seq')
_define_workload('Speech', act_prepare=prepare_speech)

tf.app.flags.DEFINE_enum("workload", None, _metadata.keys(), "Workload to run")
tf.app.flags.DEFINE_enum("action", "train", ["train", "test", "prepare"], "Action")
tf.app.flags.DEFINE_string("target", "", "Session target")
tf.app.flags.DEFINE_string("dev", "", "Device to run on")
tf.app.flags.DEFINE_integer("batch_size", None, "Batch size to use", lower_bound=1)
tf.app.flags.DEFINE_integer("num_iters", 20, "Iterations to run", lower_bound=1)
tf.app.flags.DEFINE_boolean("syn_data", True, "Wether to use synthesized data")
FLAGS = tf.app.flags.FLAGS


def build_options(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


def _run_class(creator):
    init_options = build_options(batch_size=FLAGS.batch_size, use_synthesized_data=FLAGS.syn_data)
    setup_options = build_options(target=FLAGS.target, config=tf.ConfigProto(allow_soft_placement=True))

    m = None
    retry = True
    while retry:
        try:
            m = creator(device=FLAGS.dev, init_options=init_options)
            m.setup(setup_options=setup_options)
            m.run(runstep=default_runstep, n_steps=FLAGS.num_iters)
            retry = False
        except tf.errors.UnavailableError as ex:
            time.sleep(1)
            eprint("Retry due to error: ", ex)
        finally:
            if m is not None:
                m.teardown()


def _default_train(meta):
    _run_class(meta.get_creator())


def _default_test(meta):
    _run_class(meta.get_creator(meta.name + 'Fwd'))


def _default_prepare(meta):
    pass


def main(argv):
    meta = _metadata[FLAGS.workload]

    default_actions = {
        'train': _default_train,
        'test': _default_test,
        'prepare': _default_prepare,
    }
    action = meta.actions.get(FLAGS.action, default_actions[FLAGS.action])
    if action is None:
        return

    action(meta)


if __name__ == '__main__':
    tf.app.run(main)
