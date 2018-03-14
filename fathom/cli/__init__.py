from __future__ import absolute_import, print_function, division

from absl.app import UsageError


class Workload(object):

    def __init__(self, name=None, actions=None):
        super(Workload, self).__init__()
        self.actions = actions or {}
        self.name = name

    def get_creator(self, name=None):
        if name is None:
            name = self.name

        package = 'fathom.' + name.lower()
        try:
            pkg = __import__(package, globals(), locals(), [name], 0)
            return getattr(pkg, name)
        except ImportError as ex:
            print("Class not found: ", name)
            raise UsageError()


_metadata = {}


def _define_workload(classname, act_train=None, act_test=None, act_prepare=None):
    w = Workload(classname)
    for actname in ['train', 'test', 'prepare']:
        act = locals()['act_' + actname]
        if act is not None:
            w.actions[actname] = act

    pkgname = classname.lower()

    global _metadata
    _metadata[pkgname] = w
