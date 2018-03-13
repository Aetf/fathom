from __future__ import absolute_import, print_function, division


_metadata = {}


def _define_workload(classname, act_train=None, act_test=None, act_prepare=None):
    actions = {}
    for actname in ['train', 'test', 'prepare']:
        act = locals()['act_' + actname]
        if act is not None:
            actions[actname] = act
    pkgname = classname.lower()

    global _metadata
    _metadata[pkgname] = {
        'classname': 'fathom.' + pkgname + '.' + classname,
        'actions': actions
    }
