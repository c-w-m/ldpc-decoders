import numpy as np
import argparse
import os, sys
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from misc import Registry


schedules = Registry()
register = schedules.reg
lrate_scheduler = lambda args: schedules.get(args.lrate_schedule)


class BaseSchedule:
    def __init__(self, args, total_steps=-1):
        self.args = args
        self.total_steps = total_steps

    def to_str(self):
        _a = self.args
        _e = self.to_str_(_a)
        return f'{_a.lrate_schedule}_{_a.learning_rate:g}' + ('' if _e is None else f'_{_e}')

    def __call__(self, global_step, epoch): return self.get_rate_(self.args, global_step, epoch)


@register
class const(BaseSchedule):
    @classmethod
    def bind(cls, parser): pass
    def to_str_(self, _a): return None
    def get_rate_(self, _a, global_step, _): return _a.learning_rate


@register
class linear(BaseSchedule):
    @classmethod
    def bind(cls, parser):
        parser.add_argument('--lrate_linear_end', type=float, default=0.01)

    def to_str_(self, _a): return f'{_a.lrate_linear_end:g}'

    def get_rate_(self, _a, global_step, _):
        return _a.learning_rate - (_a.learning_rate-_a.lrate_linear_end)*global_step/self.total_steps


@register
class exponential(BaseSchedule):
    @classmethod
    def bind(cls, parser):
        parser.add_argument('--lrate_exp_decay_steps', default=50, type=int)
        parser.add_argument('--lrate_exp_decay_rate', default=0.9, type=float)
        parser.add_argument('--lrate_exp_staircase', type=bool, default=1, choices=[1,0])

    def to_str_(self, _a): return f'{_a.lrate_exp_decay_steps:g}_{_a.lrate_exp_decay_rate:g}'

    def get_rate_(self, _a, global_step, _):
        val = global_step/_a.lrate_exp_decay_steps
        if _a.lrate_exp_staircase: val = math.floor(val)
        return _a.learning_rate*_a.lrate_exp_decay_rate**val


@register
class epochs(BaseSchedule):
    @classmethod
    def bind(cls, parser):
        parser.add_argument('--lrate_epochs_every', help='if one value, reduce every N epochs, otherwise reduce at each value', type=int, nargs='*', default=[10])
        parser.add_argument('--lrate_epochs_frac', help='reduce by this fraction', type=float, default=0.5)

    def to_str_(self, _a):
        joined = '-'.join(map(str, (_a.lrate_epochs_every)))
        return f'{joined}_{_a.lrate_epochs_frac:g}'

    def get_rate_(self, _a, _, epoch):
        ell = _a.lrate_epochs_every
        if len(ell)==1:
            val = math.floor(epoch/ell[0])
        else:
            val = sum(np.array(ell) < epoch)
        return _a.learning_rate*(_a.lrate_epochs_frac**(val))


def bind_learning_rates(parser):
    parser.add_argument('--lrate_schedule', choices=schedules.keys(), default='const')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    for sch in schedules.values(): sch.bind(parser)
    return parser


def plot_schedule():
    import matplotlib.pyplot as plt
    import mpl
    mpl.init()

    steps = _a.total_steps
    ax_ = plt.gca()
    def inner(name):
        sch = schedules.get(name)(_a, steps)
        rates = [sch(step, step/100) for step in range(steps)]
        ax_.plot(range(steps), rates, label=name)
        print(sch.to_str())

    if not _a.single:
        for name in schedules.keys(): inner(name)
    else:
        inner(_a.lrate_schedule)

    mpl.fmt_ax(ax_, 'Step', 'Learning rate', leg=1)
    if _a.ylog: ax_.set_yscale('log')
    plt.show()


def parse_args():
    parser = bind_learning_rates(argparse.ArgumentParser())
    parser.add_argument('--total_steps', type=int, default=1000)
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--ylog', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    _a = parse_args()
    print('[Arguments]', vars(_a))
    plot_schedule()
