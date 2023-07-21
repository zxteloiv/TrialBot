import os.path


class HyperParamSet:
    def __str__(self):
        obj = {}
        for attr in dir(self):
            if attr.startswith('_'):
                continue

            val = getattr(self, attr)
            if any(isinstance(val, t) for t in (tuple, list, dict, int, str, float)):
                obj[attr] = str(val)

        return "\n".join("%s: %s" % (k, v) for k, v in obj.items())

    @staticmethod
    def common_settings(root_path=None):
        p = HyperParamSet()

        p.ROOT = '.' if root_path is None else root_path
        p.SNAPSHOT_PATH = os.path.join(p.ROOT, 'snapshots')
        p.DATA_PATH = os.path.join(p.ROOT, 'data')

        p.TRAINING_LIMIT = 500  # in num of epochs
        p.batch_sz = 128

        p.OPTIM = 'adam'
        p.OPTIM_KWARGS = dict()
        p.ADAM_LR = 1e-3
        p.ADAM_BETAS = (.9, .98)
        p.ADAM_EPS = 1e-9
        p.WEIGHT_DECAY = 0.
        p.GRAD_CLIPPING = 2
        p.SGD_LR = 1e-2

        p.NS_VOCAB_KWARGS = dict()

        return p

