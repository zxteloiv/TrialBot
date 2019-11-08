import json


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
    def common_settings():
        import os.path
        hparams = HyperParamSet()

        hparams.DEVICE = -1
        ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        hparams.ROOT = ROOT
        hparams.SNAPSHOT_PATH = os.path.join(ROOT, 'snapshots')

        hparams.LOG_REPORT_INTERVAL = (1, 'iteration')
        hparams.TRAINING_LIMIT = 500  # in num of epochs
        hparams.SAVE_INTERVAL = (100, 'iteration')
        hparams.batch_sz = 128
        hparams.vocab_min_freq = 3

        hparams.ADAM_LR = 1e-3
        hparams.ADAM_BETAS = (.9, .98)
        hparams.ADAM_EPS = 1e-9

        hparams.GRAD_CLIPPING = 5

        hparams.SGD_LR = 1e-2
        hparams.DATA_PATH = os.path.join(ROOT, 'data')

        return hparams

