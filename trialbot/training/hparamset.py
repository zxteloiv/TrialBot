import json


class HyperParamSet:
    def __str__(self):
        json_obj = dict((attr, getattr(self, attr)) for attr in dir(self)
                        if hasattr(self, attr) and not attr.startswith('_'))
        return json.dumps(json_obj)

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

