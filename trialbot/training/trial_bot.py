import argparse
from typing import Callable, Optional, Any
import torch.nn
import os.path
from datetime import datetime

from trialbot.data.dataset import Dataset
from trialbot.data.ns_vocabulary import NSVocabulary
from tqdm import tqdm
from enum import Enum

from trialbot.data.translator import Translator

from .opt_parser import get_trial_bot_common_opt_parser
from .trial_registry import Registry
from .event_engine import Engine
from . import extensions as exts
import logging
logging.basicConfig()


class Events(Enum):
    """Predefined events for TrialBot training."""
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    STARTED = "started"
    COMPLETED = "completed"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    EXCEPTION_RAISED = "exception_raised"


class State:
    """An object that is used to pass internal and user-defined state between event handlers."""
    def __init__(self, **kwargs):
        self.epoch = 0
        self.iteration = 0
        self.output = None

        for k, v in kwargs.items():
            setattr(self, k, v)


class TrialBot:
    """
    A TrialBot is the direct manager for an experiment, which is responsible for only two requirements,
    Run and Serve, corresponding to run the experiment or to get loaded by some server.

    A TrialBot maintains several Components, and schedules the experiment procedure.
    Components are as follows so far,
    - args: arguments parsed from commandline
    - hparamset: the designated hyperparameters from the running shell
    - dataset: specifically train_set, dev_set, test_set
    - translator: how the dataset should be interpreted.
    - savepath: where the data will be saved if given, or set by default.
    - vocab: the vocabulary with several namespaces
    - models: the list in which all the models for the single experiments are included.

    An experiment is empirically divided into two phase, Training and Testing.
    They share the same program entry _run_ and basically consist of the similar routines,
    1. Do some preprocessing job if any (e.g. save vocab for reusing, makedirs for savepath, etc.)
    2. Initialize the updater for running the experiment.
        1) For a standard updater, the classmethod of the updater already implements the initialization
        of the iterators and optimizers.
        2) For a custom updater, the class initialization could be anywhere given the bot with its components.
        Then the custom updater must be injected into the bot with property assignment. like

        ```python
            bot.updater = Updater(bot)
        ```
    3. Run the looping until the training ends. This starts an event-driven engine and fires events
       at some time. All the DIY requirements could be implemented by extensions, which are registered
       to the engine for some specific events and get run by the event engine when the very event is
       fired.
       By default, the loop contains some epochs and some iterations of each epoch.
       The loop relies on the updater, and start a new epoch only after the updater raise an StopIteration exception.

    """
    def __init__(self,
                 args: argparse.Namespace | None = None,
                 trial_name: str = "trial",
                 get_model_func: Callable[[Any, NSVocabulary], torch.nn.Module] | None = None,
                 clean_engine: bool = False,
                 runtime_hooks: dict[str, Callable] | None = None,
                 ):
        self.runtime_hooks = dict() if runtime_hooks is None else runtime_hooks

        self.args = self._init_args(args)   # if given, use the given args, otherwise it will parse from sys.argv
        self.name = trial_name
        self.hparams = self._init_hparams()
        self.savepath = self._init_savepath()
        self.logger = logging.getLogger(__name__)

        self.updater = None

        self.state = State(epoch=0, iteration=0, output=None)
        self.get_model = get_model_func
        self.engine = self._init_engine(clean=clean_engine)
        self.datasets = self._init_dataset()
        self.translator = self._init_translator()
        self.vocab = self._init_vocab(self.train_set, self.translator)
        if self.translator and self.vocab:
            self.translator.index_with_vocab(self.vocab)
        self.models = self._init_models(self.hparams, self.vocab)

    def _init_engine(self, clean: bool = False):
        engine = Engine()
        engine.register_events(*Events)
        # events with greater priorities will get processed earlier.
        if not clean:
            # by default not to add expensive extensions, such as dumping models of each epoch,
            # evaluations at the end of each epoch, collecting garbage.
            engine.add_event_handler(Events.STARTED, exts.print_hyperparameters, 90)
            engine.add_event_handler(Events.STARTED, exts.print_snaptshot_path, 90)
            engine.add_event_handler(Events.STARTED, exts.print_models, 100)
            engine.add_event_handler(Events.STARTED, exts.ext_write_info, 105, msg=("====" * 20))
            engine.add_event_handler(Events.EPOCH_STARTED, exts.ext_write_info, 105, msg=("----" * 20))
            engine.add_event_handler(Events.EPOCH_STARTED, exts.ext_write_info, 100, msg="Epoch started")
            engine.add_event_handler(Events.EPOCH_STARTED, exts.current_epoch_logger, 99)
            engine.add_event_handler(Events.STARTED, exts.ext_write_info, 100, msg="TrailBot started")
            engine.add_event_handler(Events.STARTED, exts.time_logger, 99)
            engine.add_event_handler(Events.COMPLETED, exts.time_logger, 101)
            engine.add_event_handler(Events.COMPLETED, exts.ext_write_info, 100, msg="TrailBot completed.")
            engine.add_event_handler(Events.ITERATION_COMPLETED, exts.loss_reporter, 100)
            engine.add_event_handler(Events.ITERATION_COMPLETED, exts.end_with_nan_loss, 100)
        self.engine = engine
        return engine

    @property
    def model(self):
        return self.models[0]

    @staticmethod
    def get_default_parser():
        parser = get_trial_bot_common_opt_parser()
        parser.add_argument('models', nargs='*', help='pretrained models for the same setting')
        parser.add_argument('--vocab-path', help="the file path to save and load the vocab obj")

        parser.add_argument('--batch-size', type=int, default=0,
                            help='overwrite the hparamset.batch_sz if specified at CLI')
        parser.add_argument('--epoch', type=int, default=0,
                            help='overwrite the hparamset.TRAINING_LIMIT if specified at CLI')
        return parser

    def _init_args(self, args=None):
        if args is None:
            parser = self.get_default_parser()
            args = parser.parse_args()

        # logging args
        if args.quiet:
            self.logger.setLevel(logging.WARNING)
        elif args.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.args = args
        return args

    def _init_hparams(self):
        args = self.args
        hparams = Registry.get_hparamset(args.hparamset)
        if args.batch_size > 0:
            hparams.batch_sz = args.batch_size
        if args.epoch > 0:
            hparams.TRAINING_LIMIT = args.epoch
        if args.translator:
            hparams.TRANSLATOR = args.translator
        if 'hparams' in self.runtime_hooks:
            hparams = self.runtime_hooks['hparams'](hparams)

        self.hparams = hparams
        return hparams

    def _init_dataset(self):
        args = self.args
        self.datasets = Registry.get_dataset(args.dataset)
        return self.datasets

    def _init_savepath(self):
        args = self.args
        if args.test and len(args.models) > 0:
            # use the first model path as the savepath,
            # if this is not expected, please specify manually the
            return os.path.abspath(os.path.expanduser(os.path.dirname(args.models[0])))

        if args.snapshot_dir:
            return args.snapshot_dir

        return os.path.join(
            self.hparams.SNAPSHOT_PATH,
            args.dataset,
            self.name,
            datetime.now().strftime('%Y%m%d-%H%M%S') + ('-' + args.memo if args.memo else '')
        )

    @property
    def train_set(self):
        return self._get_classical_dataset_split(0)

    @property
    def dev_set(self):
        return self._get_classical_dataset_split(1)

    @property
    def test_set(self):
        return self._get_classical_dataset_split(2)

    def _get_classical_dataset_split(self, i):
        assert len(self.datasets) >= 3 and all(isinstance(x, Dataset) for x in self.datasets[:3])
        return self.datasets[i]

    def _init_translator(self):
        p = self.hparams
        translator_kwargs = getattr(p, 'TRANSLATOR_KWARGS', dict())
        translator = Registry.get_translator(p.TRANSLATOR, **translator_kwargs)
        self.translator = translator
        return translator

    def _init_vocab(self, dataset: Dataset, translator: Translator):
        args, logger = self.args, self.logger
        hparams = self.hparams

        if args.vocab_path:
            vocab_path = args.vocab_path
        else:
            vocab_path = os.path.join(self.savepath, 'vocab')
            logger.info(f'assuming the vocab is located at {vocab_path}')

        if os.path.exists(vocab_path):
            logger.info(f"read vocab from file {vocab_path}")
            vocab = NSVocabulary.from_files(vocab_path)

        else:
            logger.info("initialize vocab from training data")
            # count for a counter
            counter: dict[str, dict[str, int]] = dict()
            for example in tqdm(iter(dataset)):
                for namespace, w in translator.generate_namespace_tokens(example):
                    if namespace not in counter:
                        counter[namespace] = dict()

                    if w not in counter[namespace]:
                        counter[namespace][w] = 0

                    counter[namespace][w] += 1

            vocab = NSVocabulary(counter, **hparams.NS_VOCAB_KWARGS)

            if args.test:
                logger.warning(f'the vocab is not saved by default in eval mode '
                               f'although it is just built from scratch.')
            else:
                vocab.save_to_files(vocab_path)

        logger.info(str(vocab))
        return vocab

    def _init_models(self, hparams, vocab):
        args = self.args
        models = self.get_model(hparams, vocab)
        if not isinstance(models, list) and not isinstance(models, tuple):
            models = [models]

        if args.models:
            def int_to_device(device):
                if isinstance(device, torch.device):
                    return device
                if device < 0:
                    return torch.device("cpu")
                return torch.device(device)

            for model, model_path in zip(models, args.models):
                model.load_state_dict(torch.load(model_path, map_location=int_to_device(args.device)))

        if args.device >= 0:
            models = [model.cuda(args.device) for model in models]
        return models

    def run(self, training_epoch: int = 0):
        if training_epoch == 0:
            training_epoch = self.hparams.TRAINING_LIMIT

        if self.args.test:
            # testing procedure
            # 1. init updater; 2. run the testing engine
            if self.updater is None:
                from .updaters.testing_updater import TestingUpdater
                self.updater = TestingUpdater.from_bot(self)
            self._testing_engine_loop(self.updater)

        else:
            # training procedure
            # 1. ensure snapshot-dir (savepath); 2. init updater; 3. start main engine
            savepath = self.savepath
            if not os.path.exists(savepath):
                os.makedirs(savepath, mode=0o755)

            if self.updater is None:
                from .updaters.training_updater import TrainingUpdater
                self.updater = TrainingUpdater.from_bot(self)
            self._training_engine_loop(self.updater, training_epoch)

    def _training_engine_loop(self, updater, max_epoch):
        engine = self.engine
        engine.fire_event(Events.STARTED, bot=self)
        while self.state.epoch < max_epoch:
            self.state.epoch += 1
            self._epoch_loop(updater)
        engine.fire_event(Events.COMPLETED, bot=self)

    def _testing_engine_loop(self, updater):
        engine = self.engine
        with torch.no_grad():
            engine.fire_event(Events.STARTED, bot=self)
            self._epoch_loop(updater)
            engine.fire_event(Events.COMPLETED, bot=self)

    def _epoch_loop(self, updater):
        engine = self.engine
        engine.fire_event(Events.EPOCH_STARTED, bot=self)
        updater.start_epoch()
        while True:
            self.state.iteration += 1
            engine.fire_event(Events.ITERATION_STARTED, bot=self)
            try:
                self.state.output = next(updater)
                engine.fire_event(Events.ITERATION_COMPLETED, bot=self)
            except StopIteration:
                self.state.iteration -= 1
                self.state.output = None
                break
        engine.fire_event(Events.EPOCH_COMPLETED, bot=self)

    def attach_extension(self,
                         event_name: str = Events.ITERATION_COMPLETED,
                         priority: int = 100,):
        """Used as a decorator only. To add extension directly, use add_event_handler instead."""
        def decorator(handler):
            self.engine.add_event_handler(event_name, handler, priority)
            return handler
        return decorator

    def add_event_handler(self, event_name, handler, priority=100, *args, **kwargs):
        self.engine.add_event_handler(event_name, handler, priority, *args, **kwargs)

