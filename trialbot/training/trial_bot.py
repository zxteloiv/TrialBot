from typing import Callable, Optional, Dict, Any
import torch.nn
import os.path
import json
from datetime import datetime

from trialbot.data.dataset import Dataset
from trialbot.data.ns_vocabulary import NSVocabulary
from trialbot.data.iterator import Iterator
from trialbot.data.iterators.random_iterator import RandomIterator
from tqdm import tqdm
from enum import Enum

from trialbot.data.translator import Translator

from allennlp.nn.util import move_to_device

from .opt_parser import get_trial_bot_common_opt_parser
from .trial_registry import Registry
from .event_engine import Engine
from .updater import TestingUpdater, TrainingUpdater
import trialbot.training.extensions as ext_mod
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

class State(object):
    """An object that is used to pass internal and user-defined state between event handlers."""
    def __init__(self, **kwargs):
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
    They share the same program entry _run_ and is basically of the same routines,
    1. Do some preprocessing job if any (e.g. save vocab for reusing, makedirs for savepath, etc.)
    2. Initialize the iterator for running the experiment
    3. Get the assigned updater or initialize a new one, given the iterator and all the components
    4. Run the looping until the training ends. This starts an event-driven engine and fires events
       at some time. All the DIY requirements could be implemented by extensions, which are registered
       to the engine for some specific events and get run by the event engine when the very event is
       fired.

    """
    def __init__(self,
                 args = None,
                 trial_name="default_savedir",
                 get_model_func: Optional[Callable[[Any, NSVocabulary], torch.nn.Module]] = None,
                 ):
        if args is None:
            parser = TrialBot.get_default_parser()
            args = parser.parse_args()

        self.args = args
        self.name = trial_name

        # self.translator = None
        # self.vocab = None
        # self.train_set = None
        # self.dev_set = None
        # self.test_set = None
        self.hparams = None
        self.savepath = "."
        self.logger = logging.getLogger(__name__)

        self.updater = None

        self.state = State(epoch=0, iteration=0, output=None)

        self.get_model = get_model_func

        self._engine = self._make_engine()
        self._init_components()

    def _make_engine(self):
        engine = Engine()
        engine.register_events(*Events)
        # events with greater priorities will get processed earlier.
        engine.add_event_handler(Events.EPOCH_STARTED, ext_mod.ext_write_info, 100, msg="Epoch started")
        engine.add_event_handler(Events.EPOCH_STARTED, ext_mod.ext_write_info, 105, msg=("====" * 20))
        engine.add_event_handler(Events.EPOCH_STARTED, ext_mod.current_epoch_logger, 99)
        engine.add_event_handler(Events.STARTED, ext_mod.ext_write_info, 100, msg="TrailBot started")
        engine.add_event_handler(Events.STARTED, ext_mod.time_logger, 99)
        engine.add_event_handler(Events.COMPLETED, ext_mod.time_logger, 101)
        engine.add_event_handler(Events.COMPLETED, ext_mod.ext_write_info, 100, msg="TrailBot completed.")
        engine.add_event_handler(Events.ITERATION_COMPLETED, ext_mod.loss_reporter, 100)
        return engine

    @property
    def model(self):
        return self.models[0]

    @staticmethod
    def get_default_parser():
        parser = get_trial_bot_common_opt_parser()
        parser.add_argument('models', nargs='*', help='pretrained models for the same setting')
        parser.add_argument('--dry-run', action="store_true")
        parser.add_argument('--skip', type=int, help='skip NUM examples for the first iteration, intended for debug use.')
        parser.add_argument('--vocab-dump', help="the file path to save and load the vocab obj")
        return parser

    def _init_components(self):
        """
        Start a trial directly.
        """
        args = self.args

        # logging args
        if args.quiet:
            self.logger.setLevel(logging.WARNING)
        elif args.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        hparams = Registry.get_hparamset(args.hparamset)
        train_set, dev_set, test_set = Registry.get_dataset(args.dataset)
        translator = Registry.get_translator(args.translator)
        self.hparams = hparams
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set
        self.translator = translator

        vocab = self._init_vocab(train_set, translator)
        self.vocab = vocab
        translator.index_with_vocab(vocab)

        self.models = self._init_models(hparams, vocab)

        savepath = args.snapshot_dir if args.snapshot_dir else (os.path.join(
            hparams.SNAPSHOT_PATH,
            args.dataset,
            self.name,
            datetime.now().strftime('%Y%m%d-%H%M%S') + ('-' + args.memo if args.memo else '')
        ))
        self.savepath = savepath

    def _init_vocab(self,
                    dataset: Dataset,
                    translator: Translator):
        args, logger = self.args, self.logger
        hparams = self.hparams

        if args.vocab_dump and os.path.exists(args.vocab_dump):
            logger.info(f"read vocab from file {args.vocab_dump}")
            vocab = NSVocabulary.from_files(args.vocab_dump)

        else:
            logger.info("initialize vocab from training data")
            # count for a counter
            logger.debug("start initializing vocab")
            counter: Dict[str, Dict[str, int]] = dict()
            for example in tqdm(iter(dataset)):
                for namespace, w in translator.generate_namespace_tokens(example):
                    if namespace not in counter:
                        counter[namespace] = dict()

                    if w not in counter[namespace]:
                        counter[namespace][w] = 0

                    counter[namespace][w] += 1

            vocab = NSVocabulary(counter, min_count=({"tokens": 3}
                                                     if not hasattr(hparams, 'MIN_VOCAB_FREQ')
                                                     else hparams.MIN_VOCAB_FREQ))

        if args.vocab_dump:
            os.makedirs(args.vocab_dump, exist_ok=True)
            vocab.save_to_files(args.vocab_dump)
        logger.info(str(vocab))

        return vocab

    def _init_models(self, hparams, vocab):
        args = self.args
        models = self.get_model(hparams, vocab)
        if not isinstance(models, list):
            models = [models]

        if args.models:
            for model, model_path in zip(models, args.models):
                model.load_state_dict(torch.load(model_path))

        if args.device >= 0:
            models = [model.cuda(args.device) for model in models]
        return models

    def run(self):
        args, hparams = self.args, self.hparams

        if self.args.test:
            # testing procedure
            # 1. init iterator; 2. init updater; 3. run the testing engine
            hparams, model = self.hparams, self.model
            iterator = RandomIterator(self.test_set, hparams.batch_sz, self.translator, shuffle=False, repeat=False)
            updater = self.updater if self.updater is not None else self._default_test_updater(iterator)
            self._testing_engine_loop(iterator, updater)

        else:
            # training procedure
            # 1. save vocab; 2. init iterator; 3. init updater with iterator; 4. start main engine
            savepath = self.savepath
            if not os.path.exists(savepath):
                os.makedirs(savepath, mode=0o755)
            vocab_path = os.path.join(savepath, 'vocab')
            if not os.path.exists(vocab_path):
                os.makedirs(vocab_path, mode=0o755)
                self.vocab.save_to_files(vocab_path)

            repeat_iter = not args.debug
            shuffle_iter = not args.debug
            iterator = RandomIterator(self.train_set, self.hparams.batch_sz, self.translator,
                                      shuffle=shuffle_iter, repeat=repeat_iter)
            if args.debug and args.skip:
                iterator.reset(args.skip)

            updater = self.updater if self.updater is not None else self._default_train_updater(iterator)
            self._training_engine_loop(iterator, updater, hparams.TRAINING_LIMIT)

    def _default_train_updater(self, iterator):
        args, hparams, model = self.args, self.hparams, self.model
        logger = self.logger

        if hasattr(hparams, "OPTIM") and hparams.OPTIM == "SGD":
            logger.info(f"Using SGD optimzer with lr={hparams.SGD_LR}")
            optim = torch.optim.SGD(model.parameters(), hparams.SGD_LR)
        else:
            logger.info(f"Using Adam optimzer with lr={hparams.ADAM_LR} and beta={str(hparams.ADAM_BETAS)}")
            optim = torch.optim.Adam(model.parameters(), hparams.ADAM_LR, hparams.ADAM_BETAS)

        device = args.device
        dry_run = args.dry_run
        updater = TrainingUpdater(model, iterator, optim, device, dry_run)
        return updater

    def _default_test_updater(self, iterator: Iterator):
        args, model = self.args, self.model
        device = args.device
        updater = TestingUpdater(model, iterator, None, device)
        return updater

    def _training_engine_loop(self, iterator, updater, max_epoch):
        engine = self._engine
        engine.fire_event(Events.STARTED, bot=self)
        while iterator.epoch < max_epoch:
            self.state.epoch += 1
            engine.fire_event(Events.EPOCH_STARTED, bot=self)
            while True:
                self.state.iteration += 1
                engine.fire_event(Events.ITERATION_STARTED, bot=self)
                self.state.output = updater()
                engine.fire_event(Events.ITERATION_COMPLETED, bot=self)
                if iterator.is_new_epoch:
                    break

            engine.fire_event(Events.EPOCH_COMPLETED, bot=self)
        engine.fire_event(Events.COMPLETED, bot=self)

    def _testing_engine_loop(self, iterator, updater):
        engine = self._engine
        model = self.model
        with torch.no_grad():
            while True:
                output = updater()
                output = model.decode(output)
                print(json.dumps(output['predicted_tokens']))

                if iterator.is_new_epoch:
                    break

    def attach_extension(self,
                         event_name: str = Events.ITERATION_COMPLETED,
                         priority: int = 100,):
        def decorator(handler):
            self._engine.add_event_handler(event_name, handler, priority)
            return handler
        return decorator

