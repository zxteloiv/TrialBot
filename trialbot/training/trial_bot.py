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

        self.translator = None
        self.vocab = None
        self.train_set = None
        self.dev_set = None
        self.test_set = None
        self.hparams = None
        self.savepath = "."
        self.logger = logging.getLogger(__name__)


        self.state = State(epoch=0, iteration=0, output=None)

        self.get_model = get_model_func

        self._engine = self._make_engine()
        self._init_modules()

    @staticmethod
    def get_default_parser():
        parser = get_trial_bot_common_opt_parser()
        parser.add_argument('models', nargs='*', help='pretrained models for the same setting')
        parser.add_argument('--dry-run', action="store_true")
        parser.add_argument('--skip', type=int, help='skip NUM examples for the first iteration, intended for debug use.')
        parser.add_argument('--vocab-dump', help="the file path to save and load the vocab obj")
        return parser

    def _default_train_fn(self, iterator: Iterator, model: torch.nn.Module, optimizer):
        device = self.args.device
        dry_run = self.args.dry_run

        def update():
            model.train()
            optimizer.zero_grad()
            batch: Dict[str, torch.Tensor] = next(iterator)

            nonlocal device
            if device >= 0:
                batch = move_to_device(batch, device)

            output = model(**batch)
            if not dry_run:
                loss = output['loss']
                loss.backward()
                optimizer.step()
            return output

        return update

    def _default_test_fn(self, iterator: Iterator, model: torch.nn.Module):
        device = self.args.device
        def update():
            model.eval()
            batch = next(iterator)
            nonlocal device
            if 'target_tokens' in batch:
                del batch['target_tokens']
            if device >= 0:
                batch = move_to_device(batch, device)
            output = model(**batch)
            return output

        return update

    def _init_modules(self):
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
        model = self.get_model(hparams, vocab)

        if args.models:
            model.load_state_dict(torch.load(args.models[0]))

        if args.device >= 0:
            model = model.cuda(args.device)
        self.model = model

        savepath = args.snapshot_dir if args.snapshot_dir else (os.path.join(
            hparams.SNAPSHOT_PATH,
            args.dataset,
            self.name,
            datetime.now().strftime('%Y%m%d-%H%M%S') + ('-' + args.memo if args.memo else '')
        ))
        self.savepath = savepath

    def run(self):
        if self.args.test:
            self._test()
        else:
            self._train()

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

    def _train(self):
        args, hparams, model = self.args, self.hparams, self.model
        logger = self.logger

        repeat_iter = not args.debug
        shuffle_iter = not args.debug
        iterator = RandomIterator(self.train_set, hparams.batch_sz, self.translator,
                                  shuffle=shuffle_iter, repeat=repeat_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        if hasattr(hparams, "OPTIM") and hparams.OPTIM == "SGD":
            logger.info(f"Using SGD optimzer with lr={hparams.SGD_LR}")
            optim = torch.optim.SGD(model.parameters(), hparams.SGD_LR)
        else:
            logger.info(f"Using Adam optimzer with lr={hparams.ADAM_LR} and beta={str(hparams.ADAM_BETAS)}")
            optim = torch.optim.Adam(model.parameters(), hparams.ADAM_LR, hparams.ADAM_BETAS)

        savepath = self.savepath
        if not os.path.exists(savepath):
            os.makedirs(savepath, mode=0o755)
        vocab_path = os.path.join(savepath, 'vocab')
        if not os.path.exists(vocab_path):
            os.makedirs(vocab_path, mode=0o755)
            self.vocab.save_to_files(vocab_path)

        updater = self._default_train_fn(iterator, model, optim)
        self._engine_loop(iterator, updater, hparams.TRAINING_LIMIT)

    def _engine_loop(self, iterator, updater, max_epoch):
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

    def _test(self):
        hparams, model = self.hparams, self.model
        iterator = RandomIterator(self.test_set, hparams.batch_sz, self.translator, shuffle=False, repeat=False)
        updater = self._default_test_fn(iterator, model)
        with torch.no_grad():
            while True:
                output = updater()
                output = model.decode(output)
                print(json.dumps(output['predicted_tokens']))

                if iterator.is_new_epoch:
                    break

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
        engine.add_event_handler(Events.EPOCH_COMPLETED, ext_mod.every_epoch_model_saver, 100)
        engine.add_event_handler(Events.ITERATION_COMPLETED, ext_mod.loss_reporter, 100)
        return engine

    def attach_extension(self,
                         event_name: str = Events.ITERATION_COMPLETED,
                         priority: int = 100,):
        def decorator(handler):
            self._engine.add_event_handler(event_name, handler, priority)
            return handler
        return decorator

