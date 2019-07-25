import argparse

from .trial_registry import Registry


def get_trial_bot_common_opt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, help='manually set the seeds for torch')
    parser.add_argument('--device', type=int, default=-1, help="the gpu device number to override")
    parser.add_argument("--quiet", action="store_true", help="mute the log")
    parser.add_argument("--debug", action="store_true", help="print the debugging log")
    parser.add_argument('--memo', type=str, default="", help="used to remember some runtime configurations")
    parser.add_argument('--test', action="store_true", help='use testing mode')
    parser.add_argument('--hparamset', '-p', help="choose one of the available hyper-parameters",
                        choices=Registry._hparamsets.keys())
    parser.add_argument('--snapshot-dir', help="snapshot dir if continues")
    parser.add_argument('--dataset', choices=Registry._datasets.keys())
    parser.add_argument('--translator', choices=Registry._translators.keys())

    return parser