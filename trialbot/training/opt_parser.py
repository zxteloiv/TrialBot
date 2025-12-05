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
    parser.add_argument('--dev', action="store_true", help='use testing mode on dev data')
    parser.add_argument('--hparamset', '-p', help="choose one of the available hyper-parameters",
                        choices=Registry._hparamsets.keys())
    parser.add_argument('--snapshot-dir', help="snapshot dir if continues")
    parser.add_argument('--dataset', choices=Registry._datasets.keys())
    parser.add_argument('--translator', choices=Registry._translators.keys())
    
    # Multi-GPU training arguments
    parser.add_argument('--gpus', type=str, default='', 
                        help='GPU IDs to use (comma-separated), e.g., "0,1,2,3". If not specified, use all available GPUs.')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                        choices=['nccl', 'gloo', 'mpi'],
                        help='distributed backend (nccl, gloo, mpi)')
    parser.add_argument('--dist-url', type=str, default='tcp://localhost:23456',
                        help='URL used to set up distributed training')
    parser.add_argument('--world-size', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', type=int, default=0,
                        help='node rank for distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training')
    parser.add_argument('--deepspeed', action='store_true',
                        help='Enable DeepSpeed training')
    parser.add_argument('--deepspeed-config', type=str, default=None,
                        help='Path to DeepSpeed configuration file')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (automatically set by torch.distributed.launch)')

    return parser
