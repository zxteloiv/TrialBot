from typing import List, Mapping, Callable
from trialbot.training import Registry
from itertools import product


def expand_hparams(grid_conf: Mapping[str, list]):
    keys, value_lists = zip(*grid_conf.items())
    for v in product(*value_lists):
        params = dict(zip(keys, v))
        yield params

def update_registry_params(name_prefix, i, params, base_params_fn: Callable):
    name = f"{name_prefix}{i}"

    def hparams_wrapper_fn():
        p = base_params_fn()
        for k, v in params.items():
            setattr(p, k, v)
        return p

    Registry._hparamsets[name] = hparams_wrapper_fn
    return name

def import_grid_search_parameters(grid_conf: Mapping[str, list], base_param_fn: Callable, name_prefix: str = None):
    param_sets = list(expand_hparams(grid_conf))
    import re
    if name_prefix is None:
        name_prefix = re.sub('[^a-zA-Z_0-9]', '', base_param_fn.__name__) + '_'
    names = [update_registry_params(name_prefix, i, params, base_param_fn) for i, params in enumerate(param_sets)]
    return names

if __name__ == '__main__':
    grid_conf = {"batch_size": [64, 128, 256], "hidden_size": [300, 600]}
    grid_conf = {
                    "num_inner_loops": [1, 2, 3, 5, 10],
                    "batch_sz": [20, 40, 80],
                    "support_batch_sz": [100, 200, 400]
                }
    from trialbot.training.hparamset import HyperParamSet
    names = import_grid_search_parameters(grid_conf, lambda: HyperParamSet())
    print(f"names={names}")
    print(Registry._hparamsets)
    p = Registry.get_hparamset(names[-1])
    print(type(p))

    for name in names:
        print('--------' * 10)
        print("hparamset: " + name)
        print(Registry.get_hparamset(name))

