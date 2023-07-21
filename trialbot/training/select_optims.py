from functools import partial
import torch


def torch_optim_cls(p):
    name = getattr(p, "OPTIM", "adam").lower()
    kwargs = getattr(p, 'OPTIM_KWARGS', dict())

    auto_name = lambda c: (c.__name__.lower(), c)
    adam_based_optims = dict(map(auto_name, (
        torch.optim.Adam,
        torch.optim.Adamax,
        torch.optim.AdamW,
        torch.optim.NAdam,
        torch.optim.RAdam,
        torch.optim.RMSprop,
        torch.optim.SparseAdam,
    )))

    if name == "sgd":
        cls = partial(torch.optim.SGD, lr=p.SGD_LR, weight_decay=p.WEIGHT_DECAY, **kwargs)

    elif name in adam_based_optims:
        adam_kwargs = {'lr': p.ADAM_LR, 'betas': p.ADAM_BETAS, 'weight_decay': p.WEIGHT_DECAY}
        cls = partial(adam_based_optims[name], **adam_kwargs, **kwargs)

    elif name == "rmsprop":
        cls = partial(torch.optim.RMSprop, lr=p.SGD_LR, weight_decay=p.WEIGHT_DECAY, **kwargs)

    else:
        raise ValueError(f"Optimizer {name} not found.")

    return cls

