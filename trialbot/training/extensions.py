import datetime
import torch
import math
import os.path


def ext_write_info(bot, msg):
    bot.logger.info(msg)


def current_epoch_logger(bot):
    bot.logger.info(f"Current Epoch {bot.state.epoch}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def time_logger(bot):
    bot.logger.info(f"Current Time {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def every_epoch_model_saver(bot, interval=1):
    if bot.state.epoch % interval == 0:
        savedir, model = bot.savepath, bot.model
        filename = os.path.join(savedir, f"model_state_{bot.state.epoch}.th")
        torch.save(model.state_dict(), filename)
        bot.logger.info(f"model saved to {filename}")


def loss_reporter(bot, interval=4):
    if bot.state.iteration % interval != 0:
        return

    import torch
    import numpy
    output = bot.state.output
    if output is None:
        return

    loss = output.get('loss')
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    elif isinstance(loss, numpy.ndarray):
        loss = loss.item()
    else:
        return

    bot.logger.info(f"Epoch: {bot.state.epoch}, Iteration: {bot.state.iteration}, Loss: {loss:.4f}")


def print_hyperparameters(bot):
    bot.logger.info(f"Cmd Arguments Used:\n{bot.args}")
    bot.logger.info(f"Hyperparamset Used: {bot.args.hparamset}\n{str(bot.hparams)}")


def print_snaptshot_path(bot):
    bot.logger.info("Snapshot Dir: " + bot.savepath)


def print_models(bot):
    bot.logger.info("Model Specs:\n" + str(bot.models))


def collect_garbage(bot):
    import gc
    for optim in bot.updater._optims:
        optim.zero_grad()

    if hasattr(bot.state, "output") and bot.state.output is not None:
        bot.state.output = None
    gc.collect()
    if bot.args.device >= 0:
        import torch.cuda
        torch.cuda.empty_cache()


def end_with_nan_loss(bot):
    import numpy as np
    output = getattr(bot.state, 'output', None)
    if output is None:
        return
    loss = output.get('loss', None)
    if loss is None:
        return

    def _isnan(x):
        if isinstance(x, torch.Tensor):
            return bool(torch.isnan(x).any())
        elif isinstance(x, np.ndarray):
            return bool(np.isnan(x).any())
        else:
            return math.isnan(x)

    if _isnan(loss):
        bot.logger.error(f"NaN loss encountered, training ended at epoch {bot.state.epoch} iter {bot.state.iteration}")
        bot.state.epoch = bot.hparams.TRAINING_LIMIT + 1
        bot.updater.stop_epoch()


