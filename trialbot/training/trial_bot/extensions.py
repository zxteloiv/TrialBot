import datetime
import torch
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

def loss_reporter(bot):
    import torch
    import numpy
    output = bot.state.output
    loss = output['loss']
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    elif isinstance(loss, numpy.ndarray):
        loss = loss.item()

    bot.logger.info(f"Epoch: {bot.state.epoch}, Iteration: {bot.state.iteration}, Loss: {loss:.4f}")

