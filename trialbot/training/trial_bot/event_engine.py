"""
This code file is majorly based on pytorch-ignite, whose license is available at
https://github.com/pytorch/ignite/blob/master/LICENSE,
and duplicated below.

BSD 3-Clause License

Copyright (c) 2018, PyTorch team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import logging
from collections import defaultdict

class Engine(object):
    """Runs a given process_function over each batch of a dataset, emitting events as it goes.

    Args:
        process_function (callable): A function receiving a handle to the engine and the current batch
            in each iteration, and returns data to be stored in the engine's state.

    Example usage:

    .. code-block:: python

        def train_and_store_loss(engine, batch):
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            return loss.item()

        engine = Engine(train_and_store_loss)
        engine.run(data_loader)

        # Loss value is now stored in `engine.state.output`.

    """
    def __init__(self):
        self._event_handlers = defaultdict(list)
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())
        self._allowed_events = []

    def register_events(self, *event_names):
        """Add events that can be fired.

        Registering an event will let the user fire these events at any point.
        This opens the door to make the :meth:`~ignite.engine.Engine.run` loop even more
        configurable.

        By default, the events from :class:`~ignite.engine.Events` are registerd.

        Args:
            *event_names: An object (ideally a string or int) to define the
                name of the event being supported.

        Example usage:

        .. code-block:: python

            from enum import Enum

            class Custom_Events(Enum):
                FOO_EVENT = "foo_event"
                BAR_EVENT = "bar_event"

            engine = Engine(process_function)
            engine.register_events(*Custom_Events)

        """
        for name in event_names:
            self._allowed_events.append(name)

    def add_event_handler(self, event_name, handler, priority, *args, **kwargs):
        """Add an event handler to be executed when the specified event is fired.

        Args:
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.Events`
                or any `event_name` added by :meth:`~ignite.engine.Engine.register_events`.
            handler (callable): the callable event handler that should be invoked
            *args: optional args to be passed to `handler`.
            **kwargs: optional keyword args to be passed to `handler`.

        Notes:
              The handler function's first argument will be `self`, the :class:`~ignite.engine.Engine` object it
              was bound to.

              Note that other arguments can be passed to the handler in addition to the `*args` and  `**kwargs`
              passed here, for example during :attr:`~ignite.engine.Events.EXCEPTION_RAISED`.

        Example usage:

        .. code-block:: python

            engine = Engine(process_function)

            def print_epoch(engine):
                print("Epoch: {}".format(engine.state.epoch))

            engine.add_event_handler(Events.EPOCH_COMPLETED, print_epoch)

        """
        if event_name not in self._allowed_events:
            self._logger.error("attempt to add event handler to an invalid event %s.", event_name)
            raise ValueError("Event {} is not a valid event for this Engine.".format(event_name))

        self._event_handlers[event_name].append((handler, priority, args, kwargs))
        self._logger.debug("added handler for event %s.", event_name)

    def has_event_handler(self, handler, event_name=None):
        """Check if the specified event has the specified handler.

        Args:
            handler (callable): the callable event handler.
            event_name: The event the handler attached to. Set this
                to ``None`` to search all events.
        """
        if event_name is not None:
            if event_name not in self._event_handlers:
                return False
            events = [event_name]
        else:
            events = self._event_handlers
        for e in events:
            for h, _, _, _ in self._event_handlers[e]:
                if h == handler:
                    return True
        return False

    def remove_event_handler(self, handler, event_name):
        """Remove event handler `handler` from registered handlers of the engine

        Args:
            handler (callable): the callable event handler that should be removed
            event_name: The event the handler attached to.

        """
        if event_name not in self._event_handlers:
            raise ValueError("Input event name '{}' does not exist".format(event_name))

        new_event_handlers = [(h, p, args, kwargs) for h, p, args, kwargs in self._event_handlers[event_name]
                              if h != handler]
        if len(new_event_handlers) == len(self._event_handlers[event_name]):
            raise ValueError("Input handler '{}' is not found among registered event handlers".format(handler))
        self._event_handlers[event_name] = new_event_handlers

    def on(self, event_name, *args, **kwargs):
        """Decorator shortcut for add_event_handler.

        Args:
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.Events` or
                any `event_name` added by :meth:`~ignite.engine.Engine.register_events`.
            *args: optional args to be passed to `handler`.
            **kwargs: optional keyword args to be passed to `handler`.

        """
        def decorator(f):
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f
        return decorator

    def fire_event(self, event_name, *event_args, **event_kwargs):
        """Execute all the handlers associated with given event.

        This method executes all handlers associated with the event
        `event_name`. Optional positional and keyword arguments can be used to
        pass arguments to **all** handlers added with this event. These
        aguments updates arguments passed using :meth:`~ignite.engine.Engine.add_event_handler`.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.Events` or any `event_name` added by
                :meth:`~ignite.engine.Engine.register_events`.
            *event_args: optional args to be passed to all handlers.
            **event_kwargs: optional keyword args to be passed to all handlers.

        """
        if event_name in self._allowed_events:
            self._logger.debug("firing handlers for event %s ", event_name)
            for func, _, args, kwargs in sorted(self._event_handlers[event_name], key=lambda x: x[1], reverse=True):
                kwargs.update(event_kwargs)
                func(*(event_args + args), **kwargs)
