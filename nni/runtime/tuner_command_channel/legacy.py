# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import threading

from .command_type import CommandType

_logger = logging.getLogger(__name__)


_lock = threading.Lock()
try:
    if os.environ.get('NNI_PLATFORM') != 'unittest':
        _in_file = open(3, 'rb')
        _out_file = open(4, 'wb')
except OSError:
    _logger.debug('IPC pipeline not exists')


def send(command, data):
    """Send command to Training Service.
    command: CommandType object.
    data: string payload.
    """
    global _lock
    try:
        _lock.acquire()
        data = data.encode('utf8')
        msg = b'%b%014d%b' % (command.value, len(data), data)
        _logger.debug('Sending command, data: [%s]', msg)
        _out_file.write(msg)
        _out_file.flush()
    finally:
        _lock.release()


def receive():
    """Receive a command from Training Service.
    Returns a tuple of command (CommandType) and payload (str)
    """
    header = _in_file.read(16)
    _logger.debug('Received command, header: [%s]', header)
    if header is None or len(header) < 16:
        # Pipe EOF encountered
        _logger.debug('Pipe EOF encountered')
        return None, None
    length = int(header[2:])
    data = _in_file.read(length)
    command = CommandType(header[:2])
    data = data.decode('utf8')
    _logger.debug('Received command, data: [%s]', data)
    return command, data
