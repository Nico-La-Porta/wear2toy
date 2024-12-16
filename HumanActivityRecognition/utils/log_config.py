import pathlib
import logging
import logging.config
import json
import os

# How to use :
# in the beginning of the file :
# import logging
# log = logging.getLogger('myapp')
# then use log to log whatever.

LOGGING_CONFIG = os.path.join(pathlib.Path(__file__).parent.resolve(), 'base_config.json')

with open(LOGGING_CONFIG, 'r') as fd:
    log_config = json.load(fd)

if log_config['handlers'].get('file'):
    # Check if the log file path exists, if not create it
    log_file = log_config['handlers']['file'].get('filename')
    if log_file:
        log_file_path = pathlib.Path(log_file)
        if not log_file_path.parent.exists():
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

logging.config.dictConfig(log_config)
logger = logging.getLogger('myapp')

logger.info(f"Logging configured from {LOGGING_CONFIG}")


if __name__ == "__main__":
    log = logging.getLogger('myapp')
    log.debug('debug')
    log.info('info')
    log.warning('warning')
    log.critical('critical')