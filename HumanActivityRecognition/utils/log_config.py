import pathlib
import logging
import logging.config
import json
import os
import sys

# How to use :
# in the beginning of the file :
# import logging
# log = logging.getLogger('myapp')
# then use log to log whatever.

LOGGING_CONFIG = os.path.join(pathlib.Path(__file__).parent.resolve(), 'base_config.json')

with open(LOGGING_CONFIG, 'r') as fd:
    log_config = json.load(fd)
main_script = sys.argv[0]
script_name = os.path.splitext(os.path.basename(main_script))[0]
log_dir = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"{script_name}.log")

if 'file' in log_config['handlers']:
    log_config['handlers']['file']['filename'] = log_file_path

logging.config.dictConfig(log_config)
logger = logging.getLogger('myapp')

logger.info(f"Logging configured for {script_name} in {log_file_path}")


if __name__ == "__main__":
    log = logging.getLogger('myapp')
    log.debug('debug')
    log.info('info')
    log.warning('warning')
    log.critical('critical')