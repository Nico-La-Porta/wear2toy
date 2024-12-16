import os
import dotenv

# Set a default environment if none is provided
APP_ENV = os.getenv('APP_ENV')

if APP_ENV == 'hpc_supsi':
    from .config_hpc_supsi import *
elif APP_ENV == 'testing':
    from .config_test import *
elif APP_ENV == 'dev':
    from .config_dev import *
else:
    dotenv.load_dotenv()