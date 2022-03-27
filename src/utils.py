import subprocess
import os
import platform
import json
import collections

from datetime import datetime


def write_info(args, fp):
    data = {
        'host': platform.node(),
        'cwd': os.getcwd(),
        'timestamp': str(datetime.now()),
        'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
        'args': vars(args)
    }
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path