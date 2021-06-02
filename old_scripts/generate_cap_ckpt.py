from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers import reader
from sleeprnn.common import constants, pkeys

if __name__ == "__main__":
    dataset_name = constants.CAP_SS_NAME

    # Load from scratch
    params = {pkeys.FS: 200}
    dataset = reader.load_dataset(dataset_name, load_checkpoint=False, params=params)
    # Create checkpoint
    dataset.save_checkpoint()
    del dataset
    # Verify checkpoint
    dataset = reader.load_dataset(dataset_name, load_checkpoint=True, params=params)
