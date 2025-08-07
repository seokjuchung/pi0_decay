import sys
import numpy as np

SOFTWARE_DIR = '/nevis/houston/home/sc5303/anomaly/offline_anomaly/spine' # Change this path to your software install if you are not on S3DF

# Set software directory
sys.path.insert(0, SOFTWARE_DIR)

DATA_PATH = '/nevis/houston/home/sc5303/anomaly/offline_anomaly/anomalous_showers/pi0_decay/larcv_mc_20250729_145154_092166.root'

cfg = """
io:
  loader:
    batch_size: 128
    shuffle: False
    num_workers: 4
    collate_fn: all
    dataset:
      name: larcv
      file_keys: DATA_PATH
      limit_num_files: 10
      schema:
        input_data:
          parser: sparse3d
          sparse_event: sparse3d_pcluster
        seg_label:
          parser: sparse3d
          sparse_event: sparse3d_pcluster_semantics
        meta:
          parser: meta
          sparse_event: sparse3d_pcluster
        run_info:
          parser: meta
          sparse_event: sparse3d_pcluster
""".replace('DATA_PATH', DATA_PATH)

import yaml
from spine.driver import Driver

cfg = yaml.safe_load(cfg)

# prepare function configures necessary "handlers"
driver = Driver(cfg)

data = driver.process()

from spine.utils.globals import SHAPE_COL, VALUE_COL
from spine.vis.point import scatter_points
from spine.vis.layout import layout3d
from spine.utils.globals import GHOST_SHP

# Save input_data to a numpy .npy file
for entry in range(len(data['input_data'])):
    input_data = data['input_data'][entry]
    seg_label = data['seg_label'][entry][:, SHAPE_COL]
    nonghost_mask = seg_label < GHOST_SHP
    # Collect all data without adding entry as the first column
    if entry == 0:
        all_data = []
    entry_data = input_data[nonghost_mask]
    all_data.append(entry_data)

# After the loop, concatenate and save to a single file
all_data = np.vstack(all_data)
np.save('input_data_all.npy', all_data)
