import sys
import numpy as np
import os
import glob
import yaml
SOFTWARE_DIR = '/nevis/houston/home/sc5303/anomaly/offline_anomaly/spine'
sys.path.insert(0, SOFTWARE_DIR)
from spine.driver import Driver

DATA_DIR = '/nevis/riverside/data/sc5303/sbnd/offline_ad/pi0/'
OUTPUT_DIR = DATA_DIR  # Save NPY files to the same directory

# Get all LArCV files
larcv_files = sorted(glob.glob(os.path.join(DATA_DIR, 'larcv*.root')))
print(f"Found {len(larcv_files)} LArCV files")

# Configuration template
cfg_template = """
io:
  loader:
    batch_size: 128
    shuffle: False
    num_workers: 4
    collate_fn: all
    dataset:
      name: larcv
      file_keys: [FILE_PATH]
      limit_num_files: 1
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
"""

from spine.utils.globals import SHAPE_COL, VALUE_COL, GHOST_SHP

# Process each file individually
for file_idx, larcv_file in enumerate(larcv_files):
    print(f"\nProcessing file {file_idx + 1}/{len(larcv_files)}: {os.path.basename(larcv_file)}")
    
    # Create configuration for this file
    cfg = cfg_template.replace('FILE_PATH', larcv_file)
    cfg_dict = yaml.safe_load(cfg)
    
    try:
        # Initialize driver
        driver = Driver(cfg_dict)
        data = driver.process()
        
        # Collect all input data from all entries
        all_data = []
        
        for entry in range(len(data['input_data'])):
            input_data = data['input_data'][entry]
            seg_label = data['seg_label'][entry][:, SHAPE_COL]
            nonghost_mask = seg_label < GHOST_SHP
            
            # Get non-ghost data for this entry
            entry_data = input_data[nonghost_mask]
            all_data.append(entry_data)
        
        # Concatenate all data from this file
        if all_data:
            all_data_concatenated = np.vstack(all_data)
            
            # Create output filename based on input filename
            input_basename = os.path.basename(larcv_file)
            output_filename = input_basename.replace('.root', '.npy')
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Save to NPY file
            np.save(output_path, all_data_concatenated)
            
            # Get file size for reporting
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Saved {output_filename} ({file_size_mb:.1f} MB) with {all_data_concatenated.shape[0]} points")
            
        else:
            print(f"No data found in {os.path.basename(larcv_file)}")
            
    except Exception as e:
        print(f"Error processing {os.path.basename(larcv_file)}: {str(e)}")
        continue

print("\nProcessing complete!")
