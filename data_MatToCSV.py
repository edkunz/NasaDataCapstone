# Change .mat files in /data/MATLAB/ to .csv files in /data/CSV/ for easier handling
import os
import pandas as pd
from scipy.io import loadmat
import numpy as np
from typing import List, Dict, Any

# Function to read all .mat files from a directory
def read_matlab_files(directory: str) -> Dict[str, Any]:
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.mat'):
            filepath = os.path.join(directory, filename)
            try:
                mat_data = loadmat(filepath)
                data[filename] = mat_data
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return data

# Convert MATLAB data to pandas DataFrame
def matlab_to_dataframe(mat_data: Dict[str, Any]) -> pd.DataFrame:
    records = []
    for key, value in mat_data.items():
        if isinstance(value, np.ndarray):
            flattened = value.flatten()
            for item in flattened:
                records.append({'variable': key, 'value': item})
        else:
            records.append({'variable': key, 'value': value})
    df = pd.DataFrame(records)
    return df

# Read MATLAB files from /data/MATLAB/ that start with MATLAB
matlab_files = read_matlab_files('data/MATLAB/')
matlab_files = {k: v for k, v in matlab_files.items() if k.startswith('MATLAB')}
# Convert to DataFrames
dataframes = {filename: matlab_to_dataframe(data) for filename, data in matlab_files.items()}
# Save dataframes to CSV files in /data/CSV/
os.makedirs('data/CSV/', exist_ok=True)
for filename, df in dataframes.items():
    csv_filename = filename.replace('.mat', '.csv')
    df.to_csv(os.path.join('data/CSV/', csv_filename), index=False)

