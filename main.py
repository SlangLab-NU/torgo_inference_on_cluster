'''
Fine-tune the wave2vec model on the Torgo dataset.

This is the main file for the project.
'''

# Import libraries
import sys
import os
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict, Audio

# Import custom modules
from config import torgo_dataset_path, torgo_csv_path

'''
--------------------------------------------------------------------------------
Store the command line arguments in variables
    - speaker_id: The speaker ID to fine-tune the model on
    - num_epochs: The number of epochs to fine-tune the model for
--------------------------------------------------------------------------------
'''
if len(sys.argv) < 2:
    print("Please provide the speaker ID and the number of epochs (optional).")
    sys.exit(1)
  
speaker_id = sys.argv[1]
if len(sys.argv) == 3:
    if sys.argv[2].isdigit() and int(sys.argv[2]) > 0:
        num_epochs = int(sys.argv[2])
    else:
        print("Please provide a valid number of epochs.")
        sys.exit(1)
else:
    num_epochs = 30

print()
print("Speaker ID: ", speaker_id)
print("Number of epochs: ", num_epochs)
print()


'''
--------------------------------------------------------------------------------
Read the Torgo dataset CSV file and store the data in a dictionary.
--------------------------------------------------------------------------------
'''
data_df = pd.read_csv(torgo_csv_path)
dataset_csv = load_dataset('csv', data_files=torgo_csv_path)
print("Dataset CSV: ", dataset_csv)