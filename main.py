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
  
test_speaker = sys.argv[1]
if len(sys.argv) == 3:
    if sys.argv[2].isdigit() and int(sys.argv[2]) > 0:
        num_epochs = int(sys.argv[2])
    else:
        print("Please provide a valid number of epochs.")
        sys.exit(1)
else:
    num_epochs = 30

print()
print("Test Speaker: ", test_speaker)
print("Number of epochs: ", num_epochs)
print()


'''
--------------------------------------------------------------------------------
Read the Torgo dataset CSV file and store the data in a dictionary.
--------------------------------------------------------------------------------
'''
data_df = pd.read_csv(torgo_csv_path)
dataset_csv = load_dataset('csv', data_files=torgo_csv_path)
speakers = data_df['speaker_id'].unique()

print()
print("Unique speakers found in the dataset:")
print(", ".join(speakers))

if test_speaker not in speakers:
    print("Test Speaker not found in the dataset.")
    sys.exit(1)
    
print()


'''
--------------------------------------------------------------------------------
Split the dataset into training / validation / test sets.
--------------------------------------------------------------------------------
'''
valid_speaker = 'F03' if test_speaker != 'F03' else 'F04'
train_speaker = [s for s in speakers if s not in [test_speaker, valid_speaker]]

torgo_dataset = DatasetDict()
torgo_dataset['train'] = dataset_csv['train'].filter(lambda x: x in train_speaker, input_columns=['speaker_id'])
torgo_dataset['validation'] = dataset_csv['train'].filter(lambda x: x == valid_speaker, input_columns=['speaker_id'])
torgo_dataset['test'] = dataset_csv['train'].filter(lambda x: x == test_speaker, input_columns=['speaker_id'])

print(torgo_dataset)

