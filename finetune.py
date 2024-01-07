'''
Fine-tune the wave2vec model on the Torgo dataset. This script takes in the
speaker ID as a command line argument. The script will then split the dataset
into training, validation, and test sets. The model will be fine-tuned on the
training set and validated on the validation set. The test set will be used to
evaluate the model after fine-tuning. The model will be fine-tuned for 30 epochs
by default. The number of epochs can be specified as a command line argument.

This is the main file for the project.
'''

# Import libraries
import sys
import os
import re
import json
import torch

import pandas as pd
import numpy as np

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
from datetime import datetime

# Import custom modules
from config import torgo_dataset_path, torgo_csv_path


def main():
    '''
    --------------------------------------------------------------------------------
    Check if the paths to the Torgo dataset and the Torgo dataset CSV file are valid.
    --------------------------------------------------------------------------------
    '''
    print(torgo_csv_path)
    print(torgo_dataset_path)
    if not os.path.exists(torgo_dataset_path):
        print("Please provide a valid path to the Torgo dataset in config.py.")
        sys.exit(1)

    torgo_dataset_dir_path = torgo_dataset_path + \
        '/' if torgo_dataset_path[-1] != '/' else torgo_dataset_path

    if not os.path.exists(torgo_csv_path):
        print("Please provide a valid path to the Torgo dataset CSV file in config.py.")
        sys.exit(1)

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
    List of constants
    --------------------------------------------------------------------------------
    '''
    text_count_threshold = 40
    # For each data, if the total text count across all speakers in the train and
    # validation datasets is less than the threshold and the text exists in the
    # test dataset, remove the corresponding data from the train and validation dataset.
    # Otherwise, remove the corresponding data from the 'test' dataset. This aims to
    # retain 60% to 70% of the test dataset.
    #
    # For example:
    # (1) If "The dog is brown" is spoken 30 times in total across all
    # speakers in the train and validation dataset and the phrase exists in the test
    # dataset, remove the corresponding data from the train and validation datasets.
    # (2) On the other hand, if "The dog is brown" is spoken 30 times in total across all
    # speakers in the train and validation dataset, but the phrase does not exist in
    # the test dataset, the corresponding data does not need to be removed from the
    # train and validation datasets.
    # (3) If "The dog is brown" is spoken 50 times in total across all speakers in
    # the train and validation dataset, remove the corresponding data from the test
    # dataset instead.

    # Repository name on Hugging Face
    repo_suffix = '-test'
    repo_name = f'torgo_xlsr_finetune_{test_speaker}{repo_suffix}'
    repo_path = f'macarious/{repo_name}'

    # Path to save model / checkpoints{repo_name}'
    model_save_path_local = f'/model/{repo_name}'

    # Model to be fine-tuned with Torgo dataset
    model_name = "facebook/wav2vec2-large-xlsr-53"

    '''
    --------------------------------------------------------------------------------
    Create the following directory, if it does not exist:
    - model
    - logs
    - results
    --------------------------------------------------------------------------------
    '''
    if not os.path.exists('model'):
        os.makedirs('model')

    if not os.path.exists('logs'):
        os.makedirs('logs')

    if not os.path.exists('results'):
        os.makedirs('results')

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
    train_speaker = [s for s in speakers if s not in [
        test_speaker, valid_speaker]]

    torgo_dataset = DatasetDict()
    torgo_dataset['train'] = dataset_csv['train'].filter(
        lambda x: x in train_speaker, input_columns=['speaker_id'])
    torgo_dataset['validation'] = dataset_csv['train'].filter(
        lambda x: x == valid_speaker, input_columns=['speaker_id'])
    torgo_dataset['test'] = dataset_csv['train'].filter(
        lambda x: x == test_speaker, input_columns=['speaker_id'])

    '''
    --------------------------------------------------------------------------------
    Count the number of times the text has been spoken in each of the 'train',
    'validation', and 'test' sets. Remove text according to the predetermined
    text_count_threshold.
    --------------------------------------------------------------------------------
    '''
    unique_texts = set(torgo_dataset['train'].unique(column='text')) | set(
        torgo_dataset['validation'].unique(column='text')) | set(torgo_dataset['test'].unique(column='text'))
    unique_texts_count = {}

    for text in unique_texts:
        unique_texts_count[text] = {'train_validation': 0, 'test': 0}

    for text in torgo_dataset['train']['text']:
        unique_texts_count[text]['train_validation'] += 1

    for text in torgo_dataset['validation']['text']:
        unique_texts_count[text]['train_validation'] += 1

    for text in torgo_dataset['test']['text']:
        unique_texts_count[text]['test'] += 1

    texts_to_keep_in_train_validation = []
    texts_to_keep_in_test = []
    for text in unique_texts_count:
        if unique_texts_count[text]['train_validation'] < text_count_threshold and unique_texts_count[text]['test'] > 0:
            texts_to_keep_in_test.append(text)
        else:
            texts_to_keep_in_train_validation.append(text)

    original_data_count = {'train': len(torgo_dataset['train']), 'validation': len(
        torgo_dataset['validation']), 'test': len(torgo_dataset['test'])}

    # Update the three dataset splits
    torgo_dataset['train'] = torgo_dataset['train'].filter(
        lambda x: x['text'] in texts_to_keep_in_train_validation)
    torgo_dataset['validation'] = torgo_dataset['validation'].filter(
        lambda x: x['text'] in texts_to_keep_in_train_validation)
    torgo_dataset['test'] = torgo_dataset['test'].filter(
        lambda x: x['text'] in texts_to_keep_in_test)

    print()
    print(
        f"After applying the text count threshold of {text_count_threshold}, the number of data in each dataset is:")
    print(
        f'Train:       {len(torgo_dataset["train"])}/{original_data_count["train"]} ({len(torgo_dataset["train"]) * 100 // original_data_count["train"]}%)')
    print(
        f'Validation:  {len(torgo_dataset["validation"])}/{original_data_count["validation"]} ({len(torgo_dataset["validation"]) * 100 // original_data_count["validation"]}%)')
    print(
        f'Test:        {len(torgo_dataset["test"])}/{original_data_count["test"]} ({len(torgo_dataset["test"]) * 100 // original_data_count["test"]}%)')
    print()

    '''
    --------------------------------------------------------------------------------
    Build Processor with Tokenizer and Feature Extractor
    --------------------------------------------------------------------------------
    '''
    # Remove special characters from the text
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\`\�0-9]'

    def remove_special_characters(batch):
        batch['text'] = re.sub(chars_to_ignore_regex,
                               ' ', batch['text']).lower()
        return batch

    torgo_dataset = torgo_dataset.map(remove_special_characters)

    # Create a diciontary of tokenizer vocabularies
    vocab_list = []
    for dataset in torgo_dataset.values():
        for text in dataset['text']:
            text = text.replace(' ', '|')
            vocab_list.extend(text)

    vocab_dict = {}
    vocab_dict['[PAD]'] = 0
    vocab_dict['<s>'] = 1
    vocab_dict['</s>'] = 2
    vocab_dict['[UNK]'] = 3
    vocab_list = sorted(list(set(vocab_list)))
    vocab_dict.update({v: k + len(vocab_dict)
                      for k, v in enumerate(vocab_list)})

    print()
    print("Vocab Dictionary:")
    print(vocab_dict)
    print()

    with open('./vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    # Build the tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        './vocab.json', unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')

    # Build the feature extractor
    sampling_rate = 16000
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=sampling_rate, padding_value=0.0, do_normalize=True, return_attention_mask=True)

    # Build the processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    # tokenizer.push_to_hub(repo_path)
    # feature_extractor.push_to_hub(repo_path)
    '''
    --------------------------------------------------------------------------------
    Preprocess the dataset
    - Load and resample the audio data
    - Extract values from the loaded audio file
    - Encode the transcriptions to label ids
    --------------------------------------------------------------------------------
    '''
    def prepare_torgo_dataset(batch):
        # Load audio data into batch
        audio = batch['audio']

        # Extract values
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        # Encode to label ids
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids

        return batch

    torgo_dataset = torgo_dataset.map(
        lambda x: {'audio': torgo_dataset_dir_path + x['audio']})
    torgo_dataset = torgo_dataset.cast_column(
        "audio", Audio(sampling_rate=sampling_rate))
    torgo_dataset = torgo_dataset.map(prepare_torgo_dataset, remove_columns=[
                                      'session', 'audio', 'speaker_id', 'text'], num_proc=4)

    # Filter audio within a certain length
    min_input_length_in_sec = 1.0
    max_input_length_in_sec = 10.0

    torgo_dataset = torgo_dataset.filter(
        lambda x: min_input_length_in_sec *
        sampling_rate < x < max_input_length_in_sec * sampling_rate,
        input_columns=["input_length"]
    )

    print()
    print("After filtering audio within a certain length, the number of data in each dataset is:")
    print(
        f'Train:       {len(torgo_dataset["train"])}/{original_data_count["train"]} ({len(torgo_dataset["train"]) * 100 // original_data_count["train"]}%)')
    print(
        f'Validation:  {len(torgo_dataset["validation"])}/{original_data_count["validation"]} ({len(torgo_dataset["validation"]) * 100 // original_data_count["validation"]}%)')
    print(
        f'Test:        {len(torgo_dataset["test"])}/{original_data_count["test"]} ({len(torgo_dataset["test"]) * 100 // original_data_count["test"]}%)')
    print()

    # Remove the "input_length" column
    torgo_dataset = torgo_dataset.remove_columns(["input_length"])


if __name__ == "__main__":
    main()
