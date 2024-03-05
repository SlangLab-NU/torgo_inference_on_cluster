'''
This script is used to evaluate the performance of the model. It will be called by the main.py script.
It can also be called by the user separately to evaluate the performance of the model on a given dataset.
The repository name on Hugging Face is in the format torgo_xlsr_finetune_[speaker_id][repo_suffix].
It outputs the Word Error Rate (WER) for the training, validation, and test sets, and saves the predictions
and references to CSV files. It also saves a summary of the Word Error Rates to a CSV file.

Positional Arguments
speaker_id	Speaker ID in the format [MF]C?[0-9]{2}

Options	Descriptions
-h, --help	show this help message and exit
--keep_all_data	Keep all text or only repeated text (default: False)
--repo_suffix	Repository suffix
'''

import os
import sys
import argparse
import torch
import re
import logging
import pandas as pd

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, DatasetDict, Audio
from evaluate import load
from tqdm import tqdm
from datetime import datetime


def predict_and_evaluate():
    print('Predict and Evaluate')
    '''
    --------------------------------------------------------------------------------
    Store the command line arguments in variables
    Possible arguments:
    - Speaker ID: Speaker ID in the format [MF]C?[0-9]{2}
    Optional arguments:
    --keep_all_data: Keep all text or only repeated text (default: False)
    --repo_suffix: Repository suffix
    --------------------------------------------------------------------------------
    '''
    parser = argparse.ArgumentParser(
        description='Fine-tune the model on a specified speaker ID.')

    # Required argument: speaker ID
    parser.add_argument('speaker_id', type=str,
                        help='Speaker ID in the format [MF]C?[0-9]{2}')

    # Optional arguments with default values
    parser.add_argument('--keep_all_data', action='store_true',
                        help='Keep all text or only repeated text')
    parser.add_argument('--repo_suffix', type=str,
                        default='', help='Repository suffix')

    args = parser.parse_args()

    # Check if the speaker ID is valid
    if not re.match(r'^[MF]C?[0-9]{2}$', args.speaker_id):
        print("Please provide a valid speaker ID.")
        sys.exit(1)
    test_speaker = args.speaker_id

    # Accessing optional arguments
    keep_all_data = args.keep_all_data
    repo_suffix = args.repo_suffix
    if args.repo_suffix and not re.match(r'^[_-]', args.repo_suffix):
        repo_suffix = '_' + args.repo_suffix

    '''
    --------------------------------------------------------------------------------
    Check if the paths to the Torgo dataset and the Torgo dataset CSV file are valid.
    --------------------------------------------------------------------------------
    '''
    # Saved in the same directory as this script (inside container)
    torgo_csv_path = "./torgo.csv"

    # Use --bind option to save to a different directory when running on Cluster
    torgo_dataset_path = '/torgo_dataset'
    output_path = '/output'

    if not os.path.exists(torgo_dataset_path):
        print(f"""
            Please bind the Torgo dataset directory to the container using the --bind option:
            --bind [path to Torgo dataset directory]:/torgo_dataset
            """)
        sys.exit(1)

    if not os.path.exists(output_path):
        print(f"""
            Please bind the output directory to the container using the --bind option:
            --bind [path to output directory]:/output
            """)
        sys.exit(1)

    torgo_dataset_dir_path = torgo_dataset_path + \
        '/' if torgo_dataset_path[-1] != '/' else torgo_dataset_path

    if not os.path.exists(torgo_csv_path):
        print(
            "Error loading the Torgo dataset CSV file. Please make sure that the Torgo dataset CSV file is in the same directory as this script.")
        sys.exit(1)

    '''
    --------------------------------------------------------------------------------
    Repository name on Hugging Face
    --------------------------------------------------------------------------------
    '''
    # Repository name on Hugging Face
    repo_name = f'torgo_xlsr_finetune_{test_speaker}{repo_suffix}'
    repo_path = f'macarious/{repo_name}'

    '''
    --------------------------------------------------------------------------------
    Set up the logging configuration
    --------------------------------------------------------------------------------
    '''
    if not os.path.exists(output_path + '/logs'):
        os.makedirs(output_path + '/logs')

    log_dir = f'{output_path}/logs/{repo_name}'

    # Create the results directory for the current speaker, if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_name = test_speaker + '_evaluate' + '_' + \
        datetime.now().strftime("%Y%m%d_%H%M%S") + '.log'
    log_file_path = log_dir + '/' + log_file_name

    logging.basicConfig(
        filename=log_file_path,
        filemode='a',
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        level=logging.INFO
    )

    # Log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    logging.info("Test Speaker: " + test_speaker)
    logging.info("Log File Path: " + log_file_path + '\n')
    if keep_all_data:
        logging.info("Keep all data in training/validation/test sets\n")

    '''
    --------------------------------------------------------------------------------
    Read the Torgo dataset CSV file and store the data in a dictionary.
    --------------------------------------------------------------------------------
    '''
    data_df = pd.read_csv(torgo_csv_path)
    dataset_csv = load_dataset('csv', data_files=torgo_csv_path)

    # Check if the following columns exist in the dataset ['session', 'audio', 'text', 'speaker_id']
    expected_columns = ['session', 'audio', 'text', 'speaker_id']
    not_found_columns = []
    for column in expected_columns:
        if column not in dataset_csv['train'].column_names:
            not_found_columns.append(column)

    if len(not_found_columns) > 0:
        logging.error(
            "The following columns are not found in the dataset:" + " [" + ", ".join(not_found_columns) + "]")
        sys.exit(1)

    '''
    --------------------------------------------------------------------------------
    Use GPU if available
    --------------------------------------------------------------------------------
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using GPU: " + torch.cuda.get_device_name(0) + '\n')

    else:
        device = torch.device("cpu")
        logging.info("Using CPU\n")

    '''
    --------------------------------------------------------------------------------
    Load the processor and the pretrained model
    --------------------------------------------------------------------------------
    '''
    # Access the model from the repository
    logging.info("Loading the model and processor from " + repo_path)
    model = Wav2Vec2ForCTC.from_pretrained(repo_path)
    logging.info("Model loaded from " + repo_path)
    processor = Wav2Vec2Processor.from_pretrained(repo_path)
    logging.info("Processor loaded from " + repo_path + '\n')

    # Move model to GPU
    if torch.cuda.is_available():
        model.to("cuda")

    '''
    --------------------------------------------------------------------------------
    Split the dataset into training / validation / test sets.
    --------------------------------------------------------------------------------
    '''
    logging.info(
        "Splitting the dataset into training / validation / test sets...")

    # Extract the unique speakers in the dataset
    speakers = data_df['speaker_id'].unique()

    logging.info("Unique speakers found in the dataset:")
    logging.info(str(speakers) + '\n')

    if test_speaker not in speakers:
        logging.error("Test Speaker not found in the dataset.")
        sys.exit(1)

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
    Remove text according to the overlapped-prompt-removal protocol
    --------------------------------------------------------------------------------
    '''
    original_data_count = {'train': len(torgo_dataset['train']), 'validation': len(
        torgo_dataset['validation']), 'test': len(torgo_dataset['test'])}

    if not keep_all_data:

        # Update the three dataset splits (if ['test_data'] == 1, keep in test, if ['test_data'] == 0, keep in train and validation)
        torgo_dataset['train'] = torgo_dataset['train'].filter(
            lambda x: x['test_data'] == 0)
        torgo_dataset['validation'] = torgo_dataset['validation'].filter(
            lambda x: x['test_data'] == 0)
        torgo_dataset['test'] = torgo_dataset['test'].filter(
            lambda x: x['test_data'] == 1)

        # Drop the 'test_data' column
        torgo_dataset['train'] = torgo_dataset['train'].remove_columns([
                                                                       'test_data'])
        torgo_dataset['validation'] = torgo_dataset['validation'].remove_columns([
                                                                                 'test_data'])
        torgo_dataset['test'] = torgo_dataset['test'].remove_columns([
                                                                     'test_data'])

        logging.info(
            f"After removal of repeated prompts, the number of data in each dataset is:")
        logging.info(
            f'Train:       {len(torgo_dataset["train"])}/{original_data_count["train"]} ({len(torgo_dataset["train"]) * 100 // original_data_count["train"]}%)')
        logging.info(
            f'Validation:  {len(torgo_dataset["validation"])}/{original_data_count["validation"]} ({len(torgo_dataset["validation"]) * 100 // original_data_count["validation"]}%)')
        logging.info(
            f'Test:        {len(torgo_dataset["test"])}/{original_data_count["test"]} ({len(torgo_dataset["test"]) * 100 // original_data_count["test"]}%)\n')

    '''
    --------------------------------------------------------------------------------
    Preprocess the dataset
    - Load and resample the audio data
    - Extract values from the loaded audio file
    - Encode the transcriptions to label ids
    --------------------------------------------------------------------------------
    '''
    sampling_rate = 16000

    def prepare_torgo_dataset(batch):
        # Load audio data into batch
        audio = batch['audio']

        # Extract values
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        return batch

    torgo_dataset = torgo_dataset.map(
        lambda x: {'audio': torgo_dataset_dir_path + x['audio']})
    torgo_dataset = torgo_dataset.cast_column(
        "audio", Audio(sampling_rate=sampling_rate))
    torgo_dataset = torgo_dataset.map(prepare_torgo_dataset, remove_columns=[
                                      'session', 'audio', 'speaker_id'], num_proc=4)

    # Filter audio within a certain length
    min_input_length_in_sec = 1.0
    max_input_length_in_sec = 10.0

    torgo_dataset = torgo_dataset.filter(
        lambda x: min_input_length_in_sec *
        sampling_rate < x < max_input_length_in_sec * sampling_rate,
        input_columns=["input_length"]
    )

    logging.info(
        "After filtering audio within a certain length, the number of data in each dataset is:")

    if original_data_count['train'] != 0:
        logging.info(
            f'Train:       {len(torgo_dataset["train"])}/{original_data_count["train"]} ({len(torgo_dataset["train"]) * 100 // original_data_count["train"]}%)')
    else:
        logging.info(f'Train:       {len(torgo_dataset["train"])}/0 (0%)')

    if original_data_count['validation'] != 0:
        logging.info(
            f'Validation:  {len(torgo_dataset["validation"])}/{original_data_count["validation"]} ({len(torgo_dataset["validation"]) * 100 // original_data_count["validation"]}%)')
    else:
        logging.info(
            f'Validation:  {len(torgo_dataset["validation"])}/0 (0%)')

    if original_data_count['test'] != 0:
        logging.info(
            f'Test:        {len(torgo_dataset["test"])}/{original_data_count["test"]} ({len(torgo_dataset["test"]) * 100 // original_data_count["test"]}%)\n')
    else:
        logging.info(f'Test:        {len(torgo_dataset["test"])}/0 (0%)\n')

    # Remove the "input_length" column
    torgo_dataset = torgo_dataset.remove_columns(["input_length"])

    '''
    --------------------------------------------------------------------------------
    Prepare for prediction and evaluation
    --------------------------------------------------------------------------------
    '''
    logging.info("Start Evaluation")

    # Create the results directory, if it does not exist
    if not os.path.exists(output_path + '/results'):
        os.makedirs(output_path + '/results')

    results_dir = f'{output_path}/results/{repo_name}'

    # Create the results directory for the current speaker, if it does not exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    def predict_dataset(dataset):
        '''
        Predict on the dataset

        Parameters:
        dataset (datasets.Dataset): Dataset to predict on

        Returns:
        predictions (list): List of predictions
        references (list): List of references
        '''

        predictions = []
        references = []

        for i in tqdm(range(dataset.num_rows)):
            inputs = processor(
                dataset[i]["input_values"], sampling_rate=sampling_rate, return_tensors="pt", padding=True)

            # Move input to GPU
            if torch.cuda.is_available():
                inputs = {key: val.to("cuda") for key, val in inputs.items()}

            # Predict
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            prediction = processor.batch_decode(predicted_ids)[0].lower()

            reference = dataset[i]["text"].lower()

            predictions.append(prediction)
            references.append(reference)

        return predictions, references

    # Load the WER metric
    wer_metric = load("wer")

    '''
    --------------------------------------------------------------------------------
    Predict and evaluate on the training set
    --------------------------------------------------------------------------------
    '''
    # Predict on the training set
    logging.info("Predicting on the training set...")
    train_predictions, train_references = predict_dataset(
        torgo_dataset['train'])
    train_wer = wer_metric.compute(
        predictions=train_predictions, references=train_references)
    logging.info("Word Error Rate: " + str(train_wer))

    # Save the predictions and references to a CSV file
    train_df = pd.DataFrame(
        {'predictions': train_predictions, 'references': train_references})
    train_df.to_csv(
        f'{results_dir}/{test_speaker}_predictions_train.csv', index=False)
    logging.info("Predictions saved to: " +
                 f'{results_dir}/{test_speaker}_predictions_train.csv' + '\n')

    '''
    --------------------------------------------------------------------------------
    Predict and evaluate on the validation set
    --------------------------------------------------------------------------------
    '''
    # Predict on the validation set
    logging.info("Predicting on the validation set...")
    validation_predictions, validation_references = predict_dataset(
        torgo_dataset['validation'])
    validation_wer = wer_metric.compute(
        predictions=validation_predictions, references=validation_references)
    logging.info("Word Error Rate: " +
                 str(validation_wer))

    # Save the predictions and references to a CSV file
    validation_df = pd.DataFrame(
        {'predictions': validation_predictions, 'references': validation_references})
    validation_df.to_csv(
        f'{results_dir}/{test_speaker}_predictions_validation.csv', index=False)
    logging.info("Predictions saved to: " +
                 f'{results_dir}/{test_speaker}_predictions_validation.csv' + '\n')

    '''
    --------------------------------------------------------------------------------
    Predict and evaluate on the test set
    --------------------------------------------------------------------------------
    '''
    # Predict on the test set
    logging.info("Predicting on the test set...")
    test_predictions, test_references = predict_dataset(torgo_dataset['test'])

    test_wer = wer_metric.compute(
        predictions=test_predictions, references=test_references)
    logging.info("Word Error Rate: " + str(test_wer))

    # Save the predictions and references to a CSV file
    test_df = pd.DataFrame(
        {'predictions': test_predictions, 'references': test_references})
    test_df.to_csv(
        f'{results_dir}/{test_speaker}_predictions_test.csv', index=False)
    logging.info("Predictions saved to: " +
                 f'{results_dir}/{test_speaker}_predictions_test.csv' + '\n')

    '''
    --------------------------------------------------------------------------------
    Summarize the Word Error Rates
    --------------------------------------------------------------------------------
    '''
    # Save the summary to a CSV file
    summary_df = pd.DataFrame({'Split': ['train', 'validation', 'test'], 'WER': [
                              train_wer, validation_wer, test_wer]})
    summary_df.to_csv(
        f'{results_dir}/{test_speaker}_wer_summary.csv', index=False)
    logging.info("Summary of Word Error Rates saved to " +
                 f'{results_dir}/{test_speaker}_wer_summary.csv' + '\n')

    '''
    --------------------------------------------------------------------------------
    End of Script
    --------------------------------------------------------------------------------
    '''

    logging.info("End of Script")
    logging.info("--------------------------------------------\n")


if __name__ == "__main__":
    predict_and_evaluate()
