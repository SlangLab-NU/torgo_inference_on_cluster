'''
Fine-tune the wave2vec model on the Torgo dataset. This script takes in the
speaker ID as a command line argument. The script will then split the dataset
into training, validation, and test sets. The model will be fine-tuned on the
training set and validated on the validation set. The test set will be used to
evaluate the model after fine-tuning. The model will be fine-tuned for 30 epochs
by default. The number of epochs can be specified as a command line argument.

This script uses a leave-one-speaker-out approach. The model will be fine-tuned
on all the speakers except the speaker specified in the command line argument.

This script accepts 2 command line arguments:
1. Speaker ID (required): e.g. F01
2. Number of epochs (optional): e.g. --epochs 30; default to 30
3. Debug mode (optional): e.g. --debug; default to False

Example usage:
python finetune.py F01
python finetune.py F01 --epochs 20
python finetune.py F01 --epochs 1 --debug

Use "python3" instead of "python" depending on your system.

In debug mode, the script will only use 20 random samples from the dataset for
debugging purposes. The dataset will be reduced from 1,000+ samples to 20. It
should take less than 5 minutes to run the script in debug mode.

This is the main file for the project.
'''

# Import libraries
import sys
import os
import re
import json
import torch
import logging
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import Dict, List, Union
from evaluate import load
from tqdm import tqdm
from datetime import datetime


def main():
    print("Start of Script\n")
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
    training_args_path = '/training_args'

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

    if not os.path.exists(training_args_path):
        print(f"""
            Please bind the training_args.json file to the container using the --bind option:
            --bind [path to training_args.json file]:/training_args.json
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
    Store the command line arguments in variables
    - speaker_id: The speaker ID to fine-tune the model on
    - num_epochs: The number of epochs to fine-tune the model for
    --------------------------------------------------------------------------------
    '''
    if len(sys.argv) < 2:
        print(
            "Please provide the speaker ID and the number of epochs (optional).")
        sys.exit(1)

    # Check if the first argument does not start with '--'
    if sys.argv[1].startswith('--'):
        print("Please provide a valid speaker ID.")
        sys.exit(1)

    test_speaker = sys.argv[1]
    num_epochs = 20
    debug_mode = False

    # Optional arguments:
    #   num of epochs (ex. --epochs 30)
    #   debug mode (ex. --debug)
    if len(sys.argv) > 2:
        for i in range(2, len(sys.argv)):
            if sys.argv[i] == '--epochs':
                if sys.argv[i+1].isdigit() and int(sys.argv[i+1]) > 0:
                    num_epochs = int(sys.argv[i+1])
                else:
                    print("Please provide a valid number of epochs.")
                    sys.exit(1)
            elif sys.argv[i] == '--debug':
                debug_mode = True
                print("Debug Mode: " + str(debug_mode))

    '''
    --------------------------------------------------------------------------------
    Set up the logging configuration
    --------------------------------------------------------------------------------
    '''
    if not os.path.exists(output_path + '/logs'):
        os.makedirs(output_path + '/logs')

    log_file_name = test_speaker + '_' + \
        datetime.now().strftime("%Y%m%d_%H%M%S") + '.log'
    log_file_path = output_path + '/logs/' + log_file_name

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
    logging.info("Number of epochs: " + str(num_epochs))
    logging.info("Log File Path: " + log_file_path + '\n')

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
    Repository name on Hugging Face and Local Path to save model / checkpoints
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
    repo_suffix = '-test' if debug_mode else ''
    repo_name = f'torgo_xlsr_finetune_{test_speaker}{repo_suffix}'
    repo_path = f'macarious/{repo_name}'

    # Path to save model / checkpoints{repo_name}'
    model_local_path = output_path + '/model/' + repo_name

    # Model to be fine-tuned with Torgo dataset
    pretrained_model_name = "facebook/wav2vec2-large-xlsr-53"

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
    ********************************************************************************
    ********************************************************************************
    DEBUG CODE
    ********************************************************************************
    ********************************************************************************
    '''
    if debug_mode:
        # Use 20 random samples from the dataset for debugging
        logging.info("--------------------------------------------")
        logging.info("DEBUG MODE")
        dataset_csv_original_size = len(dataset_csv['train'])
        dataset_csv['train'] = dataset_csv['train'].shuffle(
            seed=42).select(range(20))
        logging.info("The dataset has been reduced from " + str(dataset_csv_original_size) +
                     " to " + str(len(dataset_csv['train'])) + " for debugging.")
        logging.info("--------------------------------------------\n")
    '''
    ********************************************************************************
    ********************************************************************************
    END OF DEBUG CODE
    ********************************************************************************
    ********************************************************************************
    '''

    # Extract the unique speakers in the dataset
    speakers = data_df['speaker_id'].unique()

    logging.info("Unique speakers found in the dataset:")
    logging.info(str(speakers) + '\n')

    if test_speaker not in speakers:
        logging.error("Test Speaker not found in the dataset.")
        sys.exit(1)

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

    logging.info(
        f"After applying the text count threshold of {text_count_threshold}, the number of data in each dataset is:")
    logging.info(
        f'Train:       {len(torgo_dataset["train"])}/{original_data_count["train"]} ({len(torgo_dataset["train"]) * 100 // original_data_count["train"]}%)')
    logging.info(
        f'Validation:  {len(torgo_dataset["validation"])}/{original_data_count["validation"]} ({len(torgo_dataset["validation"]) * 100 // original_data_count["validation"]}%)')
    logging.info(
        f'Test:        {len(torgo_dataset["test"])}/{original_data_count["test"]} ({len(torgo_dataset["test"]) * 100 // original_data_count["test"]}%)\n')

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

    logging.info("Vocab Dictionary:")
    logging.info(str(vocab_dict) + '\n')

    # Create a directory to store the vocab.json file
    if not os.path.exists(output_path + '/vocab'):
        os.makedirs(output_path + '/vocab')

    vocab_file_name = repo_name + '_vocab.json'
    vocab_file_path = output_path + '/vocab/' + vocab_file_name

    # Save the vocab.json file
    with open(vocab_file_path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    # Build the tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file_path, unk_token='[UNK]', pad_token='[PAD]')

    # Build the feature extractor
    sampling_rate = 16000
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=sampling_rate, padding_value=0.0, return_attention_mask=True)

    # Build the processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Save the tokenizer and feature extractor to the repository
    tokenizer.push_to_hub(repo_path)
    feature_extractor.push_to_hub(repo_path)
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
    Define a DataCollator:
    wave2vec2 has a much larger input length as compared to the output length. For
    the input size, it is efficient to pad training batches to the longest sample
    in the batch (not overall sample)
    --------------------------------------------------------------------------------
    '''
    # Define the data collator
    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
        """
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]}
                              for feature in features]
            label_features = [{"input_ids": feature["labels"]}
                              for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )

            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels

            return batch

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True)

    '''
    --------------------------------------------------------------------------------
    Define the Evaluation Metrics
    --------------------------------------------------------------------------------
    '''
    wer_metric = load("wer")

    def compute_metrics(pred):
        """
            Compute Word Error Rate (WER) for the model predictions.

            Parameters:
            pred (transformers.file_utils.ModelOutput): Model predictions.

            Returns:
            dict: A dictionary containing the computed metrics.
        """
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -
                       100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        logging.info("Current Word Error Rate: " + str(wer))

        return {"wer": wer}

    '''
    --------------------------------------------------------------------------------
    Load the model
    --------------------------------------------------------------------------------
    '''
    # Load the model
    model = Wav2Vec2ForCTC.from_pretrained(
        pretrained_model_name,
        ctc_loss_reduction="mean",
        vocab_size=len(processor.tokenizer),
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
    )

    # Freeze the feature extractor
    # (parameters of pre-trained part of the model won't be updated during training)
    model.freeze_feature_encoder()

    # Release unoccupied cache memory
    torch.cuda.empty_cache()

    '''
    --------------------------------------------------------------------------------
    Define the Training Arguments
    --------------------------------------------------------------------------------
    '''

    # Load the training arguments from training_args.json
    with open(training_args_path + '/training_args.json') as training_args_file:
        training_args_dict = json.load(training_args_file)

    # Create the model directory, if it does not exist
    if not os.path.exists(output_path + '/model'):
        os.makedirs(output_path + '/model')

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=model_local_path,
        hub_model_id=repo_name,
        num_train_epochs=num_epochs,
        **training_args_dict
    )

    '''
    --------------------------------------------------------------------------------
    Define the Trainer
    --------------------------------------------------------------------------------
    '''
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=torgo_dataset["train"],
        eval_dataset=torgo_dataset["validation"],
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.create_model_card(
        language="en",
        tags=["audio", "speech", "wav2vec2"],
        model_name=repo_name,
        finetuned_from="facebook/wav2vec2-large-xlsr-53",
        tasks=["automatic-speech-recognition"],
        dataset="torgo",
    )

    '''
    --------------------------------------------------------------------------------
    Start Training
    --------------------------------------------------------------------------------
    '''
    logging.info("Start Training")
    logging.info("Training Arguments:")
    training_arg_log_dict = {"Training Epochs": num_epochs,
                             "Training Batch Size": training_args_dict['per_device_train_batch_size'],
                             "Evaluation Batch Size": training_args_dict['per_device_eval_batch_size'],
                             "Learning Rate": training_args_dict['learning_rate'],
                             "Weight Decay": training_args_dict['weight_decay']}
    logging.info(str(training_arg_log_dict))

    train_start_time = datetime.now()

    # Train from scratch if there is no checkpoint in the repository
    # Check if checkpoint-* directories exist in the repository
    checkpoint_files = [f for f in os.listdir(output_path + '/model/' + repo_name) if f.startswith(
        'checkpoint-') and os.path.isdir(output_path + '/model/' + repo_name + '/' + f)]
    if len(checkpoint_files) == 0:
        logging.info(
            "No checkpoint found in the repository. Training from scratch.")
        trainer.train()
    else:
        logging.info(
            f"Checkpoint found in the repository. Checkpoint files found: {checkpoint_files}")
        resume_from_checkpoint = f"{model_local_path}/{checkpoint_files[-1]}"
        logging.info(f"Resuming from checkpoint: {resume_from_checkpoint}\n")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    train_end_time = datetime.now()

    logging.info("Training completed in " +
                 str(train_end_time - train_start_time) + '\n')

    logging.info("Training Log Metrics:")
    for history in trainer.state.log_history:
        logging.info(str(history) + '\n')

    trainer.push_to_hub()
    logging.info("Model pushed to Hugging Face Hub.\n")

    '''
    --------------------------------------------------------------------------------
    Prepare for prediction and evaluation
    --------------------------------------------------------------------------------
    '''
    logging.info("Start Evaluation")

    # Create the results directory, if it does not exist
    if not os.path.exists(output_path + '/results'):
        os.makedirs(output_path + '/results')

    # Create the results directory for the current speaker, if it does not exist
    if not os.path.exists(f'{output_path}/results/{repo_name}'):
        os.makedirs(f'{output_path}/results/{repo_name}')

    results_dir = f'{output_path}/results/{repo_name}'

    # Access the model from the repository
    model = Wav2Vec2ForCTC.from_pretrained(repo_path)
    processor = Wav2Vec2Processor.from_pretrained(repo_path)

    # Move model to GPU
    if torch.cuda.is_available():
        model.to("cuda")

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

            label_ids = dataset[i]["labels"]
            reference = processor.batch_decode(label_ids, group_tokens=False)
            reference = ''.join(
                [' ' if c == '' else c for c in reference])  # remove padding

            predictions.append(prediction)
            references.append(reference)

        return predictions, references

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
    main()
