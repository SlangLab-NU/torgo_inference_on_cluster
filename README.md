# Fine-tune the wave2vec model on the Torgo dataset.

This README file is under construction.

## Number of Audio Data per Speaker used in Training/Validation/Testing

### No Repeated Data between the Train/Validation Sets and the Test Set
Some prompts are left out in order to avoid having prompt overlaps between train/validation set and test set. The target speakers refer to the data in the test set.

For each data, if the total text count across all speakers in the train and validation datasets is less than the threshold and the text exists in the test dataset, remove the corresponding data from the train and validation dataset. Otherwise, remove the corresponding data from the 'test' dataset. This aims to retain 60% to 70% of the test dataset.

For example:

Using the default value of 40 for the text_count_threshold:

1. If "The dog is brown" is spoken 30 times in total across all speakers in the train and validation dataset and the phrase exists in the test
dataset, remove the corresponding data from the train and validation datasets.

2. On the other hand, if "The dog is brown" is spoken 30 times in total across all speakers in the train and validation dataset, but the phrase does not exist in the test dataset, the corresponding data does not need to be removed from the train and validation datasets.

3. If "The dog is brown" is spoken 50 times in total across all speakers in the train and validation dataset, remove the corresponding data from the test dataset instead.

Statistics on the effect of the threshold value can be found here for each scenario: https://drive.google.com/drive/folders/1bQw87pVD63N-T_pS_yJo7UOrmVJFa5fY?usp=sharing

| Speaker      | Train  | Validation | Test   |
| ------------ | ------ | ---------- | ------ |
| F01 (severe) | 13374  | 913        | 122    |
| F03 (mild)   | 7567   | 245        | 756    |
| F04 (mild)   | 8457   | 515        | 482    |
| M01 (severe) | 8937   | 560        | 489    |
| M02 (severe) | 8738   | 542        | 510    |
| M03 (mild)   | 8479   | 520        | 573    |
| M04 (severe) | 10676  | 706        | 369    |
| M05 (sev/mid)| 8071   | 454        | 328    |

### All Data are Retained in the Train/Validation/Test Sets

| Speaker      | Train  | Validation | Test   |
| ------------ | ------ | ---------- | ------ |
| F01 (severe) | 14854  | 1017       | 211    |
| F03 (mild)   | 14408  | 657        | 1017   |
| F04 (mild)   | 14408  | 1017       | 657    |
| M01 (severe) | 14356  | 1017       | 709    |
| M02 (severe) | 14330  | 1017       | 735    |
| M03 (mild)   | 14265  | 1017       | 800    |
| M04 (severe) | 14487  | 1017       | 578    |
| M05 (sev/mid)| 14555  | 510        | 510    |

## Word Error Rates Summary

### No Repeated Data between the Train/Validation Sets and the Test Set

| Speaker      | Epochs | Train  | Validation | Test   |
| ------------ | ------ | ------ | ---------- | ------ |
| F01 (severe) | 20     | 0.0143 | 0.2408     | 0.7871 |
| F03 (mild)   | 20     | 0.0189 | 0.1319     | 0.6930 |
| F04 (mild)   | 20     | 0.0145 | 0.3310     | 0.4039 |
| M01 (severe) | 20     | 0.0104 | 0.3198     | 0.8568 |
| M02 (severe) | 20     | 0.0107 | 0.3004     | 0.9043 |
| M03 (mild)   | 20     | 0.0124 | 0.3247     | 0.4194 |
| M04 (severe) | 20     | 0.0101 | 0.2925     | 0.9332 |
| M05 (sev/mid)| 20     | 0.0157 | 0.3589     | 0.9191 |

M01 was also tested with 40 epochs.

| Speaker      | Epochs | Train  | Validation | Test   |
| ------------ | ------ | ------ | ---------- | ------ |
| M01 (severe) | 40     | 0.0035 | 0.3052     | 0.8779 |

### All Data are Retained in the Train/Validation/Test Sets

| Speaker      | Epochs | Train  | Validation | Test   |
| ------------ | ------ | ------ | ---------- | ------ |
| F01 (severe) | 20     | 0.0124 | 0.2329     | 0.4645 |
| F03 (mild)   | 20     | 0.0135 | 0.0450     | 0.2581 |
| F04 (mild)   | 20     | 0.0137 | 0.2386     | 0.0492 |
| M01 (severe) | 20     | 0.0126 | 0.2474     | 0.4072 |
| M02 (severe) | 20     | 0.0124 | 0.2463     | 0.5440 |
| M03 (mild)   | 20     | 0.0129 | 0.2375     | 0.0317 |
| M04 (severe) | 20     | 0.0115 | 0.2318     | 0.6450 |
| M05 (sev/mid)| 20     | 0.0125 | 0.2245     | 0.5104 |

## Building Docker on Local Machine

Run the following command in the root directory to build the dockerfile:

`docker build -t macarious/finetune .`

Push the dockerfile to Docker Hub:

`docker push macarious/finetune:latest`

## Running Script from Docker on the Cluster

### on `user_name@xfer.discovery.neu.edu`:

#### 1. Load singularity on the Cluster:

`module load singularity/3.5.3`

#### 2. Pull the docker from dockerhub (on user_name@xfer.discovery.neu.edu):

`singularity pull docker://macarious/finetune:latest`

This pulls docker file and builds the image `finetune.sif`.

### on `user_name@login.discovery.neu.edu`:

#### 1. Use GPU from Cluster (on user_name@xfer.discovery.neu.edu):

(see https://github.com/SlangLab-NU/links/wiki/Working-with-sbatch-and-srun-on-the-cluster-with-GPU-nodes)

`srun --partition=gpu --nodes=1 --gres=gpu:t4:1 --time=08:00:00 --pty /bin/bash`

#### 2. Load singularity on the Cluster:

`module load singularity/3.5.3`

#### 3. Execute the image with Singularity:

```
singularity run --nv --bind /work/van-speech-nlp/data/torgo:/torgo_dataset,/work/van-speech-nlp/hui.mac/torgo_inference_on_cluster:/output,/work/van-speech-nlp/hui.mac/torgo_inference_on_cluster:/training_args --pwd /scripts /work/van-speech-nlp/hui.mac/finetune_latest.sif /bin/bash
```

#### 4. Log in to Hugging Face:

`huggingface-cli login`

#### 5. Run the scripts:

Example: `python3 train.py F03`

Example: `python3 train.py F03 --keep_all_data --repo_suffix _keep_all`

Example: `python3 predict_and_evaluate.py F03`

Example: `python3 predict_and_evaluate.py F03 --keep_all_data`

Example: `python3 predict_and_evaluate.py F03 --keep_all_data --repo_suffix _keep_all`

#### 6. Clear cache cache in cluster if it is full:

`rm -rf /home/hui.mac/.cache/`

`rm -rf /home/hui.mac/.singularity/cache`

## The Training Script: `train.py`

Fine-tune the wave2vec model on the Torgo dataset. This script takes in the
speaker ID as a command line argument. The script will then split the dataset
into training, validation, and test sets. The model will be fine-tuned on the
training set and validated on the validation set.

This script uses a leave-one-speaker-out approach. The model will be fine-tuned
on all the speakers except the speaker specified in the command line argument.

The number of epochs and other training parameters can be adjusted using the optional
arguments. The model will be fine-tuned for 20 epochs by default. The model will
be saved to Huggingface with the repository name:

`torgo_xlsr_finetune_[speaker_id][repo_suffix]`

This script accepts the following arguments:

| Positional Arguments | Descriptions                            |
| -------------------- | --------------------------------------- |
| speaker_id           | Speaker ID in the format [MF]C?[0-9]{2} |

| Options                         | Descriptions                                                         |
| ------------------------------- | -------------------------------------------------------------------- |
| `-h, --help`                    | show this help message and exit                                      |
| `--learning_rate`               | Learning rate (default: 0.0001)                                      |
| `--train_batch_size`            | Training batch size (default: 4)                                     |
| `--eval_batch_size`             | Evaluation batch size (default: 4)                                   |
| `--seed`                        | Random seed (default: 42)                                            |
| `--gradient_accumulation_steps` | Gradient accumulation steps (default: 2)                             |
| `--optimizer`                   | Optimizer type (default: adamw_torch)                                |
| `--lr_scheduler_type`           | Learning rate scheduler type (default: linear)                       |
| `--num_epochs`                  | Number of epochs (default: 20)                                       |
| `--repeated_text_threshold`     | Repeated text threshold (default: 40)                                |
| `--keep_all_data`               | Keep all data in the test set; overrides the repeated_text_threshold |
| `--debug`                       | Enable debug mode                                                    |
| `--repo_suffix`                 | Repository suffix                                                    |

Example usage:
`python train.py F01`
`python train.py F01 --num_epochs 1 --debug`

Use `python3` instead of `python` depending on your system.

In debug mode, the script will only use 20 random samples from the dataset for
debugging purposes. The dataset will be reduced from 1,000+ samples to 20. It
should take less than 5 minutes to run the script in debug mode.

The model can be evaluated using predict_and_evaluate.py afterwards.

## The Prediction and Evaluation Script: `predict_and_evaluate.py`

This script is used to evaluate the performance of the model. It will be called by the main.py script.
It can also be called by the user separately to evaluate the performance of the model on a given dataset.
The repository name on Hugging Face is in the format:

`torgo_xlsr_finetune_[speaker_id][repo_suffix]`

It outputs the Word Error Rate (WER) for the training, validation, and test sets, and saves the predictions
and references to CSV files. It also saves a summary of the Word Error Rates to a CSV file.

This script accepts the following arguments:

| Positional Arguments | Descriptions                            |
| -------------------- | --------------------------------------- |
| speaker_id           | Speaker ID in the format [MF]C?[0-9]{2} |

| Options                     | Descriptions                                                               |
| --------------------------- | -------------------------------------------------------------------------- |
| `-h, --help`                | show this help message and exit                                            |
| `--repeated_text_threshold` | Threshold for repeated text filtering (default: 40)                        |
| `--keep_all_data`           | Keep all text or only repeated text; overrides the repeated_text_threshold |
| `--repo_suffix`             | Repository suffix                                                          |
