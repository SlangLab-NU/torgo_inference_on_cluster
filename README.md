# Fine-tune the wave2vec model on the Torgo dataset.
This README file is under construction.

## Word Error Rates Summary
| Speaker | Epochs | Train | Validation | Test | Test (trained and evaluated with all text) |
|---------|--------|-------|------------|------|-----------------|
| M01 (severe) | 20 | 0.0104 | 0.3198 | 0.8568 | 0.4072 |
| M01 (severe) | 40 | 0.0035 | 0.3052 | 0.8779 |  |
| M02 (severe) | 20 | 0.0107 | 0.3004 | 0.9043 |  |
| M03 (mild)   | 20 | 0.0124 | 0.3247 | 0.4194 |  |
| M04 (severe) | 20 | 0.0101 | 0.2925 | 0.9332 |  |

## Building Docker on Local Machine

Run the following command in the root directory to build the the dockerfile:

`docker build -t macarious/finetune .`

Push the dockerfile to Docker Hub:

`docker push macarious/finetune:latest`

## Running Script from Docker on the Cluster

### on `user_name@xfer.discovery.neu.edu`:

#### 1. Load singularity on the Cluster:

`module load singularity/3.5.3`

#### 2. Pull the docker from dockerhub (on user_name@xfer.discovery.neu.edu):

`singularity pull docker://macarious/finetune:latest`

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

Example: `python3 train.py M01`

Example: `python3 train.py M01 --repeated_text_threshold 1 --repo_suffix keep_all`

Example: `python3 predict_and_evaluate.py M01`

Example: `python3 predict_and_evaluate.py M01 --keep_all_text True`

Example: `python3 predict_and_evaluate.py M01 --keep_all_text True --repo_suffix keep_all`


#### 6. Clear cache cache in cluster if it is full:

`rm -rf /home/hui.mac/.cache/`

`rm -rf /home/hui.mac/.singularity/cache`

## The Training Script: `train.py`
Fine-tune the wave2vec model on the Torgo dataset. This script takes in the
speaker ID as a command line argument. The script will then split the dataset
into training, validation, and test sets. The model will be fine-tuned on the
training set and validated on the validation set. The test set will be used to
evaluate the model after fine-tuning. The model will be fine-tuned for 20 epochs
by default. The number of epochs can be specified as a command line argument.

This script uses a leave-one-speaker-out approach. The model will be fine-tuned
on all the speakers except the speaker specified in the command line argument.

This script accepts the following arguments:
```
    positional arguments:
    speaker_id            Speaker ID in the format [MF]C?[0-9]{2}

    options:
    -h, --help            show this help message and exit
    --learning_rate LEARNING_RATE
                            Learning rate (default: 0.0001)
    --train_batch_size TRAIN_BATCH_SIZE
                            Training batch size (default: 4)
    --eval_batch_size EVAL_BATCH_SIZE
                            Evaluation batch size (default: 4)
    --seed SEED           Random seed (default: 42)
    --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                            Gradient accumulation steps (default: 2)
    --total_train_batch_size TOTAL_TRAIN_BATCH_SIZE
                            Total training batch size (default: 8)
    --optimizer OPTIMIZER
                            Optimizer type (default: Adam)
    --lr_scheduler_type LR_SCHEDULER_TYPE
                            Learning rate scheduler type (default: linear)
    --lr_scheduler_warmup_steps LR_SCHEDULER_WARMUP_STEPS
                            Learning rate scheduler warmup steps (default: 1000)
    --num_epochs NUM_EPOCHS
                            Number of epochs (default: 20)
    --repeated_text_threshold REPEATED_TEXT_THRESHOLD
                            Repeated text threshold (default: 40)
    --debug               Enable debug mode
    --repo_suffix REPO_SUFFIX
                            Repository suffix
```

Example usage:
`python train.py F01`
`python train.py F01 --num_epochs 1 --debug

Use `python3` instead of `python` depending on your system.

In debug mode, the script will only use 20 random samples from the dataset for
debugging purposes. The dataset will be reduced from 1,000+ samples to 20. It
should take less than 5 minutes to run the script in debug mode.

## The Prediction and Evaluation Script: `predict_and_evaluate.py`