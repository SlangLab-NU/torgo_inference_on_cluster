# Fine-tune the wave2vec model on the Torgo dataset.
This README file is under construction.

## Build Docker
Run the following command in the root directory to build the the dockerfile:
`docker build -t [docker_account_name]/finetune .`

## Push the image (enter Docker Hub credentials when prompted, or use `docker login`)
`docker push [docker_account_name]/finetune`

## Check if the image exists
`docker images`

## Push the dockerfile to Docker Hub
`docker push [docker_account_name]/finetune:latest`

## Running Docker

### Running Docker Locally
Run the following command to run the dockerfile:
`docker run finetune.py F01 --epochs 1 --debug`

### Running Docker on the Cluster
Load singularity on the Cluster
`module load singularity/3.5.3`

`singularity pull docker://[docker_account_name]/finetune:latest`

### Running GPU jobs
(see https://github.com/SlangLab-NU/links/wiki/Working-with-sbatch-and-srun-on-the-cluster-with-GPU-nodes)
`srun --partition=gpu --nodes=1 --gres=gpu:t4:1 --time=08:00:00 --pty /bin/bash`

### Define the image
`singularity_image=/work/van-speech-nlp/hui.mac/finetune_latest.sif`

`SINGULARITYENV_HF_ACCESS_TOKEN=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`

### Execute the image
```
singularity run --nv --bind /work/van-speech-nlp/data/torgo:/torgo_dataset_path --env HF_ACCESS_TOKEN=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 $singularity_image \
python3 finetune.py F01 --epochs 1 --debug
```

```
singularity run --nv --bind /work/van-speech-nlp/data/torgo:/torgo_dataset_path,/work/van-speech-nlp/hui.mac:/output_path --pwd /scripts $singularity_image \
python3 finetune.py F01 --epochs 1 --debug
```

```
singularity run --nv --bind /work/van-speech-nlp/data/torgo:/torgo_dataset_path --pwd /scripts $singularity_image /bin/bash
```