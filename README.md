# Fine-tune the wave2vec model on the Torgo dataset.
This README file is under construction.

## Word Error Rates
| Speaker | Epochs | Train | Validation | Test |
|---------|--------|-------|------------|------|
| M01 (severe) | 20 | 0.0104 | 0.3198 | 0.8568 |
| M01 (severe) | 40 | 0.0035 | 0.3052 | 0.8779 |
| M02 (severe) | 20 | 0.0107 | 0.3004 | 0.9043 |
| M03 (mild)   | 20 | 0.0124 | 0.3247 | 0.4194 |
| M04 (severe) | 20 | 0.0101 | 0.2925 | 0.9332 |

## Running the Script
### Build Docker
Run the following command in the root directory to build the the dockerfile:
`docker build -t macarious/finetune .`

### Push the image (enter Docker Hub credentials when prompted, or use `docker login`)
`docker push macarious/finetune`

### Check if the image exists
`docker images`

### Push the dockerfile to Docker Hub
`docker push macarious/finetune:latest`

### Running Docker

#### Running Docker Locally
Run the following command to run the dockerfile:
`docker run finetune.py F01 --epochs 1 --debug`

#### Running Docker on the Cluster (on user_name@xfer.discovery.neu.edu)
Load singularity on the Cluster
`module load singularity/3.5.3`

Pull the docker from dockerhub
`singularity pull docker://macarious/finetune:latest`

Use GPU from Cluster
(see https://github.com/SlangLab-NU/links/wiki/Working-with-sbatch-and-srun-on-the-cluster-with-GPU-nodes)
`srun --partition=gpu --nodes=1 --gres=gpu:t4:1 --time=08:00:00 --pty /bin/bash`

Execute the image with Singularity
```
singularity run --nv --bind /work/van-speech-nlp/data/torgo:/torgo_dataset,/work/van-speech-nlp/hui.mac/torgo_inference_on_cluster:/output,/work/van-speech-nlp/hui.mac/torgo_inference_on_cluster:/training_args --pwd /scripts /work/van-speech-nlp/hui.mac/finetune_latest.sif /bin/bash
```

Log in to Hugging Face
`huggingface-cli login`

Run the script
Example: `python3 finetune.py M03 --epochs 40`
Example: `python3 finetune.py M01`
