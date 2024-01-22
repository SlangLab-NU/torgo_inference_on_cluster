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
`docker run finetune F01 --epochs 1 --debug`

### Running Docker on the Cluster
Load singularity on the Cluster
`module load singularity/3.5.3`

`singularity pull docker://macarious/finetune:latest`