# Fine-tune the wave2vec model on the Torgo dataset.
This README file is under construction.

## Build Docker
Run the following command in the root directory to build the the dockerfile:
`docker build -t finetune .`

# Running Docker
Run the following command to run the dockerfile:
`docker run finetune F01 --epochs 1 --debug`