# This file is used to build the docker image for the project
# Author: Macarious Hui

# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch

# Set working directory
WORKDIR /scripts

# Copy the current directory contents into the container at /app
COPY . /scripts

# Install Git and Git LFS
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y git-lfs && \
    git lfs install

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN rm -rf /root/.cache/pip

# Expose port 5000
EXPOSE 5000