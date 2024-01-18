# This file is used to build the docker image for the project
# Author: Macarious Hui
FROM pytorch/pytorch

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN rm -rf /root/.cache/pip

# Set environment variables
ENV HF_ACCESS_TOKEN='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'

# Expose port 5000
EXPOSE 5000

# Define the command to be run when launching the container
ENTRYPOINT ["python", "finetune.py"]

# Default arguments if not specified
CMD ["F01", "--epochs", "30", "--debug"]