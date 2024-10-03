# Use a specific Python 3.9 version with the 'bookworm' tag
FROM python:3.12.1-slim

# Ensures that Python output is displayed in real-time
ENV PYTHONUNBUFFERED True

# Set the working directory in the container
ENV APP_HOME /back-end
WORKDIR $APP_HOME

# Copy all the files from the current directory to the container's working directory
COPY . ./

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install the dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support from the official PyTorch repository
# RUN pip install --no-cache-dir \
#     torch==2.3.1+cu118 \
#     torchvision==0.18.1+cu118 \
#     -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Command to run the application using gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 1800 app:app
