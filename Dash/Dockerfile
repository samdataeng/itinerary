# base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . /app

# Expose the port
EXPOSE 8050

# Set the command
CMD ["python", "app.py"]