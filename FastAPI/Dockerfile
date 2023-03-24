# Start with the official Python image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable for the port to expose
ENV PORT=8000

# Expose the port that the application will run on
EXPOSE $PORT

# Start the application
CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]
