# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy all files (code + models + templates) to container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 (used by Cloud Run and App Engine)
EXPOSE 8080

# Run the app using Gunicorn server
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "app:app"]
