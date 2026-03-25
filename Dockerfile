FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Pre-generate datasets so the app starts instantly
# Adjust eta values / dims as needed
RUN python datasets.py --eta 0 0.05 0.1 0.2 0.5

EXPOSE 7860

CMD ["python", "app.py"]
