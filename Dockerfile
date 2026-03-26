FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python datasets.py --eta 0 0.05 0.1 0.2 0.5

EXPOSE 7860

CMD ["python", "app.py"]
