FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/nltk_download.py .
RUN mkdir -p /usr/local/nltk_data && \
    python nltk_download.py /usr/local/nltk_data && \
    rm nltk_download.py
ENV NLTK_DATA=/usr/local/nltk_data

COPY src/ ./src/
COPY App.py .
COPY templates/ ./templates/

COPY deploy/model/ ./deploy/model/

ENV PORT=5000
EXPOSE 5000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 App:app"]
