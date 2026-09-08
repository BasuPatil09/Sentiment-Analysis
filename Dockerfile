FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /usr/local/nltk_data && \
    python -m nltk.downloader -d /usr/local/nltk_data \
        stopwords wordnet punkt punkt_tab omw-1.4
ENV NLTK_DATA=/usr/local/nltk_data

COPY src/ ./src/
COPY App.py .
COPY templates/ ./templates/

COPY deploy/model/ ./deploy/model/

ENV PORT=5000
EXPOSE 5000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 2 --timeout 120 App:app"]
