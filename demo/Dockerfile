FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import nltk; nltk.download('vader_lexicon', download_dir='/nltk_data')"

COPY app.py .

ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/nltk_data

EXPOSE 7860

CMD ["python", "app.py"]
